import argparse
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from spektral.layers import GATConv
from tensorflow.keras import Sequential

from utils.replay_buffer_entr import ReplayBuffer
from utils.util import Utility


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=4, help="number of agents")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--no-episodes", type=int, default=50000, help="number of episodes")
    parser.add_argument("--no-neighbors", type=int, default=2, help="number of neigbors to cooperate")
    parser.add_argument("--seed", type=int, default=1, help="seed")

    # Experience Replay
    parser.add_argument("--max-buffer-size", type=int, default=500000, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--soft-update", type=bool, default=True, help="Mode of updating the target network")
    parser.add_argument("--loss-type", type=str, default="huber", help="Loss function: huber or mse")

    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon exploration")
    parser.add_argument("--epsilon-decay", type=float, default=0.0003, help="epsilon decay")
    parser.add_argument("--min-epsilon", type=float, default=0.01, help="min epsilon")
    parser.add_argument("--max-epsilon", type=float, default=1.0, help="max epsilon")

    # GNN training parameters
    parser.add_argument("--no-neurons", type=int, default=32, help="number of neurons on the first gnn")
    parser.add_argument("--l2-reg", type=float, default=2.5e-4, help="kernel regularizer")

    # Q-learning training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

    # Evaluation
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--exp-name", type=str, default='gat-entr', help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=50,
                        help="save model once every time this many episodes are completed")
    return parser.parse_args()


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(no_agents=arglist.no_agents)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def graph_net(arglist):
    I1 = Input(shape=(no_agents, no_features), name="graph_input")
    Adj = Input(shape=(no_agents, no_agents), name="adj")
    gat = GATConv(
        arglist.no_neurons,
        activation='elu',
        attn_heads=2,
        concat_heads=True,
    )([I1, Adj])

    dense = Dense(arglist.no_neurons,
                  kernel_initializer=tf.keras.initializers.he_uniform(),
                  activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                  name="dense_layer")

    last_dense = Dense(no_actions, kernel_initializer=tf.keras.initializers.he_uniform(),
                       name="last_dense_layer")
    split = Lambda(lambda x: tf.squeeze(tf.split(x, num_or_size_splits=no_agents, axis=1), axis=2))(gat)
    outputs = []
    for j in list(range(no_agents)):
        outputs.append(last_dense(dense(split[j])))

    V = tf.stack(outputs, axis=1)
    model = Model([I1, Adj], V)
    model._name = "final_network"

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=arglist.lr),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['acc']
                  )

    tf.keras.utils.plot_model(model, show_shapes=True)
    return model


def get_predictions(graph, adj, net):
    graph = tf.expand_dims(graph, axis=0)
    adj = tf.expand_dims(adj, axis=0)
    preds = net.predict([graph, adj])
    return preds


def get_actions(predictions, epsilon):
    prob = np.array(predictions)
    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    entropies = dist.entropy()
    # action = dist.sample()
    best_actions = tf.argmax(predictions, axis=-1)[0]
    actions = []
    for i in range(no_agents):
        if np.random.rand() < epsilon:
            actions.append(np.random.randint(0, no_actions))
        else:
            actions.append(best_actions.numpy()[i])
    return np.array(actions), entropies


def __build_conf():
    model = graph_net(arglist)
    model_t = graph_net(arglist)
    model_t.set_weights(model.get_weights())
    return model, model_t


def get_eval_reward(env, model, u):
    k_lst = list(range(arglist.no_neighbors + 2))[2:]  # [2,3]
    reward_total = []
    for _ in range(3):
        obs_n = env.reset()
        adj = u.get_adj(obs_n, k_lst)
        reward = 0
        for i in range(arglist.max_episode_len):
            predictions = get_predictions(u.to_tensor(np.array(obs_n)), adj, model)
            predictions = tf.squeeze(predictions, axis=0)
            actions = [tf.argmax(prediction, axis=-1).numpy() for prediction in predictions]

            # Observe next state, reward and done value
            new_obs_n, rew_n, done_n, _ = env.step(actions)
            adj = u.get_adj(new_obs_n, k_lst)
            obs_n = new_obs_n
            reward += rew_n[0]
        reward_total.append(reward)
    return reward_total


def main(arglist):
    global no_actions, no_features, no_agents
    env = make_env(arglist.scenario)
    env.discrete_action_input = True

    obs_shape_n = env.observation_space
    no_agents = env.n
    batch_size = arglist.batch_size
    no_neighbors = arglist.no_neighbors

    epsilon = arglist.epsilon
    epsilon_decay = arglist.epsilon_decay
    min_epsilon = arglist.min_epsilon
    max_epsilon = arglist.max_epsilon
    u = Utility(no_agents, is_gat=True)
    u.create_seed(arglist.seed)

    k_lst = list(range(no_neighbors + 2))[2:]  # [2,3]
    beta = 0.05  # Hyperparameter that controls the influence of entropy loss

    # Velocity.x Velocity.y Pos.x Pos.y {Land.Pos.x Land.Pos.y}*10 {Ent.Pos.x Ent.Pos.y}*9
    no_features = obs_shape_n[0].shape[0]
    no_actions = env.action_space[0].n
    model, model_t = __build_conf()

    # Results
    episode_rewards = [0.0]  # sum of rewards for all agents
    final_ep_rewards = []  # sum of rewards for training curve
    result_path = os.path.join("results", arglist.exp_name)
    res = os.path.join(result_path, " %s.csv" % arglist.exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    optimizer = tf.keras.optimizers.Adam(lr=arglist.lr)
    init_loss = np.inf

    replay_buffer = ReplayBuffer(arglist.max_buffer_size)  # Init Buffer
    episode_step = 0
    train_step = 0

    t_start = time.time()
    obs_n = env.reset()
    adj = u.get_adj(obs_n, k_lst)

    print('Starting iterations...')
    while True:
        episode_step += 1
        terminal = (episode_step >= arglist.max_episode_len)
        if episode_step % 3 == 0:
            adj = u.get_adj(obs_n, k_lst)

        predictions = get_predictions(u.to_tensor(np.array(obs_n)), adj, model)
        actions, entropies = get_actions(predictions, epsilon)
        # Observe next state, reward and done value
        new_obs_n, rew_n, done_n, _ = env.step(actions)
        done = all(done_n) or terminal
        cooperative_reward = rew_n[0]

        # Store the data in the replay memory
        replay_buffer.add(obs_n, adj, entropies, actions, cooperative_reward, new_obs_n, done)
        obs_n = new_obs_n

        episode_rewards[-1] += cooperative_reward

        if done or terminal:
            obs_n = env.reset()
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * train_step / 25)
            episode_step = 0
            episode_rewards.append(0)

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # Train the models
        if replay_buffer.can_provide_sample(batch_size) and train_step % 100 == 0:
            state, adj_n, entropies, actions, rewards, new_state, dones = replay_buffer.sample(batch_size)

            with tf.GradientTape() as tape:
                # Calculate TD-target. The Model.predict() method returns numpy() array without taping the forward pass.
                target_q_values = model_t([new_state, adj_n])
                # Apply max(Q) to obtain the TD-target
                target_q_tot = tf.reduce_max(target_q_values, axis=-1)
                # Apply VDN to reduce the agent-dimension
                max_q_tot = tf.reduce_sum(target_q_tot, axis=-1)
                y = rewards + (1. - dones) * arglist.gamma * (max_q_tot + beta * entropies)

                # Predictions
                action_one_hot = tf.one_hot(actions, no_actions, name='action_one_hot')
                q_values = model([state, adj_n])
                q_tot = tf.reduce_sum(q_values * action_one_hot, axis=-1, name='q_acted')
                pred = tf.reduce_sum(q_tot, axis=1)

                if "huber" in arglist.loss_type:
                    loss = tf.reduce_sum(u.huber_loss(pred, tf.stop_gradient(y)))
                elif "mse" in arglist.loss_type:
                    loss = tf.losses.mean_squared_error(pred, tf.stop_gradient(y))
                else:
                    raise RuntimeError(
                        "Loss function should be either Huber or MSE. %s found!" % arglist.loss_type)
                gradients = tape.gradient(loss, model.trainable_variables)
                local_clipped = u.clip_by_local_norm(gradients, 0.1)
            optimizer.apply_gradients(zip(local_clipped, model.trainable_variables))

            if loss.numpy() < init_loss:
                tf.saved_model.save(model, result_path)
                init_loss = loss.numpy()

        # train target model
        if arglist.soft_update:
            weights = model.get_weights()
            target_weights = model_t.get_weights()

            for w in range(len(weights)):
                target_weights[w] = arglist.tau * weights[w] + (1 - arglist.tau) * target_weights[w]
            model_t.set_weights(target_weights)
        elif train_step % 200 == 0:
            model_t.set_weights(model.get_weights())

        # display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            eval_reward = get_eval_reward(env, model, u)
            with open(res, "a+") as f:
                mes_dict = {"steps": train_step, "episodes": len(episode_rewards),
                            "train_episode_reward": np.round(np.mean(episode_rewards[-arglist.save_rate:]), 3),
                            "eval_episode_reward": np.round(np.mean(eval_reward), 3),
                            "loss": round(loss.numpy(), 3),
                            "time": round(time.time() - t_start, 3)}
                print(mes_dict)
                for item in list(mes_dict.values()):
                    f.write("%s\t" % item)
                f.write("\n")
                f.close()
        t_start = time.time()


if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)
