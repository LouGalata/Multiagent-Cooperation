import argparse
import os
import time

import numpy as np
import tensorflow as tf
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from spektral.layers import GATConv
from tensorflow.keras import Sequential

from buffers.replay_buffer import ReplayBuffer
from commons import util as u
from commons.OUNoise import OUNoise
from commons.weight_decay_optimizers import AdamW


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread_ivan2", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=4, help="number of agents")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--no-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--no-neighbors", type=int, default=2, help="number of neigbors to cooperate")
    parser.add_argument("--seed", type=int, default=3, help="seed")

    # Experience Replay
    parser.add_argument("--max-buffer-size", type=int, default=1e6, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--loss-type", type=str, default="huber", help="Loss function: huber or mse")

    # GNN training parameters
    parser.add_argument("--no-neurons", type=int, default=256, help="number of neurons on the first gnn")

    # Q-learning training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")
    parser.add_argument("--soft-update", type=bool, default=True, help="Mode of updating the target network")
    parser.add_argument("--use-ounoise", type=bool, default=True, help="Use Ornstein Uhlenbeck Process")
    # Evaluation
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--exp-name", type=str, default='igat4', help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--update-rate", type=int, default=30,
                        help="update policy after each x steps")
    parser.add_argument("--update-times", type=int, default=20,
                        help="Number of times we update the networks")
    return parser.parse_args()


def graph_net(arglist):
    I1 = Input(shape=(no_agents, no_features), name="graph_input")
    Adj = Input(shape=(no_agents, no_agents), name="adj")
    gat = GATConv(
        arglist.no_neurons,
        activation='relu',
        attn_heads=4,
        concat_heads=True,
    )([I1, Adj])
    concat = tf.keras.layers.Concatenate(axis=2)([I1, gat])

    split = Lambda(lambda x: tf.squeeze(tf.split(x, num_or_size_splits=no_agents, axis=1), axis=2))(concat)
    outputs = []
    for j in list(range(no_agents)):
        dense = Dense(arglist.no_neurons,
                      kernel_initializer=tf.keras.initializers.he_uniform(),
                      activation=tf.keras.layers.LeakyReLU(alpha=0.1))

        last_dense = Dense(no_actions, kernel_initializer=tf.keras.initializers.he_uniform())
        output = last_dense(dense(split[j]))
        output = tf.keras.activations.tanh(output)
        outputs.append(output)

    V = tf.stack(outputs, axis=1)
    model = Model([I1, Adj], V)
    model._name = "final_network"

    tf.keras.utils.plot_model(model, show_shapes=True)
    return model


def get_predictions(graph, adj, net):
    graph = tf.expand_dims(graph, axis=0)
    adj = tf.expand_dims(adj, axis=0)
    preds = net.predict([graph, adj])
    return preds


def get_actions(predictions, noise, noise_mode):
    outputs = predictions
    if arglist.use_ounoise:
        outputs += noise * noise_mode.noise()
        outputs = tf.clip_by_value(outputs, -1, 1)
    outputs = tf.squeeze(outputs, axis=0)
    return np.array(outputs)


def __build_conf():
    model = graph_net(arglist)
    model_t = graph_net(arglist)
    model_t.set_weights(model.get_weights())
    return model, model_t


def get_eval_reward(env, model):
    k_lst = list(range(arglist.no_neighbors + 2))[2:]  # [2,3]
    reward_total = []
    for _ in range(3):
        obs_n = env.reset()
        adj = u.get_adj(obs_n, k_lst, no_agents, is_gat=True)
        reward = 0
        for i in range(arglist.max_episode_len):
            predictions = get_predictions(u.to_tensor(np.array(obs_n)), adj, model)
            predictions = tf.squeeze(predictions, axis=0)
            # actions = [tf.argmax(prediction, axis=-1).numpy() for prediction in predictions]

            # Observe next state, reward and done value
            new_obs_n, rew_n, done_n, _ = env.step(predictions.numpy())
            adj = u.get_adj(new_obs_n, k_lst, no_agents, is_gat=True)
            obs_n = new_obs_n
            reward += rew_n[0]
        reward_total.append(reward)
    return reward_total


def main(arglist):
    global no_actions, no_features, no_agents
    env = u.make_env(arglist.scenario, arglist.no_agents)

    obs_shape_n = env.observation_space
    act_shape_n = env.action_space
    act_shape_n = u.space_n_to_shape_n(act_shape_n)
    no_agents = env.n
    batch_size = arglist.batch_size
    no_neighbors = arglist.no_neighbors

    k_lst = list(range(no_neighbors + 2))[2:]  # [2,3]
    u.create_seed(arglist.seed)

    noise_mode = OUNoise(act_shape_n[0], scale=1.0)
    noise = 0.1
    reduction_noise = 0.999
    # Velocity.x Velocity.y Pos.x Pos.y {Land.Pos.x Land.Pos.y}*10 {Ent.Pos.x Ent.Pos.y}*9
    no_features = obs_shape_n[0].shape[0]
    no_actions = act_shape_n[0][0]

    # model, model_t = __build_conf()
    model = tf.keras.models.load_model('results/mixing/igat4/')
    # optimizer = AdamW(learning_rate=arglist.lr, weight_decay=0.0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=arglist.lr)
    # Results
    episode_rewards = [0.0]  # sum of rewards for all agents
    result_path = os.path.join("asymptotic", arglist.exp_name)
    res = os.path.join(result_path, "%s.csv" % arglist.exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    replay_buffer = ReplayBuffer(arglist.max_buffer_size)  # Init Buffer
    episode_step = 0
    train_step = 0

    t_start = time.time()
    obs_n = env.reset()
    adj = u.get_adj(obs_n, k_lst, no_agents, is_gat=True)

    print('Starting iterations...')
    while True:
        episode_step += 1
        terminal = (episode_step >= arglist.max_episode_len)
        if episode_step % 3 == 0:
            adj = u.get_adj(obs_n, k_lst, no_agents, is_gat=True)

        predictions = get_predictions(u.to_tensor(np.array(obs_n)), adj, model)
        actions = get_actions(predictions, noise, noise_mode)

        # Observe next state, reward and done value
        new_obs_n, rew_n, done_n, _ = env.step(actions)
        done = all(done_n) or terminal
        cooperative_reward = rew_n[0]
        # Store the data in the replay memory
        replay_buffer.add(obs_n, adj, actions, cooperative_reward, new_obs_n, done)
        obs_n = new_obs_n

        episode_rewards[-1] += cooperative_reward

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        if terminal:
            with open(res, "a+") as f:
                mes_dict = {"episodes": len(episode_rewards),
                            "train_episode_reward": np.round(np.mean(episode_rewards[-2]), 3),
                            # "eval_episode_reward": np.round(np.mean(eval_reward), 3),
                            # "loss": round(loss.numpy(), 3),
                            "time": round(time.time() - t_start, 3)}
                print(mes_dict)
                for item in list(mes_dict.values()):
                    f.write("%s\t" % item)
                f.write("\n")
                f.close()
        t_start = time.time()

        # Train the models
        train_cond = not arglist.display and terminal
        if train_cond and len(replay_buffer) > arglist.batch_size:
            if terminal and len(episode_rewards) % arglist.update_rate == 0:  # only update every 30 episodes
                for _ in range(arglist.update_times):
                    state, adj_n, actions, rewards, new_state, dones = replay_buffer.sample(batch_size)
                    noise *= reduction_noise

                    with tf.GradientTape() as tape:
                        target_q_values = model_t([new_state, adj_n])
                        # Apply max(Q) to obtain the TD-target
                        target_q_tot = tf.reduce_sum(target_q_values, axis=1)
                        # Apply VDN to reduce the agent-dimension
                        max_q_tot = tf.reduce_max(target_q_tot, axis=-1)
                        y = rewards + (1. - dones) * arglist.gamma * max_q_tot

                        # Predictions
                        action_one_hot = tf.one_hot(tf.argmax(actions, axis=2, name='action_one_hot'), no_actions)
                        q_values = model([state, adj_n])
                        # VDN summation
                        q_tot = tf.reduce_sum(q_values * action_one_hot, axis=1, name='q_acted')
                        pred = tf.reduce_sum(q_tot, axis=1)

                        # loss = tf.reduce_mean(0.5 * tf.square(pred - tf.stop_gradient(y)), name="loss_mse")
                        if "huber" in arglist.loss_type:
                            # Computing the Huber Loss
                            loss = tf.reduce_sum(u.huber_loss(pred, tf.stop_gradient(y)))
                        elif "mse" in arglist.loss_type:
                            # Computing the MSE loss
                            loss = tf.losses.mean_squared_error(pred, tf.stop_gradient(y))
                        gradients = tape.gradient(loss, model.trainable_variables)
                        local_clipped = u.clip_by_local_norm(gradients, 0.1)
                    optimizer.apply_gradients(zip(local_clipped, model.trainable_variables))

                    tf.saved_model.save(model, result_path)

                # display training output
            if train_step % arglist.save_rate == 0:
                # eval_reward = get_eval_reward(env, model)
                with open(res, "a+") as f:
                    mes_dict = {"steps": train_step, "episodes": len(episode_rewards),
                                "train_episode_reward": np.round(np.mean(episode_rewards[-arglist.save_rate:]), 3),
                                # "eval_episode_reward": np.round(np.mean(eval_reward), 3),
                                # "loss": round(loss.numpy(), 3),
                                "time": round(time.time() - t_start, 3)}
                    print(mes_dict)
                    for item in list(mes_dict.values()):
                        f.write("%s\t" % item)
                    f.write("\n")
                    f.close()
            t_start = time.time()

        # if train_cond and arglist.soft_update:
        #     weights = model.get_weights()
        #     target_weights = model_t.get_weights()
        #
        #     for w in range(len(weights)):
        #         target_weights[w] = arglist.tau * weights[w] + (1 - arglist.tau) * target_weights[w]
        #     model_t.set_weights(target_weights)



if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)
