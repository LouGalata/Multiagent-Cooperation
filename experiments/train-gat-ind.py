import argparse
import os
import pickle
import random
import sys
import time

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda, Dense, concatenate, merge
from keras.models import Model
from scipy.spatial import cKDTree
from spektral.layers import GATConv
from tensorflow.keras import Sequential

from replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=6, help="number of agents")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=30000, help="number of episodes")
    parser.add_argument("--num-neighbors", type=int, default=2, help="number of neigbors to cooperate")
    parser.add_argument("--seed", type=int, default=1, help="seed")

    # Experience Replay
    parser.add_argument("--max-buffer-size", type=int, default=20000, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon exploration")
    parser.add_argument("--epsilon-decay", type=float, default=0.0003, help="epsilon decay")
    parser.add_argument("--min-epsilon", type=float, default=0.01, help="min epsilon")
    parser.add_argument("--max-epsilon", type=float, default=1.0, help="max epsilon")

    # GNN training parameters
    parser.add_argument("--num-neurons", type=int, default=32, help="number of neurons on the first gnn")
    parser.add_argument("--l2-reg", type=float, default=2.5e-4, help="kernel regularizer")

    # Q-learning training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

    # Evaluation
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--exp-name", type=str, default='VDN-GAT-ELU_v2', help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=50,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")

    return parser.parse_args()


def to_tensor(arg):
    arg = tf.convert_to_tensor(arg)
    return arg


def create_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    # Here is defined the num_agents
    world = scenario.make_world(no_agents=arglist.no_agents, seed=arglist.seed)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def __get_callbacks(logdir):
    callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                 tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=5),
                 tf.keras.callbacks.TensorBoard(logdir, update_freq="epoch", profile_batch=0),

                 tf.keras.callbacks.ModelCheckpoint(
                     filepath=os.path.join(logdir, "cp.ckpt"),
                     save_best_only=True,
                     save_freq=25,
                     save_weights_only=False,
                     monitor='loss',
                     verbose=0)
                 ]
    return callbacks


def get_adj(arr, k_lst):
    """
    Take as input the new obs. In position 4 to k, there are the x and y coordinates of each agent
    Make an adjacency matrix, where each agent communicates with the k closest ones
    """
    points = [i[2:4] for i in arr]
    adj = np.zeros((no_agents, no_agents), dtype=float)
    # construct a kd-tree
    tree = cKDTree(points)
    for cnt, row in enumerate(points):
        # find k nearest neighbors for each element of data, squeezing out the zero result (the first nearest
        # neighbor is always itself)
        dd, ii = tree.query(row, k=k_lst)
        # apply an index filter on data to get the nearest neighbor elements
        adj[cnt][ii] = 1
        # adjacency[cnt, ii] = 1.0

    # add self-loops and symmetric normalization
    adj = GATConv.preprocess(adj).astype('f4')
    # Batch Mode needs dense inputs
    return adj


def graph_net(arglist):
    I1 = Input(shape=(no_agents, feature_dim), name="graph_input")
    Adj = Input(shape=(no_agents, no_agents), name="adj")
    gat = GATConv(
        arglist.num_neurons,
        activation='elu',
        attn_heads=4,
        concat_heads=True,
    )([I1, Adj])


    split = Lambda(lambda x: tf.squeeze(tf.split(x, num_or_size_splits=no_agents, axis=1), axis=2))(gat)
    outputs = []
    for j in list(range(no_agents)):
        dense = Dense(arglist.num_neurons,
                      kernel_initializer=tf.keras.initializers.he_uniform(),
                      activation=tf.keras.layers.LeakyReLU(alpha=0.1))

        last_dense = Dense(num_actions, kernel_initializer=tf.keras.initializers.he_uniform())
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


def clip_by_local_norm(gradients, norm):
    """
    Clips gradients by their own norm, NOT by the global norm
    as it should be done (according to TF documentation).
    This here is the way MADDPG does it.
    """
    for idx, grad in enumerate(gradients):
        gradients[idx] = tf.clip_by_norm(grad, norm)
    return gradients


def get_predictions(graph, adj, net):
    graph = tf.expand_dims(graph, axis=0)
    adj = tf.expand_dims(adj, axis=0)
    preds = net.predict([graph, adj])
    return preds


def get_actions(predictions, epsilon):
    best_actions = tf.argmax(predictions, axis=-1)[0]
    actions = []
    for i in range(no_agents):
        if np.random.rand() < epsilon:
            actions.append(np.random.randint(0, num_actions))
        else:
            actions.append(best_actions.numpy()[i])
    return np.array(actions)


def __build_conf():
    hparams_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', arglist.exp_name))
    logdir = os.path.join(hparams_log_dir, "hidden-units=%d-batch-size=%d" %
                          (arglist.num_neurons, arglist.batch_size))

    model = graph_net(arglist)
    model_t = graph_net(arglist)
    model_t.set_weights(model.get_weights())

    callbacks = __get_callbacks(logdir)
    return model, model_t, callbacks


def main(arglist):
    global num_actions, feature_dim, no_agents
    env = make_env(arglist.scenario)
    env.discrete_action_input = True

    obs_shape_n = env.observation_space
    no_agents = env.n
    batch_size = arglist.batch_size
    no_neighbors = arglist.num_neighbors
    epsilon = arglist.epsilon
    epsilon_decay = arglist.epsilon_decay
    min_epsilon = arglist.min_epsilon
    max_epsilon = arglist.max_epsilon
    k_lst = list(range(no_neighbors + 2))[2:]  # [2,3]

    # Velocity.x Velocity.y Pos.x Pos.y {Land.Pos.x Land.Pos.y}*10 {Ent.Pos.x Ent.Pos.y}*9
    num_features = obs_shape_n[0].shape[0]
    num_actions = env.action_space[0].n
    feature_dim = num_features  # the size of node features
    model, model_t, callback = __build_conf()
    optimizer = tf.keras.optimizers.Adam(lr=arglist.lr)
    init_loss = np.inf
    # Results
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    result_path = os.path.join(os.path.dirname(__file__), '..', "/rewards-per-episode.csv")
    if not os.path.exists(result_path):
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
    replay_buffer = ReplayBuffer(arglist.max_buffer_size)  # Init Buffer
    episode_step = 0
    train_step = 0

    t_start = time.time()
    obs_n = env.reset()
    adj = get_adj(obs_n, k_lst)

    print('Starting iterations...')
    while True:
        episode_step += 1
        terminal = (episode_step >= arglist.max_episode_len)
        if episode_step % 3 == 0:
            adj = get_adj(obs_n, k_lst)

        predictions = get_predictions(to_tensor(np.array(obs_n)), adj, model)
        actions = get_actions(predictions, epsilon)

        # Observe next state, reward and done value
        new_obs_n, rew_n, done_n, _ = env.step(actions)
        done = all(done_n)
        cooperative_reward = rew_n[0]
        # Store the data in the replay memory
        replay_buffer.add(obs_n, adj, actions, cooperative_reward, new_obs_n, done)
        obs_n = new_obs_n


        episode_rewards[-1] += cooperative_reward
        for i, rew in enumerate(rew_n):
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * train_step/25)
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # Train the models
        if replay_buffer.can_provide_sample(batch_size) and train_step % 100 == 0:
            # Pass a batch of states through the policy network to calculate the Q(s, a)
            # Pass a batch of states through the target network to calculate the Q'(s', a)
            batch = replay_buffer.sample(batch_size)
            state, adj_n, actions, rew_n, new_state, done_n = [], [], [], [], [], []
            # for e in batch:
            for e in range(batch_size):
                state.append(batch[0][e])
                new_state.append(batch[4][e])
                adj_n.append(batch[1][e])
                actions.append(batch[2][e])
                rew_n.append(batch[3][e])
                done_n.append(batch[5][e])
            actions = np.asarray(actions)
            rewards = np.asarray(rew_n)
            dones = np.asarray(done_n)
            adj_n = np.asarray(adj_n)
            state = np.asarray(state)
            new_state = np.asarray(new_state)

            with tf.GradientTape() as tape:
                # Calculate TD-target. The Model.predict() method returns numpy() array without taping the forward pass.
                target_q_values = model_t([new_state, adj_n])
                # Apply max(Q) to obtain the TD-target
                target_q_tot = tf.reduce_max(target_q_values, axis=-1)
                # Apply VDN to reduce the agent-dimension
                max_q_tot = tf.reduce_sum(target_q_tot, axis=-1)
                y = rewards + (1. - dones) * arglist.gamma * max_q_tot

                # Predictions
                action_one_hot = tf.one_hot(actions, num_actions, name='action_one_hot')
                q_values = model([state, adj_n])
                q_tot = tf.reduce_sum(q_values * action_one_hot, axis=-1, name='q_acted')
                pred = tf.reduce_sum(q_tot, axis=1)
                loss = tf.reduce_mean(0.5 * tf.square(pred - tf.stop_gradient(y)), name="loss_mse")
                gradients = tape.gradient(loss, model.trainable_variables)
                local_clipped = clip_by_local_norm(gradients, 0.1)
            optimizer.apply_gradients(zip(local_clipped, model.trainable_variables))

            if loss.numpy() < init_loss:
                tf.saved_model.save(model, os.path.abspath(os.path.join(
                    os.path.dirname(__file__), '..', arglist.exp_name)))
                init_loss = loss.numpy()

            # train target model
            weights = model.get_weights()
            target_weights = model_t.get_weights()

            for w in range(len(weights)):
                target_weights[w] = arglist.tau * weights[w] + (1 - arglist.tau) * target_weights[w]
            model_t.set_weights(target_weights)

        # display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            with open(result_path, "a+") as f:
                mes_dict = {"steps": train_step, "episodes": len(episode_rewards),
                            "mean_episode_reward": round(np.mean(episode_rewards[-arglist.save_rate:]), 3),
                            "time": round(time.time() - t_start, 3)}

                for item in list(mes_dict.values()):
                    f.write("%s\t" % item)
                f.write("\n")
                f.close()
            print(mes_dict)
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                if not os.path.exists(arglist.plots_dir):
                    os.makedirs(arglist.plots_dir)
                rew_file_name = arglist.plots_dir + '/' + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                break


if __name__ == '__main__':
    arglist = parse_args()
    create_seed(arglist.seed)
    main(arglist)
