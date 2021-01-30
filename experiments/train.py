import argparse
import os
import random
import sys
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Concatenate
from keras.models import Model
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from spektral.layers import GCNConv
from tensorflow.keras import Sequential
# import keras.backend as K

from replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=300, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")

    # Experince Replay
    parser.add_argument("--max-buffer-size", type=int, default=20000, help="maximum buffer capacity")

    # Core training parameters  -- Passed through hparams
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=24, help="number of episodes to optimize at the same time")

    # GCN training parameters -- Passed through hparams
    parser.add_argument("--num-neurons", type=int, default=24, help="number of neurons on the first gcn")

    # Q-learning training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)

    # benchmark: provides diagnostic data for policies trained on the environment
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def to_tensor(arg):
    arg = tf.convert_to_tensor(arg)
    return arg


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    # Here is defined the num_agents
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def all_equal2(iterator):
    return len(set(iterator)) <= 1


def __get_callbacks(logdir):
    callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                 tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=5),
                 tf.keras.callbacks.TensorBoard(logdir,
                                                update_freq='epoch',
                                                write_graph=False,
                                                histogram_freq=5),

                 tf.keras.callbacks.ModelCheckpoint(
                     filepath=os.path.join(logdir, "cp.ckpt"),
                     save_best_only=True,
                     save_weights_only=False,
                     monitor='loss',
                     verbose=1)
                 ]
    return callbacks


def get_adj(arr, k=3):
    """
    Take as input the new obs. In position 2 to k, there are the x and y coordinates of each agent
    Make an adjacency matrix, where each agent communicates with the k closest ones
    """
    length = len(arr)
    # Try both with normal and sparce matrix
    # adjacency = np.zeros((length, length))
    x_idx = np.repeat(list(range(length)), k)

    k_lst = list(range(k + 2))[2:]  # [2,3]
    neighbors = []
    # construct a kd-tree
    tree = cKDTree(arr)
    for cnt, row in enumerate(arr):
        # find k nearest neighbors for each element of data, squeezing out the zero result (the first nearest
        # neighbor is always itself)
        dd, ii = tree.query(row, k=k_lst)
        # apply an index filter on data to get the nearest neighbor elements
        neighbors.append(ii)
        # adjacency[cnt, ii] = 1.0

    y_idx = np.concatenate(neighbors).ravel()
    data = np.ones((length * k))
    # Use a sparce matrix
    adj = csr_matrix((data, (x_idx, y_idx)), shape=(length, length))
    dense = np.array(adj.todense())
    adj = GCNConv.preprocess(dense).astype('f4')
    # Batch Mode needs dense inputs
    return adj


def GCN_net(n_neurons=None, batch_size=None):
    I1 = Input(shape=(no_agents, feature_dim), name="gcn_input")
    # Adj = Input((no_agents,), sparse=True, batch_size=batch_size, name="adj")
    Adj = Input(shape=(no_agents, no_agents), name="adj")

    encoder = GCNConv(channels=n_neurons, activation='relu', name="Encoder")([I1, Adj])
    decoder = GCNConv(channels=n_neurons, activation='relu', name="Decoder")([encoder, Adj])
    # q_net_input = tf.expand_dims(decoder, axis=0)
    output = []
    dense = Dense(n_neurons, kernel_initializer='random_normal', activation='softmax', name="dense_layer")
    for j in list(range(no_agents)):
        T = Lambda(lambda x: x[:, j], output_shape=(n_neurons,), name="lambda_layer_agent_%d" % j)(
            decoder)
        V = dense(T)
        output.append(V)

    model = Model([I1, Adj], output)
    model._name = "final_network"
    # output = Concatenate()(output)
    # vdn_model =
    # model.summary()
    # tf.keras.utils.plot_model(model, show_shapes=True)
    return model


def get_actions(graph, adj, gcn_net):
    graph = tf.expand_dims(graph, axis=0)
    adj = tf.expand_dims(adj, axis=0)
    preds = gcn_net.predict([graph, adj])
    return preds


def get_actions_egreedy(predictions, epsilon=None):
    """
    Return a random action with probability epsilon and the max with probability 1 - epsilon for each agent
    """
    return [random.randrange(num_actions) if np.random.rand() < epsilon else np.argmax(prediction) for prediction in
            predictions]


def __build_conf():
    # Configure logging
    hparams_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    logdir = os.path.join(hparams_log_dir, "hidden-untis=%d-batch-size=%d" %
                          (arglist.num_neurons, arglist.batch_size))

    model = GCN_net(n_neurons=arglist.num_neurons, batch_size=arglist.batch_size)
    model_t = GCN_net(n_neurons=arglist.num_neurons, batch_size=arglist.batch_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=arglist.lr, clipnorm=1.0, clipvalue=0.5),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['acc']
                  )
    model_t.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=arglist.lr, clipnorm=1.0, clipvalue=0.5),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=['acc']
                    )

    callbacks = __get_callbacks(logdir)

    return model, model_t, callbacks


def main(arglist):
    # Global variables
    global num_actions, feature_dim, no_agents
    # Create environment
    env = make_env(arglist.scenario, arglist.benchmark)
    env.discrete_action_input = True

    obs_shape_n = env.observation_space
    no_agents = env.n
    batch_size = arglist.batch_size
    # The last 2* (n-1) positions are the communication channel (not used)
    # The next 2* (n-1) positions are the relative positions of other agents. (Not used)
    num_features = obs_shape_n[0].shape[0]
    num_actions = env.action_space[0].n
    feature_dim = num_features - (env.n - 1) * 4  # the size of node features
    model, model_t, callback = __build_conf()
    reward_per_episode = pd.DataFrame(columns=['mean-reward'])
    agents_rewards = dict()
    for i in range(no_agents):
        reward_per_episode["agent_%d" % i] = ""

    #### Fill Buffer
    replay_buffer = ReplayBuffer(arglist.max_buffer_size)

    ###########playing#############
    i_episode = 0
    epsilon = 1.0

    # Normalize input
    # obs_n = env.reset()
    # obs_n = [x[:(num_features - (env.n - 1) * 4)] for x in obs_n]
    # observations = []
    # for i in range(100):
    #     observations.extend([x for x in obs_n])
    #     adj = get_adj(obs_n)
    #     predictions = get_actions(to_tensor(np.array(obs_n)), adj, model)
    #     actions = get_actions_egreedy(predictions, epsilon=epsilon)
    #     # Observe next state, reward and done value
    #     new_obs_n, rew_n, done_n, _ = env.step(actions)
    #     new_obs_n = [x[:(num_features - (env.n - 1) * 4)] for x in new_obs_n]
    #     obs_n = new_obs_n
    #
    # scaler = StandardScaler().fit(observations)

    while i_episode < arglist.num_episodes:
        i_episode += 1
        epsilon *= 0.996
        if epsilon < 0.01: epsilon = 0.01
        print("episode: " + str(i_episode))
        obs_n = env.reset()
        obs_n = [x[:(num_features - (env.n - 1) * 4)] for x in obs_n]

        # obs_n = scaler.transform(obs_n)
        for i in range(no_agents):
            agents_rewards["agent_%d" % i] = 0
        steps = 0
        sum_reward = 0
        while steps < arglist.max_episode_len:
            steps += 1
            adj = get_adj(obs_n)
            predictions = get_actions(to_tensor(np.array(obs_n)), adj, model)
            actions = get_actions_egreedy(predictions, epsilon=epsilon)
            # Observe next state, reward and done value
            new_obs_n, rew_n, done_n, _ = env.step(actions)
            new_obs_n = [x[:(num_features - (env.n - 1) * 4)] for x in new_obs_n]
            # new_obs_n = scaler.transform(new_obs_n)
            # Store the data in the replay memory
            replay_buffer.add(obs_n, adj, actions, rew_n, new_obs_n, done_n)
            obs_n = new_obs_n
            sum_reward += sum(rew_n)
            for i in range(no_agents):
                agents_rewards["agent_%d" % i] += rew_n[i]
            if not replay_buffer.can_provide_sample(batch_size):
                continue

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

            # Calculate TD-target
            q_values = model.predict([state, adj_n])
            target_q_values = model_t.predict([new_state, adj_n])

            for k in range(len(batch)):
                for j in range(no_agents):
                    if dones[k][j]:
                        q_values[j][k][actions[k][j]] = rewards[k][j]
                    else:
                        q_values[k][j][actions[k][j]] = rewards[k][j] + arglist.gamma * np.max(target_q_values[j][k])
            model.fit([state, adj_n], q_values, epochs=50, batch_size=batch_size, verbose=1, callbacks=callback)

            if steps % 100 == 0:
                # train target model
                weights = model.get_weights()
                target_weights = model_t.get_weights()

                for w in range(len(weights)):
                    target_weights[w] = arglist.tau * weights[w] + (1 - arglist.tau) * target_weights[w]
                model_t.set_weights(target_weights)

        # Save metrics
        mean_reward = sum_reward / steps
        agents_rewards /= steps
        data = {'mean-reward': mean_reward}
        for i in range(no_agents):
            data.update({'agent_%d' % i: agents_rewards[i]})
        reward_per_episode = reward_per_episode.append(data,  ignore_index=True)
    result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    result_path = result_path + '{}.csv'.format('reward_per_episode')
    reward_per_episode.to_csv(result_path)


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    arglist = parse_args()
    main(arglist)
