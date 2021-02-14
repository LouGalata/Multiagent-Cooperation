import argparse
import os
import random
import sys

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from spektral.layers import GCNConv
from tensorflow.keras import Sequential
import tensorflow_probability as tfp

from replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=6, help="number of agents")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=30000, help="number of episodes")
    parser.add_argument("--num-neighbors", type=int, default=2, help="number of neigbors to cooperate")

    # Experience Replay
    parser.add_argument("--max-buffer-size", type=int, default=20000, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")

    # GCN training parameters
    parser.add_argument("--num-neurons", type=int, default=32, help="number of neurons on the first gcn")

    # Q-learning training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

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
    world = scenario.make_world(no_agents=arglist.no_agents)
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
    # Try both with normal and sparce matrix
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
    adj = GCNConv.preprocess(adj).astype('f4')
    # Batch Mode needs dense inputs
    return adj


def GCN_net(n_neurons=None):
    I1 = Input(shape=(no_agents, feature_dim), name="gcn_input")
    Adj = Input(shape=(no_agents, no_agents), name="adj")

    gcn = GCNConv(n_neurons, kernel_initializer=tf.keras.initializers.he_uniform(),
                  activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                  use_bias=False,
                  name="Gcn")([I1, Adj])
    output = []
    dense = Dense(n_neurons,
                  kernel_initializer=tf.keras.initializers.he_uniform(),
                  activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                  name="dense_layer")
    last_dense = Dense(num_actions, kernel_initializer=tf.keras.initializers.he_uniform(),
                       activation=tf.keras.activations.softmax,
                       name="last_dense_layer")
    split = Lambda(lambda x: tf.squeeze(tf.split(x, num_or_size_splits=no_agents, axis=1), axis=2))(gcn)
    for j in list(range(no_agents)):
        V = dense(split[j])
        V2 = last_dense(V)
        output.append(V2)

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

#
# def get_actions_egreedy(predictions, epsilon=None):
#     """
#     Return a random action with probability epsilon and the max with probability 1 - epsilon for each agent
#     """
#     return [random.randrange(num_actions) if np.random.rand() < epsilon else np.argmax(prediction) for prediction in
#             predictions]
#
def sample_actions_from_distr(predictions, epsilon=None):
    """
    Return a random action with probability epsilon and the max with probability 1 - epsilon for each agent
    return [random.randrange(num_actions) if np.random.rand() < epsilon else np.argmax(prediction) for prediction in
            predictions]
    """
    prob = np.array(predictions)
    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    action = dist.sample()
    return [int(x[0]) for x in action.numpy()]

def __build_conf():
    # Configure logging
    hparams_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    logdir = os.path.join(hparams_log_dir, "hidden-untis=%d-batch-size=%d" %
                          (arglist.num_neurons, arglist.batch_size))

    model = GCN_net(n_neurons=arglist.num_neurons)
    model_t = GCN_net(n_neurons=arglist.num_neurons)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=arglist.lr, clipnorm=1.0, clipvalue=0.5),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['acc']
                  )
    # loss=tf.keras.losses.Huber()
    model_t.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=arglist.lr, clipnorm=1.0, clipvalue=0.5),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=['acc']
                    )

    callbacks = __get_callbacks(logdir)

    return model, model_t, callbacks


# DEBUGGING
def debugging_function(state, adj_n, model):
    # Debug intermediate outputs
    get_layer_output2 = K.function([model.layers[0].input, model.layers[1].input], [model.layers[2].output])
    layer_output2 = get_layer_output2([state, adj_n])

    get_layer_output3 = K.function([model.layers[0].input, model.layers[1].input], [model.layers[3].output])
    layer_output3 = get_layer_output3([state, adj_n])

    get_layer_output4 = K.function([model.layers[0].input, model.layers[1].input], [model.layers[4].output])
    layer_output4 = get_layer_output4([state, adj_n])

    get_layer_output5 = K.function([model.layers[0].input, model.layers[1].input], [model.layers[5].output])
    layer_output5 = get_layer_output5([state, adj_n])

    get_layer_output14 = K.function([model.layers[0].input, model.layers[1].input], [model.layers[14].output])
    layer_output14 = get_layer_output14([state, adj_n])

    get_layer_output15 = K.function([model.layers[0].input, model.layers[1].input], [model.layers[15].output])
    layer_output15 = get_layer_output15([state, adj_n])


def main(arglist):
    # Global variables
    global num_actions, feature_dim, no_agents
    # Create environment
    env = make_env(arglist.scenario)
    env.discrete_action_input = True

    obs_shape_n = env.observation_space
    no_agents = env.n
    batch_size = arglist.batch_size
    no_neighbors = arglist.num_neighbors
    k_lst = list(range(no_neighbors + 2))[2:]  # [2,3]


    # Velocity.x Velocity.y Pos.x Pos.y {Land.Pos.x Land.Pos.y}*10 {Ent.Pos.x Ent.Pos.y}*9
    num_features = obs_shape_n[0].shape[0]
    num_actions = env.action_space[0].n
    feature_dim = num_features  # the size of node features
    model, model_t, callback = __build_conf()
    reward_per_episode = pd.DataFrame(columns=['mean-reward'])
    agents_rewards = dict()
    for i in range(no_agents):
        reward_per_episode["agent_%d" % i] = ""

    #### Fill Buffer
    replay_buffer = ReplayBuffer(arglist.max_buffer_size)

    ###########playing#############
    i_episode = 0

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

    epsilon = 0.9
    decay = 0.9999
    min_epsilon = 0.1

    while i_episode < arglist.num_episodes:
        i_episode += 1
        # decayed-epsilon-greedy
        epsilon = max(min_epsilon, epsilon * decay)
        print("episode: " + str(i_episode))
        obs_n = env.reset()
        # obs_n = [(x - min(x)) / (max(x) - min(x)) for x in obs_n]

        # obs_n = scaler.transform(obs_n)
        for i in range(no_agents):
            agents_rewards["agent_%d" % i] = 0
        steps = 0
        sum_reward = 0
        while steps < arglist.max_episode_len:
            steps += 1
            adj = get_adj(obs_n, k_lst)
            predictions = get_actions(to_tensor(np.array(obs_n)), adj, model)
            actions = sample_actions_from_distr(predictions, epsilon=epsilon)
            # Observe next state, reward and done value
            new_obs_n, rew_n, done_n, _ = env.step(actions)
            # new_obs_n = [(x - min(x)) / (max(x) - min(x)) for x in new_obs_n]

            # new_obs_n = scaler.transform(new_obs_n)
            # Store the data in the replay memory
            replay_buffer.add(obs_n, adj, actions, rew_n, new_obs_n, done_n)
            obs_n = new_obs_n
            sum_reward += sum(rew_n)

            if steps == arglist.max_episode_len - 1:
                print("In episode no. %d reward is %.3f" % (i_episode, sum_reward))

            for i in range(no_agents):
                agents_rewards["agent_%d" % i] += rew_n[i]
            if not replay_buffer.can_provide_sample(batch_size):
                continue

            if steps*i_episode % 50 == 0:
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

                debugging_function(state, adj_n, model)

                for k in range(batch_size):
                    for j in range(no_agents):
                        if dones[k][j]:
                            q_values[j][k][actions[k][j]] = rewards[k][j]
                        else:
                            q_values[j][k][actions[k][j]] = rewards[k][j] + arglist.gamma * np.max(target_q_values[j][k])
                model.fit([state, adj_n], q_values, epochs=50, batch_size=batch_size, verbose=0, callbacks=callback)

                # train target model
                weights = model.get_weights()
                target_weights = model_t.get_weights()

                for w in range(len(weights)):
                    target_weights[w] = arglist.tau * weights[w] + (1 - arglist.tau) * target_weights[w]
                model_t.set_weights(target_weights)

        # Save metrics
        for key, val in agents_rewards.items():
            agents_rewards[key] = val / steps
        total_reward = sum_reward / steps
        agents_rewards['total-reward'] = total_reward
        reward_per_episode = reward_per_episode.append(agents_rewards, ignore_index=True)
    result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    result_path = result_path + '/{}.csv'.format('reward_per_episode')
    reward_per_episode.to_csv(result_path)


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    np.set_printoptions(threshold=sys.maxsize)
    arglist = parse_args()
    main(arglist)
