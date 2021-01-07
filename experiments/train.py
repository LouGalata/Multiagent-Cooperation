import argparse
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from spektral.layers import GCNConv

from experiments.replay_buffer import ReplayBuffer
from keras.models import Model
from keras.layers import Input, Dropout, Dense, Flatten

from keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow as tf
from tensorflow.keras.regularizers import l2

from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


np.set_printoptions(threshold=sys.maxsize)
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    # Experince Replay
    parser.add_argument("--max-buffer-size", type=int, default=200000, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--dropout-rate", type=float, default=0.05, help="dropout rate in the gcn")

    # GCN training parameters
    parser.add_argument("--num-neurons1", type=int, default=24, help="number of neurons on the first gcn")
    parser.add_argument("--num-neurons2", type=int, default=24, help="number of neurons on the second gcn")

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


def make_env(scenario_name, arglist, benchmark=False):
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
    return adj


def Q_Net(action_dim, input_dim):
    I1 = Input(shape=(1, input_dim))
    h1 = Flatten()(I1)
    V = Dense(input_dim, kernel_initializer='random_normal', activation='relu')(h1)
    V = Dense(action_dim, kernel_initializer='random_normal', activation='softmax')(V)
    model = Model(I1, V)
    model._name = "Qnet"
    model.summary()
    return model


def GCN_net(feature_dim=None, node_dim=None, n_neurons_1=None, n_neurons_2=None):
    I1 = Input(shape=(feature_dim, ))
    Adj = Input((node_dim, ), sparse=True)

    encoder = GCNConv(channels=n_neurons_1, activation='relu')([I1, Adj])
    decoder = GCNConv(channels=n_neurons_2, activation='relu')([encoder, Adj])

    model = Model([I1, Adj], decoder)
    model._name = "graph_network"
    # model.summary()
    return model




def get_embeddings(graph, adj, gcn_net):
    A = GCNConv.preprocess(adj).astype('f4')
    preds = gcn_net.predict([graph, A])
    return preds


def get_actions(embedding, q_net):
    tf_embeddings = tf.expand_dims(embedding, axis=0)
    tf_embeddings = tf.expand_dims(tf_embeddings, axis=0)
    action = q_net.predict([tf_embeddings])
    return action


def get_replay_buffer(agent_idx):
    replay_buffer = ReplayBuffer(arglist.max_buffer_size)
    max_replay_buffer_len = arglist.batch_size * arglist.max_episode_len
    replay_sample_index = None


def main(arglist):
    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    env.discrete_action_input = True

    # Create experience buffer


    obs_shape_n = env.observation_space

    # The last 2* (n-1) positions are the communication channel (not used)
    # The next 2* (n-1) positions are the relative positions of other agents. (Not used)

    N = len(obs_shape_n)  # the number of nodes
    num_features = obs_shape_n[0].shape[0]
    F = num_features - (env.n - 1) * 4  # the size of node features

    gcn_net = GCN_net(feature_dim=F,
                      node_dim=N,
                      n_neurons_1=arglist.num_neurons1,
                      n_neurons_2=arglist.num_neurons2)

    q_net = Q_Net(action_dim=env.action_space[0].n, input_dim=arglist.num_neurons2)

    # model._name = "final_architecture"
    # model.summary()
    # Parameters
    alpha = arglist.lr

    ###########playing#############
    i_episode = 0

    while i_episode < arglist.max_episode_len:
        alpha *= 0.996
        if alpha < 0.01:
            alpha = 0.01
        print(i_episode)
        i_episode = i_episode + 1
        obs_n = env.reset()
        obs_n = [x[:(num_features - (env.n - 1) * 4)] for x in obs_n]
        adj = get_adj(obs_n)
        # action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
        # Retrieve agents actions
        node_embeddings = get_embeddings(np.array(obs_n), adj, gcn_net)
        predictions = [get_actions(np.array(node_embedding), q_net) for node_embedding in node_embeddings.tolist()]
        actions = [np.argmax(prediction) for prediction in predictions]
        new_obs_n, rew_n, done_n, info_n = env.step(actions)

        i_episode += 1
        steps = 0
        while steps < arglist.num_episodes:
            steps += 1


if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)
