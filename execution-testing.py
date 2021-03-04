import argparse
import os
import pandas as pd
import sys
import time

import numpy as np
import tensorflow as tf
from keras.layers import Lambda
import keras
from scipy.spatial import cKDTree
from spektral.layers import GATConv



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=4, help="number of agents")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-neighbors", type=int, default=2, help="number of neigbors to cooperate")
    parser.add_argument("--use-gnn", type=bool, default=False, help="use of gnn netwrok or not")

    # Evaluation
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--exp-name", type=str, default='IQL', help="name of the experiment")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

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


def get_predictions(graph, adj, net):
    if arglist.use_gnn:
        graph = tf.expand_dims(graph, axis=0)
        adj = tf.expand_dims(adj, axis=0)
        preds = net.predict([graph, adj])
    else:
        state = Lambda(lambda x: tf.expand_dims(x, axis=0))(graph)
        # [Batch_size, 1, Features]
        splits = tf.split(state, num_or_size_splits=no_agents, axis=1)
        inputs = [tf.squeeze(x, axis=1) for x in splits]
        preds = net.predict(inputs)
    return preds



def main(arglist):
    # Global variables
    global num_actions, feature_dim, no_agents
    # Create environment
    env = make_env(arglist.scenario)
    env.discrete_action_input = True

    obs_shape_n = env.observation_space
    no_agents = env.n
    no_neighbors = arglist.num_neighbors
    k_lst = list(range(no_neighbors + 2))[2:]  # [2,3]

    # Velocity.x Velocity.y Pos.x Pos.y {Land.Pos.x Land.Pos.y}*10 {Ent.Pos.x Ent.Pos.y}*9
    num_features = obs_shape_n[0].shape[0]
    num_actions = env.action_space[0].n
    feature_dim = num_features  # the size of node features
    model = keras.models.load_model(arglist.exp_name)

    obs_n = env.reset()
    adj = get_adj(obs_n, k_lst)
    reward_total = []
    for i in range(arglist.max_episode_len):
        if i % 3 == 0:
            adj = get_adj(obs_n, k_lst)

        predictions = get_predictions(to_tensor(np.array(obs_n)), adj, model)
        predictions = tf.squeeze(predictions, axis=0)
        print("predictions: %s" % tf.shape(predictions))

        actions = [tf.argmax(prediction, axis=-1).numpy() for prediction in predictions]

        # Observe next state, reward and done value
        new_obs_n, rew_n, done_n, _ = env.step(actions)
        obs_n = new_obs_n
        reward_total.append(sum(rew_n))

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.5)
            print("Reward is %.3f" % sum(rew_n))
            env.render()
            continue
    pd.DataFrame(reward_total).to_csv(arglist.exp_name + "/rewards.csv")
    print("Final Reward is %.3f " % sum(reward_total))


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    np.set_printoptions(threshold=sys.maxsize)
    arglist = parse_args()
    main(arglist)
