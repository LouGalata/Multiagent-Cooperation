import argparse
import pandas as pd
import time

import numpy as np
import tensorflow as tf
from keras.layers import Lambda
import keras
from utils.util import Utility


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=2, help="number of agents")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-neighbors", type=int, default=2, help="number of neigbors to cooperate")
    parser.add_argument("--use-gat", type=bool, default=False, help="use of gat netwrok or not")
    parser.add_argument("--use-gcn", type=bool, default=False, help="use of gat netwrok or not")
    parser.add_argument("--use-rnn", type=bool, default=False, help="use of rnn netwrok or not")
    parser.add_argument("--history-size", type=int, default=6, help="timestep of Rnn memory")

    # Evaluation
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--exp-name", type=str, default='self-iql2-v4', help="name of the experiment")
    return parser.parse_args()


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(no_agents=arglist.no_agents)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def reformat_input(input):
    splits = tf.split(input, num_or_size_splits=no_agents, axis=1)
    return [tf.squeeze(x, axis=1) for x in splits]


def get_predictions(graph, adj, net):
    if arglist.use_gcn or arglist.use_gat:
        graph = tf.expand_dims(graph, axis=0)
        adj = tf.expand_dims(adj, axis=0)
        preds = net.predict([graph, adj])
    else:
        state = Lambda(lambda x: tf.expand_dims(x, axis=0))(graph)
        if arglist.use_rnn:
            inputs = reformat_input(state)
        else:
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
    u = Utility(no_agents, is_gat=arglist.use_gat, is_gcn=arglist.use_gcn, is_rnn=arglist.use_rnn)
    k_lst = list(range(no_neighbors + 2))[2:]  # [2,3]

    # Velocity.x Velocity.y Pos.x Pos.y {Land.Pos.x Land.Pos.y}*10 {Ent.Pos.x Ent.Pos.y}*9
    num_features = obs_shape_n[0].shape[0]
    num_actions = env.action_space[0].n
    feature_dim = num_features  # the size of node features
    model = keras.models.load_model("results/" + arglist.exp_name)

    obs_n = env.reset()
    if arglist.use_gat or arglist.use_gcn:
        adj = u.get_adj(obs_n, k_lst)
    else:
        adj = None
    if arglist.use_rnn:
        obs_n = u.reshape_state(obs_n, arglist.history_size)
    reward_total = []
    for i in range(arglist.max_episode_len):
        predictions = get_predictions(u.to_tensor(np.array(obs_n)), adj, model)
        predictions = tf.squeeze(predictions, axis=0)
        print("predictions: %s" % tf.shape(predictions))

        actions = [tf.argmax(prediction, axis=-1).numpy() for prediction in predictions]

        # Observe next state, reward and done value
        new_obs_n, rew_n, done_n, _ = env.step(actions)
        if arglist.use_gat or arglist.use_gcn:
            adj = u.get_adj(new_obs_n, k_lst)
        if arglist.use_rnn:
            new_obs_n = u.refresh_history(np.copy(obs_n), new_obs_n)
        obs_n = new_obs_n
        reward_total.append(rew_n[0])

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.5)
            print("Reward is %.3f" % sum(rew_n))
            env.render()
            continue
    pd.DataFrame(reward_total).to_csv("results/" + arglist.exp_name + "/testing_rewards.csv")
    print("Final Reward is %.3f " % sum(reward_total))


if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)
