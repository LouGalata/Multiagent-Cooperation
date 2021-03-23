"""
This file can be used to train an agent on a given environment, i.e. to replicate some
results from the papers.
"""

import argparse
import os
import random
import time
from typing import List

import numpy as np
import tensorflow as tf

from agents.AbstractAgent import AbstractAgent
from agents.maddpg import MADDPGAgent
from agents.magat import MAGATAgent
from commons.logger import RLLogger
from commons.util import softmax_to_argmax
from environments.multiagent.environment import MultiAgentEnv

if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],
                                             True)


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--exp-name", type=str, default='self-magat', help="name of the experiment")

    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--restore_fp", action="store_true", default=None,
                        help="path to restore models from: e.g. 'results/maddpg7/models/'")
    parser.add_argument("--save-rate", type=int, default=20,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--update-rate", type=int, default=200,
                        help="update policy after each x steps")
    parser.add_argument("--update-times", type=int, default=20,
                        help="Number of times we update the networks")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread_ivan", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=2, help="number of agents")
    parser.add_argument("--no-adversaries", type=int, default=0, help="number of adversaries")

    # Policies available agent: maddpg, matd3, magat
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy of good agents in env")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversary agents in env")

    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--no-episodes", type=int, default=30000, help="number of episodes")
    parser.add_argument("--no-neighbors", type=int, default=2, help="number of neigbors to cooperate")
    parser.add_argument("--seed", type=int, default=3, help="seed")


    # Experience Replay
    parser.add_argument("--buffer-size", type=int, default=1e6, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--no-neurons", type=int, default=128, help="number of neurons on the first gnn")
    parser.add_argument("--no-layers", type=int, default=2, help="number of hidden layers in critics and actors")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

    parser.add_argument("--priori-replay", type=bool, default=False, help="Use prioritized experience replay")
    parser.add_argument("--alpha", type=float, default=0.6, help="alpha value (weights prioritization vs random)")
    parser.add_argument("--beta", type=float, default=0.5, help="beta value  (controls importance sampling)")

    parser.add_argument("--use-gumbel", type=bool, default=True, help="Use Gumbel softmax")
    parser.add_argument("--noise", type=float, default=0.1, help="Add noise on actions")
    parser.add_argument("--noise-reduction", type=float, default=0.999, help="Noise decay on actions")

    parser.add_argument("--use-target-action", type=bool, default=True, help="use action from target network")
    parser.add_argument("--hard-max", type=bool, default=False, help="Only output one action")

    return parser.parse_args()


def make_env(scenario_name) -> MultiAgentEnv:
    """
    Create an environment
    :param scenario_name:
    :return:
    """
    import environments.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    # create world
    world = scenario.make_world(no_agents=arglist.no_agents)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def train(exp_name, save_rate, display, restore_fp):
    # Create environment
    env = make_env(arglist.scenario)

    logger = RLLogger(exp_name, env.n, env.n_adversaries, save_rate, arglist)
    # Create agents
    agents = get_agents(env, env.n_adversaries, arglist.good_policy, arglist.adv_policy,
                        arglist.lr, arglist.batch_size, arglist.buffer_size, arglist.no_neurons,
                        arglist.no_layers, arglist.gamma, arglist.tau, arglist.priori_replay,
                        arglist.alpha, arglist.no_episodes, arglist.max_episode_len, arglist.beta,
                        arglist.no_neighbors, logger, arglist.noise)

    # Load previous results, if necessary
    if restore_fp is not None:
        print('Loading previous state...')
        for ag_idx, agent in enumerate(agents):
            loading_episode = 0
            # fp = os.path.join(restore_fp, 'agent_{}'.format(ag_idx))
            fp = os.path.join(restore_fp,  'ep{}'.format(loading_episode), 'agent_{}'.format(ag_idx))
            agent.load(fp)

    obs_n = env.reset()

    print('Starting iterations...')
    while True:
        # get action
        if arglist.use_target_action:
            action_n = [agent.target_action(obs.astype(np.float32)[None])[0] for agent, obs in
                        zip(agents, obs_n)]
        else:
            action_n = [agent.action(obs.astype(np.float32)) for agent, obs in zip(agents, obs_n)]
        # environment step
        if arglist.hard_max:
            hard_action_n = softmax_to_argmax(action_n, agents)
            new_obs_n, rew_n, done_n, info_n = env.step(hard_action_n)
        else:
            action_n = [action.numpy() for action in action_n]
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        logger.episode_step += 1

        done = all(done_n)
        terminal = (logger.episode_step >= arglist.max_episode_len)
        done = done or terminal

        # collect experience
        for i, agent in enumerate(agents):
            agent.add_transition(obs_n, action_n, rew_n[i], new_obs_n, done)
            agent.update_noise(arglist.noise_reduction)
        obs_n = new_obs_n

        for ag_idx, rew in enumerate(rew_n):
            logger.cur_episode_reward += rew
            logger.agent_rewards[ag_idx][-1] += rew

        if done:
            obs_n = env.reset()
            episode_step = 0
            logger.record_episode_end(agents, arglist.display)

        logger.train_step += 1

        # policy updates
        train_cond = not display
        for agent in agents:
            # if train_cond and len(agent.replay_buffer) > arglist.batch_size * arglist.max_episode_len:
            if train_cond and len(agent.replay_buffer) > arglist.batch_size + 1:
                if logger.train_step % arglist.update_rate == 0:  # only update every 100 steps
                    for _ in range(arglist.update_times):
                        q_loss, pol_loss = agent.update(agents, logger.train_step)

        # for displaying learned policies
        if display:
            time.sleep(0.1)
            env.render()

        # saves logger outputs to a file similar to the way in the original MADDPG implementation
        if len(logger.episode_rewards) > arglist.no_episodes:
            logger.experiment_end()
            return logger.get_sacred_results()


def get_agents(env, num_adversaries, good_policy, adv_policy, lr, batch_size,
               buff_size, num_units, num_layers, gamma, tau, priori_replay, alpha, num_episodes,
               max_episode_len, beta, no_neighbors, logger, noise
               ) -> List[AbstractAgent]:
    """
    This function generates the agents for the environment. The parameters are meant to be filled
    by sacred, and are therefore documented in the configuration function train_config.

    :returns List[AbstractAgent] returns a list of instantiated agents
    """
    agents = []
    for agent_idx in range(num_adversaries, env.n):
        if good_policy == 'maddpg':
            agent = MADDPGAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr, num_layers,
                                num_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta, logger=logger, noise=noise)
        elif good_policy == "magat":
            agent = MAGATAgent(no_neighbors, env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers,
                               num_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta, logger=logger, noise=noise)
        else:
            raise RuntimeError('Invalid Class')
        agents.append(agent)
    print('Using good policy {} and adv policy {}'.format(good_policy, adv_policy))
    return agents


def create_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    file_observer = os.path.join("results", arglist.exp_name)
    if not os.path.exists(file_observer):
        os.makedirs(file_observer)
    train(arglist.exp_name, arglist.save_rate, arglist.display, arglist.restore_fp)


if __name__ == '__main__':
    arglist = parse_args()
    create_seed(arglist.seed)
    main()
