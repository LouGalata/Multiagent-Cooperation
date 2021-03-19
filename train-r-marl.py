"""
This file can be used to train an agent on a given environment, i.e. to replicate some
results from the papers.
"""

import argparse
import os
import time
from typing import List

import numpy as np
import tensorflow as tf

from agents import MATD3Agent
from agents.rmagat import MAGATAgent
from agents.rmaddpg import MADDPGAgent
from agents.AbstractAgent import AbstractAgent
from commons.loggerserver import RLLogger
from environments.multiagent.environment import MultiAgentEnv

if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],
                                             True)


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--exp-name", type=str, default='self-rmagat-4', help="name of the experiment")

    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--restore_fp", action="store_true", default=None,
                        help="path to restore models from: e.g. 'results/maddpg/models/' ")
    parser.add_argument("--save-rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--update-rate", type=int, default=100,
                        help="update policy after each x steps")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread_ivan", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=4, help="number of agents")
    parser.add_argument("--no-adversaries", type=int, default=0, help="number of adversaries")

    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--no-episodes", type=int, default=30000, help="number of episodes")
    parser.add_argument("--no-neighbors", type=int, default=2, help="number of neigbors to cooperate")
    parser.add_argument("--seed", type=int, default=1, help="seed")

    # Policies available agent: maddpg, matd3, magat
    parser.add_argument("--good-policy", type=str, default="magat", help="policy of good agents in env")
    parser.add_argument("--adv-policy", type=str, default="magat", help="policy of adversary agents in env")

    # Experience Replay
    parser.add_argument("--buffer-size", type=int, default=1e6, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--no-neurons", type=int, default=64, help="number of neurons on the first gnn")
    parser.add_argument("--l2-reg", type=float, default=2.5e-4, help="kernel regularizer")
    parser.add_argument("--no-layers", type=int, default=2, help="number of hidden layers in critics and actors")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

    parser.add_argument("--priori-replay", type=bool, default=False, help="Use prioritized experience replay")
    parser.add_argument("--alpha", type=float, default=0.6, help="alpha value (weights prioritization vs random)")
    parser.add_argument("--beta", type=float, default=0.5, help="beta value  (controls importance sampling)")

    parser.add_argument("--loss-type", type=str, default="huber", help="Loss function: huber or mse")
    parser.add_argument("--soft-update", type=bool, default=True, help="Mode of updating the target network")
    parser.add_argument("--clip-gradients", type=float, default=0.5, help="Norm of clipping gradients")
    parser.add_argument("--use-gumbel", type=bool, default=True, help="Use Gumbel softmax")

    parser.add_argument("--decay-mode", type=str, default="exp2", help="linear or exp")
    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon exploration")
    parser.add_argument("--epsilon-decay", type=float, default=0.0003, help="epsilon decay")
    parser.add_argument("--min-epsilon", type=float, default=0.01, help="min epsilon")
    parser.add_argument("--max-epsilon", type=float, default=1.0, help="max epsilon")

    parser.add_argument("--policy-update-rate", type=int, default=1, help="MATD3")

    parser.add_argument("--critic-action-noise-stddev", type=float, default=0.0, help="Added noise in critic updates")
    parser.add_argument("--use-target-action", type=bool, default=True, help="use action from target network")
    parser.add_argument("--hard-max", type=bool, default=False, help="Only output one action")

    parser.add_argument("--temporal-mode", type=str, default="rnn", help="Attention or rnn")
    parser.add_argument("--history-size", type=int, default=3,
                        help="number of timesteps/ history that will be used in the recurrent model")
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


def reshape_state(state, history_size):
    # Clone and concatenate the state, history_size times
    return np.tile(np.expand_dims(state, axis=1), (1, history_size, 1))


def refresh_history(history, state_next):
    history[:, :-1] = history[:, 1:]
    history[:, -1] = state_next
    return history


def train(exp_name, save_rate, display, restore_fp):
    # Create environment
    env = make_env(arglist.scenario)

    logger = RLLogger(exp_name, env.n, env.n_adversaries, save_rate, arglist)
    # Create agents
    agents = get_agents(env, env.n_adversaries, arglist.good_policy, arglist.adv_policy, arglist.history_size,
                        arglist.lr, arglist.batch_size, arglist.buffer_size, arglist.no_neurons,
                        arglist.no_layers, arglist.gamma, arglist.tau, arglist.priori_replay,
                        arglist.alpha, arglist.no_episodes, arglist.max_episode_len, arglist.beta,
                        arglist.policy_update_rate, arglist.critic_action_noise_stddev,
                        arglist.no_neighbors, logger)


    # Load previous results, if necessary
    if restore_fp is not None:
        print('Loading previous state...')
        for ag_idx, agent in enumerate(agents):
            loading_episode = 0
            fp = os.path.join(restore_fp,  'ep{}'.format(loading_episode), 'agent_{}'.format(ag_idx))
            agent.load(fp)

    obs_n = env.reset()
    obs_n = reshape_state(obs_n, arglist.history_size)
    print('Starting iterations...')
    while True:
        # get action

        if arglist.use_target_action:
            action_n = [agent.target_action(obs.astype(np.float32)[None])[0] for agent, obs in
                        zip(agents, obs_n)]
        else:
            action_n = [agent.action(obs.astype(np.float32)) for agent, obs in zip(agents, obs_n)]

        # environment step
        action_n = [action.numpy() for action in action_n]
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        new_obs_n = refresh_history(np.copy(obs_n), new_obs_n)

        logger.episode_step += 1

        done = all(done_n)
        terminal = (logger.episode_step >= arglist.max_episode_len)
        done = done or terminal

        # collect experience
        for i, agent in enumerate(agents):
            agent.add_transition(obs_n, action_n, rew_n[i], new_obs_n, done)
        obs_n = new_obs_n

        for ag_idx, rew in enumerate(rew_n):
            logger.cur_episode_reward += rew
            logger.agent_rewards[ag_idx][-1] += rew

        if done:
            obs_n = env.reset()
            obs_n = reshape_state(obs_n, arglist.history_size)
            episode_step = 0
            logger.record_episode_end(agents, arglist.display)

        logger.train_step += 1

        # policy updates
        train_cond = not display
        # if train_cond and len(agents[0].replay_buffer) > arglist.batch_size * arglist.max_episode_len:
        if train_cond and len(agents[0].replay_buffer) > arglist.batch_size + 1:
            for agent in agents:
                if logger.train_step % arglist.update_rate == 0:  # only update every 100 steps
                    q_loss, pol_loss = agent.update(agents, logger.train_step)

        # for displaying learned policies
        if display:
            time.sleep(0.1)
            env.render()

        # saves logger outputs to a file similar to the way in the original MADDPG implementation
        if len(logger.episode_rewards) > arglist.no_episodes:
            logger.experiment_end()
            return logger.get_sacred_results()


def get_agents(env, num_adversaries, good_policy, adv_policy, history_size, lr, batch_size,
               buff_size, num_units, num_layers, gamma, tau, priori_replay, alpha, num_episodes,
               max_episode_len, beta, policy_update_rate, critic_action_noise_stddev, no_neighbors, logger
               ) -> List[AbstractAgent]:
    """
    This function generates the agents for the environment. The parameters are meant to be filled
    by sacred, and are therefore documented in the configuration function train_config.

    :returns List[AbstractAgent] returns a list of instantiated agents
    """
    agents = []
    for agent_idx in range(num_adversaries):
        if adv_policy == 'maddpg':
            agent = MADDPGAgent(history_size, env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr, num_layers,
                                num_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta, logger=logger)
        elif adv_policy == "magat":
            agent = MAGATAgent(history_size, no_neighbors, env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers,
                               num_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta, logger=logger)
        elif adv_policy == 'matd3':
            agent = MATD3Agent(env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers,
                               num_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta,
                               policy_update_freq=policy_update_rate,
                               target_policy_smoothing_eps=critic_action_noise_stddev)
        else:
            raise RuntimeError('Invalid Class')
        agents.append(agent)
    for agent_idx in range(num_adversaries, env.n):
        if good_policy == 'maddpg':
            agent = MADDPGAgent(history_size, env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr, num_layers,
                                num_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta, logger=logger)
        elif good_policy == "magat":
            agent = MAGATAgent(history_size, no_neighbors, env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers,
                               num_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta, logger=logger)
        elif good_policy == 'matd3':
            agent = MATD3Agent(env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers, num_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta,
                               policy_update_freq=policy_update_rate,
                               target_policy_smoothing_eps=critic_action_noise_stddev)
        else:
            raise RuntimeError('Invalid Class')
        agents.append(agent)
    print('Using good policy {} and adv policy {}'.format(good_policy, adv_policy))
    return agents


def main():
    file_observer = os.path.join("results", arglist.exp_name)
    if not os.path.exists(file_observer):
        os.makedirs(file_observer)
    train(arglist.exp_name, arglist.save_rate, arglist.display, arglist.restore_fp)


if __name__ == '__main__':
    arglist = parse_args()
    main()
