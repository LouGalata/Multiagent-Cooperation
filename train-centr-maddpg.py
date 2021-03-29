import argparse
import os
import random
import time

import numpy as np
import tensorflow as tf
from commons.logger import RLLogger
from agents.centralized_maddpg import MADDPGAgent, MADDPGCriticNetwork
from buffers.replay_buffer_iql import EfficientReplayBuffer
from commons import util as u
from environments.multiagent.environment import MultiAgentEnv


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--exp-name", type=str, default='debug-centr', help="name of the experiment")

    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--restore_fp", action="store_true", default=None,
                        help="path to restore models from: e.g. 'results/maddpg7/models/'")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread_ivan", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=5, help="number of agents")
    parser.add_argument("--no-adversaries", type=int, default=0, help="number of adversaries")

    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--no-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--no-neighbors", type=int, default=2, help="number of neigbors to cooperate")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--reward", type=int, default=5, help="reward added if agents is close to the landmark")

    # Core training parameters
    parser.add_argument("--max-buffer-size", type=int, default=1e6, help="maximum buffer capacity")

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--no-critic-neurons", type=int, default=256, help="number of neurons on the first gnn")
    parser.add_argument("--no-actor-neurons", type=int, default=64, help="number of neurons on the first gnn")

    parser.add_argument("--use-ounoise", type=bool, default=True, help="Use Ornstein Uhlenbeck Process")
    parser.add_argument("--noise", type=float, default=0.1, help="Add noise on actions")
    parser.add_argument("--noise-reduction", type=float, default=0.999, help="Noise decay on actions")

    # Q-learning training parameters
    parser.add_argument("--no-layers", type=int, default=2, help="number dense layers")
    parser.add_argument("--no-neurons", type=int, default=64, help="number of neurons on the first gnn")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

    parser.add_argument("--use-target-action", type=bool, default=True, help="use action from target network")
    parser.add_argument("--hard-max", type=bool, default=False, help="Only output one action")

    parser.add_argument("--save-rate", type=int, default=20,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--update-rate", type=int, default=30,
                        help="update policy after each x steps")
    parser.add_argument("--update-times", type=int, default=20,
                        help="Number of times we update the networks")
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
    world = scenario.make_world(no_agents=arglist.no_agents, reward_const=arglist.reward)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_agents(env, lr, num_layers, num_actor_neurons, tau, noise, use_ounoise, logger):
    agents = []
    for agent_idx in range(arglist.no_agents):
        agent = MADDPGAgent(env.observation_space, env.action_space, agent_idx, lr,
                            num_layers, num_actor_neurons, tau, noise=noise,
                            use_ounoise=use_ounoise, logger=logger)
        agents.append(agent)
    return agents


def update_target_networks(critic_net, target_critic_net):
    def update_critic_network(net: MADDPGCriticNetwork, target_net: MADDPGCriticNetwork):
        net_weights = np.array(net.model.get_weights())
        target_net_weights = np.array(target_net.model.get_weights())
        new_weights = arglist.tau * net_weights + (1.0 - arglist.tau) * target_net_weights
        target_net.model.set_weights(new_weights)

    update_critic_network(critic_net, target_critic_net)


def main():
    no_agents = arglist.no_agents
    u.create_seed(arglist.seed)
    env = make_env(arglist.scenario)
    logger = RLLogger(arglist.exp_name, env.n, env.n_adversaries, arglist.save_rate, arglist)

    obs_shape_n = u.space_n_to_shape_n(env.observation_space)
    act_shape_n = u.space_n_to_shape_n(env.action_space)
    # Result paths
    model_path = os.path.join("results", arglist.exp_name, 'models')
    os.makedirs(model_path, exist_ok=True)

    critic = MADDPGCriticNetwork(arglist.no_layers, arglist.no_critic_neurons, arglist.lr, obs_shape_n, act_shape_n,
                                 wd=1e-5)
    critic_target = MADDPGCriticNetwork(arglist.no_layers, arglist.no_critic_neurons, arglist.lr, obs_shape_n,
                                        act_shape_n, wd=1e-5)
    critic_target.model.set_weights(critic.model.get_weights())

    agents = get_agents(env, arglist.lr, arglist.no_layers, arglist.no_actor_neurons,
                        arglist.tau, arglist.noise, arglist.use_ounoise, logger)

    obs_n = env.reset()
    replay_buffer = EfficientReplayBuffer(int(arglist.max_buffer_size), no_agents, obs_shape_n,
                                          act_shape_n)  # Init Buffer
    # Load previous results if necessary
    if arglist.restore_fp:
        print('Loading previous state...')
        for ag_idx, agent in enumerate(agents):
            fp = os.path.join(model_path, 'agent_{}'.format(ag_idx))
            agent.load(fp)
        critic.load(model_path + '/critic.h5')
        critic_target.load(model_path + '/critic_target.h5')

    print('Starting iterations...')
    while True:
        logger.episode_step += 1
        action_n = [agent.action(obs.astype(np.float32)).numpy() for agent, obs in zip(agents, obs_n)]
        new_obs_n, rew_n, done_n, _ = env.step(action_n)
        cooperative_reward = rew_n[0]
        terminal = (logger.episode_step >= arglist.max_episode_len)
        done = all(done_n) or terminal
        # collect experience
        replay_buffer.add(obs_n, action_n, cooperative_reward, new_obs_n, done)
        obs_n = new_obs_n

        if done:
            obs_n = env.reset()
            episode_step = 0
            logger.record_episode_end(agents, arglist.display)

        for ag_idx, rew in enumerate(rew_n):
            logger.cur_episode_reward += cooperative_reward
            logger.agent_rewards[ag_idx][-1] += cooperative_reward

        logger.train_step += 1
        train_cond = not arglist.display

        if train_cond and len(replay_buffer) > arglist.batch_size:
            if len(logger.episode_rewards) % arglist.update_rate == 0:  # only update every 30 episodes
                for _ in range(arglist.update_times):
                    # Sample: Shapes --> (no-agents, batch_size, features)
                    state, actions, rewards, new_state, dones = replay_buffer.sample(arglist.batch_size)
                    target_act_next = [a.target_action(obs) for a, obs in zip(agents, new_state)]
                    target_q_next = critic_target.predict(new_state, target_act_next)
                    q_train_target = rewards + (1. - dones) * arglist.gamma * target_q_next

                    loss, td_loss = critic.train_step(state, actions, q_train_target)
                    logger.save_logger("critic_loss", np.mean(td_loss), logger.train_step, 0)
                    update_target_networks(critic, critic_target)
                    critic.save(model_path + '/critic.h5')
                    critic_target.save(model_path + '/critic_target.h5')
                    for agent in agents:
                        pol_loss = agent.update(state, actions, critic, logger.train_step)

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()

        # saves logger outputs to a file similar to the way in the original MADDPG implementation
        if len(logger.episode_rewards) > arglist.no_episodes:
            logger.experiment_end()
            return logger.get_sacred_results()


def create_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == '__main__':
    arglist = parse_args()
    create_seed(arglist.seed)
    main()
