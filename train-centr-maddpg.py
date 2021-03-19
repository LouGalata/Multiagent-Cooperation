import argparse
import os
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from buffers.replay_buffer_iql import EfficientReplayBuffer
from commons import util as u
from agents.centralized_maddpg import MADDPGAgent, MADDPGCriticNetwork


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=2, help="number of agents")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--no-episodes", type=int, default=30000, help="number of episodes")
    parser.add_argument("--no-neighbors", type=int, default=2, help="number of neigbors to cooperate")
    parser.add_argument("--seed", type=int, default=1, help="seed")

    # Experience Replay
    parser.add_argument("--max-buffer-size", type=int, default=1e6, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--loss-type", type=str, default="huber", help="Loss function: huber or mse")
    parser.add_argument("--use-gumbel", type=bool, default=False, help="Use Gumbel softmax")


    # Q-learning training parameters
    parser.add_argument("--no-layers", type=int, default=2, help="number dense layers")
    parser.add_argument("--no-neurons", type=int, default=64, help="number of neurons on the first gnn")
    parser.add_argument("--l2-reg", type=float, default=1e-2, help="kernel regularizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

    # Evaluation
    parser.add_argument("--restore-fp", type=bool, default=False, help="Load saved model")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--exp-name", type=str, default='self-maddpg2', help="name of the experiment")
    parser.add_argument("--update-rate", type=int, default=10,
                        help="update model once every time this many episodes are completed")
    parser.add_argument("--save-rate", type=int, default=300,
                        help="save model once every time this many episodes are completed")
    return parser.parse_args()


def get_agents(obs_shape_n, act_shape_n, path):
    agents = []
    for agent_idx in range(arglist.no_agents):
        fp = os.path.join(path, 'agent_{}'.format(agent_idx))
        agent = MADDPGAgent(obs_shape_n, act_shape_n, agent_idx, arglist.lr, arglist.no_layers, arglist.no_neurons,
                            fp, arglist.tau, arglist.use_gumbel)

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
    env = u.make_env(arglist.scenario, no_agents)

    obs_shape_n = u.space_n_to_shape_n(env.observation_space)
    act_shape_n = u.space_n_to_shape_n(env.action_space)
    # Result paths
    result_path = os.path.join("results", arglist.exp_name)
    res = os.path.join(result_path, "%s.csv" % arglist.exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    critic = MADDPGCriticNetwork(arglist.no_layers, arglist.no_neurons, arglist.lr, obs_shape_n, act_shape_n)
    critic_target = MADDPGCriticNetwork(arglist.no_layers, arglist.no_neurons, arglist.lr, obs_shape_n, act_shape_n)
    critic_target.model.set_weights(critic.model.get_weights())

    agents = get_agents(obs_shape_n, act_shape_n, result_path)

    obs_n = env.reset()
    replay_buffer = EfficientReplayBuffer(int(arglist.max_buffer_size), no_agents, obs_shape_n, act_shape_n)  # Init Buffer
    k_lst = list(range(arglist.no_neighbors + 2))[2:]
    # Load previous results if necessary
    if arglist.restore_fp:
        print('Loading previous state...')
        for ag_idx, agent in enumerate(agents):
            fp = os.path.join(result_path, 'agent_{}'.format(ag_idx))
            agent.load(fp)


    episode_step = 0
    train_step = 0
    episode_rewards = [0.0]  # sum of rewards for all agents

    testing_rewards = []
    print('Starting iterations...')
    while True:
        episode_step += 1

        action_n = [agent.action(obs.astype(np.float32)) for agent, obs in zip(agents, obs_n)]
        action_n = [action.numpy() for action in action_n]
        new_obs_n, rew_n, done_n, _ = env.step(action_n)

        terminal = (episode_step >= arglist.max_episode_len)
        done = all(done_n) or terminal
        cooperative_reward = rew_n[0]
        # collect experience
        replay_buffer.add(obs_n, action_n, cooperative_reward, new_obs_n, done)
        obs_n = new_obs_n

        cooperative_reward = rew_n[0]
        episode_rewards[-1] += cooperative_reward

        if arglist.restore_fp:
            testing_rewards.append(cooperative_reward)
        if done:
            if arglist.restore_fp:
                pd.DataFrame(testing_rewards).to_csv("results/" + arglist.exp_name + "/testing_rewards.csv")
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)

        train_step += 1

        if not arglist.restore_fp:
            pol_loss_total = []
            if len(replay_buffer) >= arglist.batch_size * arglist.max_episode_len:
                if train_step % arglist.update_rate == 0:
                    # Sample: Shapes --> (no-agents, batch_size, features)
                    state, actions, rewards, new_state, dones = replay_buffer.sample(arglist.batch_size)
                    target_act_next = [a.target_action(obs) for a, obs in zip(agents, new_state)]
                    target_q_next = critic_target.predict(new_state, target_act_next)
                    q_train_target = rewards + (1. - dones) * arglist.gamma * target_q_next

                    loss, _ = critic.train_step(state, actions, q_train_target)
                    update_target_networks(critic, critic_target)
                    critic.save(result_path)

                    for agent in agents:
                        pol_loss = agent.update(state, actions, critic, train_step)
                        pol_loss_total.append(pol_loss.numpy())
                if train_step % arglist.save_rate == 0:
                    with open(res, "a+") as f:
                        mes_dict = {"steps": train_step, "episodes": len(episode_rewards),
                                    "train_episode_reward": np.round(np.mean(episode_rewards[-arglist.save_rate:]), 3),
                                    "critic loss": np.round(loss.numpy(), 3),
                                    "policy loss": pol_loss_total}
                        print(mes_dict)
                        for item in list(mes_dict.values()):
                            f.write("%s\t" % item)
                        f.write("\n")
                        f.close()

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()


if __name__ == '__main__':
    tf.autograph.set_verbosity(0, True)
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    arglist = parse_args()
    main()
