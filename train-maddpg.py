import argparse
import os
from typing import List

import numpy as np

from commons import util as u
from models.AbstractAgent import AbstractAgent
from models.maddpg import MADDPGAgent


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
    parser.add_argument("--max-buffer-size", type=int, default=500000, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--loss-type", type=str, default="huber", help="Loss function: huber or mse")
    parser.add_argument("--soft-update", type=bool, default=True, help="Mode of updating the target network")

    # Exploration strategies
    parser.add_argument("--decay-mode", type=str, default="exp2", help="linear or exp")
    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon exploration")
    parser.add_argument("--e-lin-decay", type=float, default=0.0001, help="linear epsilon decay")
    parser.add_argument("--epsilon-decay", type=float, default=0.0003, help="exponantial epsilon decay")
    parser.add_argument("--min-epsilon", type=float, default=0.01, help="min epsilon")
    parser.add_argument("--max-epsilon", type=float, default=1.0, help="max epsilon")

    # Prioritized Experience Replay
    parser.add_argument("--alpha", type=float, default=0.6, help="")
    parser.add_argument("--initial-beta", type=float, default=0.6, help="")

    # GNN training parameters
    parser.add_argument("--no-neurons", type=int, default=64, help="number of neurons on the first gnn")
    parser.add_argument("--l2-reg", type=float, default=2.5e-4, help="kernel regularizer")

    # Q-learning training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

    # Evaluation
    parser.add_argument("--restore-fp", type=bool, default=False, help="Load saved model")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--exp-name", type=str, default='maddpg2', help="name of the experiment")
    parser.add_argument("--update-rate", type=int, default=100,
                        help="update model once every time this many episodes are completed")
    parser.add_argument("--save-rate", type=int, default=50,
                        help="save model once every time this many episodes are completed")
    return parser.parse_args()


def get_agents(env, path) -> List[AbstractAgent]:
    agents = []
    for agent_idx in range(arglist.no_agents):
        path = os.path.join(path, 'agent_{}'.format(agent_idx))
        agent = MADDPGAgent(env.observation_space, env.action_space, agent_idx, arglist.batch_size,
                            arglist.max_buffer_size, arglist.lr, arglist.no_neurons, arglist.gamma, path, arglist.tau,
                            alpha=arglist.alpha,
                            max_step=arglist.no_episodes * arglist.max_episode_len,
                            initial_beta=arglist.initial_beta)

        agents.append(agent)
    return agents


def main():
    no_agents = arglist.no_agents
    env = u.make_env(arglist.scenario, no_agents)

    # Result paths
    result_path = os.path.join("results", arglist.exp_name)
    res = os.path.join(result_path, "%s.csv" % arglist.exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    agents = get_agents(env, result_path)
    obs_n = env.reset()


    # Load previous results if necessary
    if arglist.restore_fp:
        print('Loading previous state...')
        for ag_idx, agent in enumerate(agents):
            fp = os.path.join(result_path, 'agent_{}'.format(ag_idx))
            agent.load(fp)

    episode_step = 0
    train_step = 0
    episode_rewards = [0.0]  # sum of rewards for all agents
    print('Starting iterations...')
    while True:
        episode_step += 1

        action_n = [agent.action(obs.astype(np.float32)) for agent, obs in zip(agents, obs_n)]
        action_n = [action.numpy() for action in action_n]
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        done = done or terminal

        # collect experience
        for i, agent in enumerate(agents):
            agent.add_transition(obs_n, action_n, rew_n[i], new_obs_n, done)
        obs_n = new_obs_n

        cooperative_reward = rew_n[0]
        episode_rewards[-1] += cooperative_reward

        if done:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)

        train_step += 1

        q_loss_total = 0
        if len(agents[0].replay_buffer) > arglist.batch_size * arglist.max_episode_len:
            for agent in agents:
                if train_step % arglist.update_rate == 0:  # only update every 100 steps
                    q_loss, pol_loss = agent.update(agents, train_step)
                    q_loss_total += q_loss


if __name__ == '__main__':
    arglist = parse_args()
    main()
