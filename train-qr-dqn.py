import argparse
import os
import pickle
import sys
import time
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from models.qr_dqn import ActionValueModel
from utils.replay_buffer_iql import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=4, help="number of agents")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=30000, help="number of episodes")
    parser.add_argument("--num-neighbors", type=int, default=2, help="number of neigbors to cooperate")
    parser.add_argument("--seed", type=int, default=1, help="seed")

    # Experience Replay
    parser.add_argument("--max-buffer-size", type=int, default=500000, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")

    # GNN training parameters
    parser.add_argument("--num-neurons", type=int, default=32, help="number of neurons on the first gnn")
    parser.add_argument("--l2-reg", type=float, default=2.5e-4, help="kernel regularizer")

    # Q-learning training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")
    parser.add_argument("--no-atoms", type=int, default=51, help="number of quantiles in quantile regression ")

    # Evaluation
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--exp-name", type=str, default='GAT-exp', help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=50,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")

    return parser.parse_args()


def to_tensor(arg):
    arg = tf.convert_to_tensor(arg)
    return arg


def create_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    # Here is defined the num_agents
    world = scenario.make_world(no_agents=arglist.no_agents, seed = arglist.seed)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def __get_callbacks(logdir, idx):
    path = "agent %d " % idx
    callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                 tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
                 tf.keras.callbacks.TensorBoard(logdir, update_freq="epoch", profile_batch=0),
                 tf.keras.callbacks.ModelCheckpoint(
                     filepath=os.path.join(path + '_' + logdir, "cp.ckpt"),
                     save_best_only=True,
                     save_freq=25,
                     save_weights_only=False,
                     monitor='loss',
                     verbose=0)
                 ]
    return callbacks


def __build_conf():
    hparams_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', arglist.exp_name))
    logdir = os.path.join(hparams_log_dir, "hidden-units=%d-batch-size=%d" %
                          (arglist.num_neurons, arglist.batch_size))

    model = [ActionValueModel(arglist, feature_dim, num_actions) for _ in range(no_agents)]
    model_t = [ActionValueModel(arglist, feature_dim, num_actions) for _ in range(no_agents)]
    for mdl, mdl_t in zip(model, model_t):
        mdl_t.model.set_weights(mdl.model.get_weights())

    callbacks = [__get_callbacks(logdir, i) for i in range(no_agents)]
    return model, model_t, callbacks


def update_q_values(arglist, batch_size, no_agents, actions, rewards, dones, q_values, target_q_values):
    for j in range(no_agents):
        for k in range(batch_size):
            q_values[j][k][actions[j][k]] = rewards[j][k] + arglist.gamma * (1.0 - float(dones[j][k])) * np.max(
                target_q_values[j][k])
    return q_values


def main(arglist):
    global num_actions, feature_dim, no_agents, no_atoms
    env = make_env(arglist.scenario)
    env.discrete_action_input = True

    obs_shape_n = env.observation_space
    no_agents = env.n
    batch_size = arglist.batch_size
    no_atoms = arglist.no_atoms

    # Velocity.x Velocity.y Pos.x Pos.y {Land.Pos.x Land.Pos.y}*10 {Ent.Pos.x Ent.Pos.y}*9
    num_features = obs_shape_n[0].shape[0]
    num_actions = env.action_space[0].n
    feature_dim = num_features  # the size of node features
    model, model_t, callback = __build_conf()

    # Results
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    result_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, arglist.exp_name + "/rewards-per-episode.csv"))
    if not os.path.exists(result_path):
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

    replay_buffer = ReplayBuffer(arglist.max_buffer_size)  # Init Buffer
    episode_step = 0
    train_step = 0

    t_start = time.time()
    obs_n = env.reset()

    print('Starting iterations...')
    while True:
        episode_step += 1
        terminal = (episode_step >= arglist.max_episode_len)
        smp_actions = [mdl.get_action(st, train_step) for mdl, st in zip(model, obs_n)]
        actions = []
        for act in smp_actions:
            action_ = np.zeros(num_actions)
            action_[act] = 1
            actions.append(action_)
        # Observe next state, reward and done value
        new_obs_n, rew_n, done_n, _ = env.step(smp_actions)
        done = all(done_n) or terminal
        # Store the data in the replay memory
        replay_buffer.add(obs_n, actions, rew_n, new_obs_n, done_n)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # Train the models
        if replay_buffer.can_provide_sample(batch_size) and train_step % 100 == 0:
            # Pass a batch of states through the policy network to calculate the Q(s, a)
            # Pass a batch of states through the target network to calculate the Q'(s', a)
            batch = replay_buffer.sample(batch_size)
            state, actions, rew_n, new_state, done_n = [], [], [], [], []
            # for e in batch:
            for e in range(batch_size):
                state.append(batch[0][e])
                new_state.append(batch[3][e])
                actions.append(batch[1][e])
                rew_n.append(batch[2][e])
                done_n.append(batch[4][e])\

            actions = np.swapaxes(actions, 0, 1)
            rewards = np.swapaxes(rew_n, 0, 1)
            dones = np.swapaxes(done_n, 0, 1)
            state = np.swapaxes(state, 0, 1)
            new_state = np.swapaxes(new_state, 0, 1)
            # shape of q: (no_agenets, batch_size, no_actions, no_atoms)
            # shape of rewards: (no_agents, batch_size)

            q = [net.predict(st) for net, st in zip(model_t, new_state)]
            next_actions = [np.argmax(np.mean(q_n, axis=2), axis=1) for q_n in q]
            thetas = np.empty(no_agents, dtype=np.object)
            for j in range(no_agents):
                theta = []
                for k in range(batch_size):
                    if dones[j][k]:
                        theta.append(np.ones(no_atoms) * rewards[j][k])
                    else:
                        theta.append(rewards[j][k] + arglist.gamma * q[j][k][next_actions[j][k]])
                thetas[j] = theta

            for net, st, y, act in zip(model, state, thetas, actions):
                net.train(st, y, act)

            # train target model
            for mdl, mdl_t in zip(model, model_t):
                weights = mdl.model.get_weights()
                target_weights = mdl_t.model.get_weights()
                for w in range(len(weights)):
                    target_weights[w] = arglist.tau * weights[w] + (1 - arglist.tau) * target_weights[w]
                mdl_t.model.set_weights(target_weights)

        # display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            with open(result_path, "a+") as f:
                mes_dict = {"steps": train_step, "episodes": len(episode_rewards),
                            "mean_episode_reward" : round(np.mean(episode_rewards[-arglist.save_rate:]), 3),
                            "time": round(time.time() - t_start, 3)}
                print(mes_dict)
                for item in list(mes_dict.values()):
                    f.write("%s\t" % item)
                f.write("\n")
                f.close()
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                if not os.path.exists(arglist.plots_dir):
                    os.makedirs(arglist.plots_dir)
                rew_file_name = arglist.plots_dir + '/' + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                break


if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')
    np.set_printoptions(threshold=sys.maxsize)
    arglist = parse_args()
    create_seed(arglist.seed)
    main(arglist)
