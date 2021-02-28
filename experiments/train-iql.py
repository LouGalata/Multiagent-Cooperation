import argparse
import os
import pickle
import sys
import time
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras import Sequential

from replay_buffer_iql import ReplayBuffer


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
    parser.add_argument("--max-buffer-size", type=int, default=20000, help="maximum buffer capacity")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")

    # GNN training parameters
    parser.add_argument("--num-neurons", type=int, default=32, help="number of neurons on the first gnn")
    parser.add_argument("--l2-reg", type=float, default=2.5e-4, help="kernel regularizer")

    # Q-learning training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

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


def IQL_net(arglist):
    I1 = Input(shape=(feature_dim,), name="gcn_input")
    dense = Dense(arglist.num_neurons,
                  kernel_initializer=tf.keras.initializers.he_uniform(),
                  activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                  name="dense_layer")(I1)
    med_dense = Dense(arglist.num_neurons,
                  kernel_initializer=tf.keras.initializers.he_uniform(),
                  activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                  name="med_layer")(dense)

    last_dense = Dense(num_actions, kernel_initializer=tf.keras.initializers.he_uniform(),
                       activation=tf.keras.activations.softmax,
                       name="last_dense_layer")(med_dense)
    model = Model(I1, last_dense)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=arglist.lr),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['acc']
                  )
    return model


def get_actions(state, nets):
    predictions = []
    for st, net in zip(state, nets):
        st = tf.expand_dims(st, axis=0)
        pred_n = net.predict(st)
        predictions.append(pred_n)
    return predictions


def sample_actions_from_distr(predictions):
    prob = np.array(predictions)
    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    action = dist.sample()
    return [int(x[0]) for x in action.numpy()]


def __build_conf():
    hparams_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', arglist.exp_name))
    logdir = os.path.join(hparams_log_dir, "hidden-units=%d-batch-size=%d" %
                          (arglist.num_neurons, arglist.batch_size))

    model = [IQL_net(arglist) for _ in range(no_agents)]
    model_t = [IQL_net(arglist) for _ in range(no_agents)]
    for mdl, mdl_t in zip(model, model_t):
        mdl_t.set_weights(mdl.get_weights())

    callbacks = [__get_callbacks(logdir, i) for i in range(no_agents)]
    return model, model_t, callbacks


def update_q_values(arglist, batch_size, no_agents, actions, rewards, dones, q_values, target_q_values):
    for j in range(no_agents):
        for k in range(batch_size):
            q_values[j][k][actions[j][k]] = rewards[j][k] + arglist.gamma * (1.0 - float(dones[j][k])) * np.max(
                target_q_values[j][k])
    return q_values


def main(arglist):
    global num_actions, feature_dim, no_agents
    env = make_env(arglist.scenario)
    env.discrete_action_input = True

    obs_shape_n = env.observation_space
    no_agents = env.n
    batch_size = arglist.batch_size

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

        predictions = get_actions(to_tensor(np.array(obs_n)), model)
        actions = sample_actions_from_distr(predictions)
        # Observe next state, reward and done value
        new_obs_n, rew_n, done_n, _ = env.step(actions)
        done = all(done_n)
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

            # Calculate TD-target
            q_values = [net.predict(st) for net, st in zip(model, state)]
            target_q_values = [net.predict(st) for net, st in zip(model_t, new_state)]
            tt = time.time()
            q_values = update_q_values(arglist, batch_size, no_agents, actions, rewards, dones, q_values,
                                       target_q_values)
            print("Step %d - Update Q values time: %.3f " % (train_step, tt - time.time()))

            for cnt, (net, st, y) in enumerate(zip(model, state, q_values)):
                net.fit(st, y, epochs=50, batch_size=batch_size, verbose=0, callbacks=callback[cnt])

            # train target model
            for mdl, mdl_t in zip(model, model_t):
                weights = mdl.get_weights()
                target_weights = mdl_t.get_weights()
                for w in range(len(weights)):
                    target_weights[w] = arglist.tau * weights[w] + (1 - arglist.tau) * target_weights[w]
                mdl_t.set_weights(target_weights)

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
    print(tf.config.list_physical_devices('GPU'))
    np.set_printoptions(threshold=sys.maxsize)
    arglist = parse_args()
    create_seed(arglist.seed)
    main(arglist)
