import argparse
import os
import random
import sys
import time
import pickle

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Attention, GRU
from keras.models import Model
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from spektral.layers import GATConv
from tensorflow.keras import Sequential
import tensorflow_probability as tfp

from replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--no-agents", type=int, default=6, help="number of agents")
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
    parser.add_argument("--num-neurons", type=int, default=16, help="number of neurons on the first gnn")
    parser.add_argument("--l2-reg", type=float, default=2.5e-4, help="kernel regularizer")

    # Q-learning training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

    # Evaluation
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--exp-name", type=str, default='GAT-rnn-exp', help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=50, help="save model once every time this many episodes are completed")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

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
    world = scenario.make_world(no_agents=arglist.no_agents)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def __get_callbacks(logdir):
    callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                 tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=5),
                 tf.keras.callbacks.TensorBoard(logdir, update_freq="epoch", profile_batch=0),

                 tf.keras.callbacks.ModelCheckpoint(
                     filepath=os.path.join(logdir, "cp.ckpt"),
                     save_best_only=True,
                     save_freq=25,
                     save_weights_only=False,
                     monitor='loss',
                     verbose=0)
                 ]
    return callbacks


def get_adj(arr, k_lst):
    """
    Take as input the new obs. In position 4 to k, there are the x and y coordinates of each agent
    Make an adjacency matrix, where each agent communicates with the k closest ones
    """
    # Try both with normal and sparce matrix
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


def get_model(arglist):
    I1 = Input(shape=(no_agents, feature_dim), name="gcn_input")
    Adj = Input(shape=(no_agents, no_agents), name="adj")

    # gcn = GCNConv(n_neurons, kernel_initializer=tf.keras.initializers.he_uniform(),
    #               activation=tf.keras.layers.LeakyReLU(alpha=0.1),
    #               use_bias=False,
    #               name="Gcn")([I1, Adj])

    gat = GATConv(
        arglist.num_neurons,
        attn_heads=4,
        concat_heads=True,
        dropout_rate=0.6,
        activation="elu",
        kernel_regularizer=tf.keras.regularizers.l2(arglist.l2_reg),
        attn_kernel_regularizer=tf.keras.regularizers.l2(arglist.l2_reg),
        bias_regularizer=tf.keras.regularizers.l2(arglist.l2_reg),
    )([I1, Adj])


    output = []
    dense = Dense(arglist.num_neurons,
                  kernel_initializer=tf.keras.initializers.he_uniform(),
                  activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                  name="dense_layer")


    # attention = Attention(use_scale=True)
    last_dense = Dense(num_actions, kernel_initializer=tf.keras.initializers.he_uniform(),
                       activation=tf.keras.activations.softmax,
                       name="last_dense_layer")
    split = Lambda(lambda x: tf.squeeze(tf.split(x, num_or_size_splits=no_agents, axis=1), axis=2))(gat)
    for j in list(range(no_agents)):
        V = dense(split[j])
        V = tf.expand_dims(V, axis=1)
        # att = attention(V)
        gru = GRU(num_actions)(V)
        # V2 = last_dense(V)
        output.append(gru)

    model = Model([I1, Adj], output)
    model._name = "final_network"
    # output = Concatenate()(output)
    # vdn_model =
    # model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)
    return model


def get_actions(graph, adj, gcn_net):
    graph = tf.expand_dims(graph, axis=0)
    adj = tf.expand_dims(adj, axis=0)
    preds = gcn_net.predict([graph, adj])
    return preds


def sample_actions_from_distr(predictions):
    """
    Return a random action with probability epsilon and the max with probability 1 - epsilon for each agent
    return [random.randrange(num_actions) if np.random.rand() < epsilon else np.argmax(prediction) for prediction in
            predictions]
    """
    prob = np.array(predictions)
    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    action = dist.sample()
    return [int(x[0]) for x in action.numpy()]


def __build_conf():
    # Configure logging
    hparams_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', arglist.exp_name))
    logdir = os.path.join(hparams_log_dir, "hidden-untis=%d-batch-size=%d" %
                          (arglist.num_neurons, arglist.batch_size))

    model = get_model(arglist)
    model_t = get_model(arglist)
    model_t.set_weights(model.get_weights())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=arglist.lr, clipnorm=1.0, clipvalue=0.5),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['acc']
                  )
    # loss=tf.keras.losses.Huber()
    model_t.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=arglist.lr, clipnorm=1.0, clipvalue=0.5),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=['acc']
                    )

    callbacks = __get_callbacks(logdir)

    return model, model_t, callbacks


def main(arglist):
    # Global variables
    global num_actions, feature_dim, no_agents
    # Create environment
    env = make_env(arglist.scenario)
    env.discrete_action_input = True

    obs_shape_n = env.observation_space
    no_agents = env.n
    batch_size = arglist.batch_size
    no_neighbors = arglist.num_neighbors
    k_lst = list(range(no_neighbors + 2))[2:]  # [2,3]


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

    replay_buffer = ReplayBuffer(arglist.max_buffer_size)     # Init Buffer
    episode_step = 0
    train_step = 0

    t_start = time.time()
    obs_n = env.reset()
    adj = get_adj(obs_n, k_lst)

    print('Starting iterations...')
    while True:
        episode_step += 1
        terminal = (episode_step >= arglist.max_episode_len)
        if episode_step % 3 == 0:
            adj = get_adj(obs_n, k_lst)

        predictions = get_actions(to_tensor(np.array(obs_n)), adj, model)
        actions = sample_actions_from_distr(predictions)
        # Observe next state, reward and done value
        new_obs_n, rew_n, done_n, _ = env.step(actions)
        done = all(done_n)
        # Store the data in the replay memory
        replay_buffer.add(obs_n, adj, actions, rew_n, new_obs_n, done_n)
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
            state, adj_n, actions, rew_n, new_state, done_n = [], [], [], [], [], []
            # for e in batch:
            for e in range(batch_size):
                state.append(batch[0][e])
                new_state.append(batch[4][e])
                adj_n.append(batch[1][e])
                actions.append(batch[2][e])
                rew_n.append(batch[3][e])
                done_n.append(batch[5][e])
            actions = np.asarray(actions)
            rewards = np.asarray(rew_n)
            dones = np.asarray(done_n)
            adj_n = np.asarray(adj_n)
            state = np.asarray(state)
            new_state = np.asarray(new_state)

            # Calculate TD-target
            q_values = model.predict([state, adj_n])
            target_q_values = model_t.predict([new_state, adj_n])

            for k in range(batch_size):
                for j in range(no_agents):
                    q_values[j][k][actions[k][j]] = rewards[k][j] + arglist.gamma * (1.0 - float(dones[k][j])) * np.max(
                        target_q_values[j][k])

            model.fit([state, adj_n], q_values, epochs=50, batch_size=batch_size, verbose=0, callbacks=callback)

            # train target model
            weights = model.get_weights()
            target_weights = model_t.get_weights()

            for w in range(len(weights)):
                target_weights[w] = arglist.tau * weights[w] + (1 - arglist.tau) * target_weights[w]
            model_t.set_weights(target_weights)

        # display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            with open(result_path, "a+") as f:
                mes_dict = {"steps": train_step, "episodes": len(episode_rewards),
                            "mean_episode_reward": round(np.mean(episode_rewards[-arglist.save_rate:]), 3),
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
