import argparse
import os
import random
import sys

import numpy as np
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Dropout
from keras.models import Model
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from spektral.layers import GCNConv
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from experiments.replay_buffer import ReplayBuffer

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--episode-before-train", type=int, default=200, help="number of episodes before train")

    # Experince Replay
    parser.add_argument("--max-buffer-size", type=int, default=20000, help="maximum buffer capacity")

    # Core training parameters  -- Passed through hparams
    # parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    # parser.add_argument("--batch-size", type=int, default=1, help="number of episodes to optimize at the same time")
    # parser.add_argument("--dropout-rate", type=float, default=0.05, help="dropout rate in the gcn")

    # GCN training parameters -- Passed through hparams
    # parser.add_argument("--num-neurons1", type=int, default=24, help="number of neurons on the first gcn")
    # parser.add_argument("--num-neurons2", type=int, default=24, help="number of neurons on the second gcn")

    # Q-learning training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="smooth weights copy to target model")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)

    # benchmark: provides diagnostic data for policies trained on the environment
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    # Here is defined the num_agents
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def all_equal2(iterator):
    return len(set(iterator)) <= 1


def __get_callbacks(logdir, hparams):
    callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                 tf.keras.callbacks.TensorBoard(logdir,
                                                update_freq='epoch',
                                                write_graph=False,
                                                histogram_freq=5),

                 hp.KerasCallback(logdir, hparams, trial_id=logdir),
                 tf.keras.callbacks.ModelCheckpoint(
                     filepath=os.path.join(logdir, "cp.ckpt"),
                     save_best_only=True,
                     monitor='epoch_loss',
                     verbose=1)
                 ]
    return callbacks


def get_adj(arr, k=3):
    """
    Take as input the new obs. In position 2 to k, there are the x and y coordinates of each agent
    Make an adjacency matrix, where each agent communicates with the k closest ones
    """
    length = len(arr)
    # Try both with normal and sparce matrix
    # adjacency = np.zeros((length, length))
    x_idx = np.repeat(list(range(length)), k)

    k_lst = list(range(k + 2))[2:]  # [2,3]
    neighbors = []
    # construct a kd-tree
    tree = cKDTree(arr)
    for cnt, row in enumerate(arr):
        # find k nearest neighbors for each element of data, squeezing out the zero result (the first nearest
        # neighbor is always itself)
        dd, ii = tree.query(row, k=k_lst)
        # apply an index filter on data to get the nearest neighbor elements
        neighbors.append(ii)
        # adjacency[cnt, ii] = 1.0

    y_idx = np.concatenate(neighbors).ravel()
    data = np.ones((length * k))
    # Use a sparce matrix
    adj = csr_matrix((data, (x_idx, y_idx)), shape=(length, length))
    return adj


def GCN_net(n_neurons_1=None, n_neurons_2=None, dropout=None):
    I1 = Input(shape=(feature_dim,), name="gcn_input")
    Adj = Input((no_agents,), sparse=True, name="adj")

    encoder = GCNConv(channels=n_neurons_1, activation='relu', name="Encoder")([I1, Adj])
    decoder = GCNConv(channels=n_neurons_2, activation='relu', name="Decoder")([encoder, Adj])

    q_net_input = tf.expand_dims(decoder, axis=0)
    output = []
    for j in list(range(no_agents)):
        T = Lambda(lambda x: x[:, j], output_shape=(1, n_neurons_2,), name="lambda_layer_agent_%d" % j)(
            q_net_input)
        V1 = Dense(n_neurons_2, kernel_initializer='random_normal', activation='relu', name="FirstDense_agent_%d" % j)(
            T)
        # Drop = Dropout(dropout, name="FirstDense_agent_%d" % j)(V1),
        V = Dense(num_actions, kernel_initializer='random_normal', activation='softmax',
                  name="SecondDense_agent_%d" % j)(V1)
        output.append(V)

    model = Model([I1, Adj], output)
    model._name = "final_network"
    # model.summary()
    # tf.keras.utils.plot_model(model, show_shapes=True)
    return model


def get_actions(graph, adj, gcn_net):
    A = GCNConv.preprocess(adj).astype('f4')
    preds = gcn_net.predict([graph, A])
    return preds


def get_actions_egreedy(predictions, epsilon=None):
    """
    Return a random action with probability epsilon and the max with probability 1 - epsilon for each agent
    """
    return [random.randrange(num_actions) if np.random.rand() < epsilon else np.argmax(prediction) for prediction in
            predictions]

def __get_model(n_neurons_1=None, n_neurons_2=None, dropout=None):
    return GCN_net(n_neurons_1, n_neurons_2, dropout)

def __build_configurations():
    configurations = []
    for dr in HP_DROPOUT.domain.values:
        for hu in HP_HIDDEN_UNITS.domain.values:
            for lr in HP_LEARNING_RATE.domain.values:
                for bs in HP_BATCH_SIZE.domain.values:
                    for gamma in HP_GAMMA.domain.values:
                        for cg in HP_CLIP_GRADIENT.domain.values:
                            new = [dr, hu, lr, bs, gamma, cg]
                            configurations.append(new)
    return configurations

def __build_conf(dropout, hidden_units, learning_rate, batch_size, gamma, clip_gradient):
    # Configure logging
    hparams_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    logdir = os.path.join(hparams_log_dir, "dr=%s-hu=%d-lr=%s-bs=%d-gamma=%s-cg=%s" %
                          (dropout, hidden_units, learning_rate, batch_size, gamma,clip_gradient))


    if os.path.exists(logdir):
        print("Ignoring run %s" % logdir)
        return None, None, None

    model = __get_model(n_neurons_1=hidden_units, n_neurons_2=hidden_units, dropout=dropout)
    model_t = __get_model(n_neurons_1=hidden_units, n_neurons_2=hidden_units, dropout=dropout)

    if clip_gradient:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0, clipvalue=0.5),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[METRICS]
                      )
        model_t.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0, clipvalue=0.5),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[METRICS]
                      )
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[METRICS]
                      )
        model_t.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[METRICS]
                      )

    with tf.summary.create_file_writer(hparams_log_dir).as_default():
        hp.hparams_config(
            hparams=[HP_HIDDEN_UNITS, HP_DROPOUT, HP_LEARNING_RATE, HP_BATCH_SIZE, HP_GAMMA],
            metrics=[
                hp.Metric('epoch_loss', group="train", display_name='epoch_loss'),
                hp.Metric('mape', group="train", display_name='mape'),
                hp.Metric('mae', group="train", display_name='mae'),
                hp.Metric('rmse', group="train", display_name='rmse'),
                hp.Metric('epoch_mape', group="train", display_name='mape'),
                hp.Metric('epoch_mae', group="train", display_name='mae'),
                hp.Metric('epoch_rmse', group="train", display_name='rmse')
            ],
        )

    hparams = {
        HP_DROPOUT: dropout,
        HP_HIDDEN_UNITS: hidden_units,
        HP_LEARNING_RATE: learning_rate,
        HP_BATCH_SIZE: batch_size,
        HP_GAMMA: gamma,
        HP_CLIP_GRADIENT: clip_gradient
    }
    callbacks = __get_callbacks(logdir, hparams)

    return model, model_t, callbacks


def main(arglist):
    # Global variables
    global num_actions, feature_dim, no_agents
    # Create environment
    env = make_env(arglist.scenario, arglist.benchmark)
    env.discrete_action_input = True

    obs_shape_n = env.observation_space
    no_agents = env.n
    # The last 2* (n-1) positions are the communication channel (not used)
    # The next 2* (n-1) positions are the relative positions of other agents. (Not used)
    num_features = obs_shape_n[0].shape[0]
    num_actions = env.action_space[0].n
    feature_dim = num_features - (env.n - 1) * 4  # the size of node features

    configurations = __build_configurations()
    for configuration in configurations:
        model, model_t, callback = __build_conf(*configuration)
        _, _, _, batch_size, gamma, _ = configuration
        if model is None:
            continue

        ###########playing#############
        i_episode = 0
        replay_buffer = ReplayBuffer(arglist.max_buffer_size)
        epsilon = 1.0
        while i_episode < arglist.num_episodes:
            i_episode += 1
            epsilon *= 0.996
            if epsilon < 0.01: alpha = 0.01
            print(i_episode)
            obs_n = env.reset()
            score = 0
            steps = 0
            loss = 0
            while steps < arglist.max_episode_len:
                steps += 1

                obs_n = [x[:(num_features - (env.n - 1) * 4)] for x in obs_n]
                adj = get_adj(obs_n)
                predictions = get_actions(np.array(obs_n), adj, model)
                actions = get_actions_egreedy(predictions, epsilon=epsilon)
                # Observe next state, reward and done value
                new_obs_n, rew_n, done_n, _ = env.step(actions)
                new_obs_n = [x[:(num_features - (env.n - 1) * 4)] for x in new_obs_n]

                # Store the data in the replay memory
                replay_buffer.add(obs_n, adj, actions, rew_n, new_obs_n, done_n)

                score += sum(rew_n)
                if (i_episode - 1) % 10 == 0:
                    env.render()

                if arglist.num_episodes == steps:
                    print(score, end='\t')

                if i_episode < arglist.episode_before_train:
                    continue

                # Pass a batch of states through the policy network to calculate the Q(s, a)
                # Pass a batch of states through the target network to calculate the Q'(s', a)
                batch = replay_buffer.sample(batch_size)
                obs_n, adj_n, actions, rew_n, new_obs_n, done_n = [], [], [], [], [], []
                # for e in batch:
                for e in range(batch_size):
                    obs_n.append(batch[0][e])
                    new_obs_n.append(batch[4][e])
                    GCNConv.preprocess(batch[1][e]).astype('f4')
                    adj_n.append(batch[1][e])
                    actions.append(batch[2][e])
                    rew_n.append(batch[3][e])
                    done_n.append(batch[5][e])

                actions = np.asarray(actions)
                rewards = np.asarray(rew_n)
                dones = np.asarray(done_n)

                actions = np.asarray(actions)
                rewards = np.asarray(rew_n)
                dones = np.asarray(done_n)
                for j in range(env.n):
                    obs_n[j] = np.asarray(obs_n[j])
                    adj_n[j] = np.array(adj_n[j])
                    new_obs_n[j] = np.asarray(new_obs_n[j])


                # Calculate TD-target
                q_values = model.predict([adj_n, obs_n])
                target_q_values = model_t.predict([adj_n, obs_n])

                for k in range(len(batch)):
                    if dones[k]:
                        for j in range(no_agents):
                            q_values[j][k][actions[k][j]] = rewards[k][j]
                    else:
                        for j in range(no_agents):
                            q_values[j][k][actions[k][j]] = rewards[k][j] + HP_GAMMA * np.max(target_q_values[j][k])
                history = model.fit(obs_n, q_values, epochs=1, batch_size=batch_size, verbose=1, callbacks=callback)

                #### <!---------????
                his = 0
                for (k, v) in history.history.items():
                    his += v[0]
                loss += (his / 20)
                #### ---------> ????

                # train target model
                weights = model.get_weights()
                target_weights = model_t.get_weights()

                for w in range(len(weights)):
                    target_weights[w] = arglist.tau * weights[w] + (1 - arglist.tau) * target_weights[w]
                model_t.set_weights(target_weights)

            #######save model###############
            model.save('model.h5')


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    # Set hyper parameter search
    HP_HIDDEN_UNITS = hp.HParam('hidden_units', hp.Discrete([24]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.1]))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001, 0.0001]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1, 124]))
    HP_CLIP_GRADIENT = hp.HParam('clip_gradient', hp.Discrete([False]))
    HP_GAMMA = hp.HParam('gamma', hp.Discrete([0.95, 0.98]))

    # Define metrics to watch
    METRICS = [
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsolutePercentageError(name='mape')
    ]

    arglist = parse_args()
    main(arglist)
