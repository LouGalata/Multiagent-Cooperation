import random
from gym.spaces import Box, Discrete
import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree
from spektral.layers import GCNConv, GATConv


def get_adj(arr, k_lst, no_agents, is_gat=False, is_gcn=False):
    """
    Take as input the new obs. In position 4 to k, there are the x and y coordinates of each agent
    Make an adjacency matrix, where each agent communicates with the k closest ones
    """
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
    if is_gcn:
        adj = GCNConv.preprocess(adj).astype('f4')
    elif is_gat:
        adj = GATConv.preprocess(adj).astype('f4')
    return adj


def clip_by_local_norm(gradients, norm):
    """
    Clips gradients by their own norm, NOT by the global norm
    as it should be done (according to TF documentation).
    This here is the way MADDPG does it.
    """
    for idx, grad in enumerate(gradients):
        gradients[idx] = tf.clip_by_norm(grad, norm)
    return gradients


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


def refresh_history(history, state_next):
    """
    Function that updates the history (a set of "n" frames that is used as a state of the replay memory)
    taking out the first frame, moving the rest and adding the new frame to end of the history.
    :param history : input volume of shape state_input_shape
            The history that will be refreshed (basically a set of n frames concatenated
            [np.array dtype=np.int8]) as a state on the replay memory.
    :param state_next : Image (np.array of dtype=np.uint8 of input_shape)
            Frame (np.array dtype=np.int8) of the environment's current state after a action was taken.
    :return nothing
    """

    history[:, :-1] = history[:, 1:]
    history[:, -1] = state_next
    return history


def reshape_state(state, history_size):
    # Clone and concatenate the state, history_size times
    return np.tile(np.expand_dims(state, axis=1), (1, history_size, 1))



def gumbel_softmax_sample(logits):
    """
    Produces Gumbel softmax samples from the input log-probabilities (logits).
    These are used, because they are differentiable approximations of the distribution of an argmax.
    """
    uniform_noise = tf.random.uniform(tf.shape(logits))
    gumbel = -tf.math.log(-tf.math.log(uniform_noise))
    noisy_logits = gumbel + logits  # / temperature
    return tf.math.softmax(noisy_logits)


def to_tensor(arg):
    arg = tf.convert_to_tensor(arg)
    return arg


def create_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def space_n_to_shape_n(space_n):
    """
    Takes a list of gym spaces and returns a list of their shapes
    """
    return np.array([space_to_shape(space) for space in space_n])


def space_to_shape(space):
    """
    Takes a gym.space and returns its shape
    """
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [space.n]
    else:
        raise RuntimeError("Unknown space type. Can't return shape.")


def softmax_to_argmax(action_n, agents):
    """
    If given a list of action probabilities performs argmax on each of them and outputs
    a one-hot-encoded representation.
    Example:
        [0.1, 0.8, 0.1, 0.0] -> [0.0, 1.0, 0.0, 0.0]
    :param action_n: list of actions per agent
    :param agents: list of agents
    :return List of one-hot-encoded argmax per agent
    """
    hard_action_n = []
    for ag_idx, (action, agent) in enumerate(zip(action_n, agents)):
        hard_action_n.append(tf.keras.utils.to_categorical(np.argmax(action), agent.act_shape_n[ag_idx,0]))

    return hard_action_n


def make_env(scenario_name, no_agents):
    from environments.multiagent.environment import MultiAgentEnv
    import environments.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(no_agents=no_agents)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

