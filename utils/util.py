from scipy.spatial import cKDTree
import numpy as np
from spektral.layers import GCNConv, GATConv
import tensorflow as tf
import random


class Utility(object):
    def __init__(self, no_agents, is_rnn=False, is_gcn=False, is_gat=False):
        self.no_agents = no_agents
        self.is_rnn = is_rnn
        self.is_gcn = is_gcn
        self.is_gat = is_gat

    def to_tensor(self, arg):
        arg = tf.convert_to_tensor(arg)
        return arg

    def create_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def get_adj(self, arr, k_lst):
        """
        Take as input the new obs. In position 4 to k, there are the x and y coordinates of each agent
        Make an adjacency matrix, where each agent communicates with the k closest ones
        """
        points = [i[2:4] for i in arr]
        adj = np.zeros((self.no_agents, self.no_agents), dtype=float)
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
        if self.is_gcn:
            adj = GCNConv.preprocess(adj).astype('f4')
        elif self.is_gat:
            adj = GATConv.preprocess(adj).astype('f4')
        return adj


    def clip_by_local_norm(self, gradients, norm):
        """
        Clips gradients by their own norm, NOT by the global norm
        as it should be done (according to TF documentation).
        This here is the way MADDPG does it.
        """
        for idx, grad in enumerate(gradients):
            gradients[idx] = tf.clip_by_norm(grad, norm)
        return gradients


    def huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = tf.keras.backend.abs(error) < clip_delta

        squared_loss = 0.5 * tf.keras.backend.square(error)
        linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

        return tf.where(cond, squared_loss, linear_loss)


    def refresh_history(self, history, state_next):
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


    def reshape_state(self, state, history_size):
        # Clone and concatenate the state, history_size times
        return np.tile(np.expand_dims(state, axis=1), (1, history_size, 1))