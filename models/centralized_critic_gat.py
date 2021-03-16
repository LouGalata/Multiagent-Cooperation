import keras
import tensorflow as tf
import numpy as np
from spektral.layers import GATConv
from commons import util as u


class MADDPGCriticNetwork(object):
    def __init__(self, no_layers, units_per_layer, lr, obs_shape_n, act_shape_n, no_neighbors=2):
        """
        Implementation of a critic to represent the Q-Values. Basically just a fully-connected
        regression ANN.
        """
        self.lr = lr
        self.clip_norm = 0.5
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.no_layers = no_layers
        self.obs_shape_n = obs_shape_n  # nd.array(no_agents --> no_features)
        self.act_shape_n = act_shape_n  # nd.array(no_agents --> no_actions)
        self.no_agents = len(self.obs_shape_n)
        self.no_features = self.obs_shape_n[0][0]
        self.no_actions = self.act_shape_n[0][0]
        # GAT
        self.k_lst = list(range(no_neighbors + 2))[2:]

        self.graph_input = tf.keras.layers.Input((self.no_agents, self.no_features + self.no_actions), name="graph_input")
        self.adj = tf.keras.layers.Input(shape=(self.no_agents, self.no_agents), name="adj")
        # (2, (None, 15))
        self.gat = GATConv(
            units_per_layer,
            activation='elu',
            attn_heads=2,
            concat_heads=True,
        )([self.graph_input, self.adj])

        self.hidden_layers = []
        for idx in range(self.no_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu')
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(1, activation='linear')
        self.flatten = keras.layers.Flatten()(self.gat)
        x = self.flatten
        for idx in range(self.no_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)

        # connect layers
        self.model = tf.keras.Model(inputs=[self.graph_input, self.adj],  # list concatenation
                                    outputs=[x])

        # tf.keras.utils.plot_model(self.model, show_shapes=True)
        self.model.compile(self.optimizer, loss='mse')

    def predict(self, obs_n, act_n, adjacency):
        """
        Predict the value of the input.
        It should be
        graph: (batch_size, no_agents, features+no_actions) coming (no_agents, batch_size, features)
        adj: (batch_size, no_agents, no_agents)
        """
        concatenated_input = tf.concat([obs_n, act_n], axis=-1)
        concatenated_input = tf.transpose(concatenated_input, [1, 0, 2])
        return self._predict_internal(concatenated_input, adjacency)

    def _predict_internal(self, concatenated_input, adjacency):
        # x = self.gat()[concatenated_input, adjacency] # NOT WORKING
        # x = self.flatten(x)
        # for idx in range(self.no_layers):
        #     x = self.hidden_layers[idx](x)
        # x = self.output_layer(x)
        x = self.model.predict([concatenated_input, adjacency])
        return x

    def train_step(self, obs_n, act_n, adjacency, target_q):
        """
        Train the critic network with the observations, actions, rewards and next observations, and next actions.
        """
        concatenated_input = np.concatenate([obs_n, act_n], axis=-1)
        concatenated_input = np.swapaxes(concatenated_input, 1, 0)
        return self._train_step_internal(concatenated_input, adjacency, target_q)

    @tf.function
    def _train_step_internal(self, concatenated_input, adjacency, target_q):
        """
        Internal function, because concatenation can not be done inside tf.function
        """
        with tf.GradientTape() as tape:
            q_pred = self.model([concatenated_input, adjacency], training=True)
            td_loss = tf.math.square(target_q - q_pred)
            loss = tf.reduce_mean(td_loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        local_clipped = u.clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))

        return loss, td_loss

    def save(self, fp):
        tf.saved_model.save(self.model, fp + '/critic')

    def load(self, fp):
        self.model = keras.models.load_model(fp + '/critic')
