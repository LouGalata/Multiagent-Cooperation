import keras
import numpy as np
import tensorflow as tf
from gym.spaces import Space, Discrete
from spektral.layers import GATConv
from commons.OUNoise import OUNoise
from commons import util as u
from commons.weight_decay_optimizers import AdamW


class MADDPGAgent(object):
    def __init__(self, obs_space_n, act_space_n, agent_index, lr, no_layers, num_units,
                 tau, noise=0.0, use_ounoise=False, logger=None):
        self.logger = logger
        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = u.space_n_to_shape_n(obs_space_n)
        act_shape_n = u.space_n_to_shape_n(act_space_n)
        act_type = type(act_space_n[0])

        self.policy = MADDPGPolicyNetwork(no_layers, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type,
                                          agent_index, noise, use_ounoise)

        self.policy_target = MADDPGPolicyNetwork(no_layers, num_units, lr, obs_shape_n, act_shape_n[agent_index],
                                                 act_type,
                                                 agent_index, noise, use_ounoise)
        self.policy_target.model.set_weights(self.policy.model.get_weights())

        self.agent_index = agent_index
        self.tau = tau

    def action(self, obs):
        """
        Get an action from the non-target policy
        """
        return self.policy.get_action(obs[None])[0]

    def target_action(self, obs):
        """
        Get an action from the target policy
        """
        return self.policy_target.get_action(obs)

    def update_target_networks(self, tau):
        """
        Implements the updates of the target networks, which slowly follow the real network.
        """

        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            net_weights = np.array(net.get_weights())
            target_net_weights = np.array(target_net.get_weights())
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        update_target_network(self.policy.model, self.policy_target.model)

    def update(self, obs_n, acts_n, adjacency, critic, train_step):
        # Train the policy.
        policy_loss = self.policy.train(obs_n, acts_n, adjacency, critic)
        # Update target networks.
        self.update_target_networks(self.tau)
        self.logger.save_logger("policy_loss", policy_loss.numpy(), train_step, self.agent_index)
        return policy_loss

    def save(self, fp):
        self.policy.model.save_weights(fp + 'policy.h5')
        self.policy_target.model.save_weights(fp + 'policy_target.h5')

    def load(self, fp):
        self.policy.model.load_weights(fp + 'policy.h5')
        self.policy_target.model.load_weights(fp + 'policy_target.h5')


class MADDPGPolicyNetwork(object):
    def __init__(self, no_layers, units_per_layer, lr, obs_n_shape, act_shape, act_type, agent_index, noise, use_ounoise):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        """
        self.no_layers = no_layers
        self.lr = lr
        self.obs_n_shape = obs_n_shape
        self.act_shape = act_shape
        self.agent_index = agent_index
        self.clip_norm = 0.5
        if act_type is Discrete:
            self.use_gumbel = True
        else:
            self.use_gumbel = False

        self.noise = noise
        self.noise_mode = OUNoise(act_shape[0], scale=1.0)
        self.use_ounoise = use_ounoise

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        ### set up network structure
        self.obs_input = tf.keras.layers.Input(shape=self.obs_n_shape[agent_index])

        self.hidden_layers = []
        for idx in range(no_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}pol_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        if self.use_gumbel:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='linear',
                                                      name='ag{}pol_out{}'.format(agent_index, no_layers))
        else:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='tanh',
                                                      name='ag{}pol_out{}'.format(agent_index, no_layers))

        # connect layers
        x = self.obs_input
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        self.model = tf.keras.Model(inputs=[self.obs_input], outputs=[x])

    def forward_pass(self, obs):
        x = obs
        for idx in range(self.no_layers):
            x = self.hidden_layers[idx](x)
        outputs = self.output_layer(x)  # log probabilities of the gumbel softmax dist are the output of the network
        return outputs

    @classmethod
    def gumbel_softmax_sample(cls, logits):
        """
        Produces Gumbel softmax samples from the input log-probabilities (logits).
        These are used, because they are differentiable approximations of the distribution of an argmax.
        """
        uniform_noise = tf.random.uniform(tf.shape(logits))
        gumbel = -tf.math.log(-tf.math.log(uniform_noise))
        noisy_logits = gumbel + logits  # / temperature
        return tf.math.softmax(noisy_logits)

    @tf.function
    def get_action(self, obs):
        outputs = self.forward_pass(obs)
        if self.use_gumbel:
            outputs = self.gumbel_softmax_sample(outputs)
        elif self.use_ounoise:
            outputs = outputs + self.noise * self.noise_mode.noise()
            outputs = tf.clip_by_value(outputs, -1, 1)
        return outputs

    # The state and the action that executed in the environment from an agent
    def train(self, obs_n, act_n, adjacency, q_network):
        with tf.GradientTape() as tape:
            # linear output layer
            x = self.forward_pass(obs_n[self.agent_index])
            act_n = tf.unstack(act_n)
            if self.use_gumbel:
                logits = x
                # log probabilities of the gumbel softmax dist are the output of the network
                act_n[self.agent_index] = self.gumbel_softmax_sample(logits)
            elif self.use_ounoise:
                act_n[self.agent_index] += self.noise * self.noise_mode.noise()  # For continuous actions
                act_n[self.agent_index] = tf.clip_by_value(act_n[self.agent_index], -1, 1)
            else:
                act_n[self.agent_index] = x

            concatenated_input = tf.concat([obs_n, act_n], axis=-1)
            concatenated_input = tf.transpose(concatenated_input, [1, 0, 2])
            q_value = q_network.model([concatenated_input, adjacency])
            policy_regularization = tf.math.reduce_mean(tf.math.square(x))
            loss = -tf.math.reduce_mean(q_value) + 1e-3 * policy_regularization  # gradient plus regularization

        gradients = tape.gradient(loss, self.model.trainable_variables)  # todo not sure if this really works
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        local_clipped = u.clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss


class MADDPGCriticNetwork(object):
    def __init__(self, no_layers, units_per_layer, lr, obs_shape_n, act_shape_n, no_neighbors=2, wd=0.0):
        """
        Implementation of a critic to represent the Q-Values. Basically just a fully-connected
        regression ANN.
        """
        self.lr = lr
        self.clip_norm = 0.5
        self.optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        self.no_layers = no_layers
        self.obs_shape_n = obs_shape_n  # nd.array(no_agents --> no_features)
        self.act_shape_n = act_shape_n  # nd.array(no_agents --> no_actions)
        self.no_agents = len(self.obs_shape_n)
        self.no_features = self.obs_shape_n[0][0]
        self.no_actions = self.act_shape_n[0][0]
        # GAT
        self.k_lst = list(range(no_neighbors + 2))[2:]

        self.graph_input = tf.keras.layers.Input((self.no_agents, self.no_features + self.no_actions),
                                                 name="graph_input")
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
        self.model.save_weights(fp)

    def load(self, fp):
        self.model.load_weights(fp)
