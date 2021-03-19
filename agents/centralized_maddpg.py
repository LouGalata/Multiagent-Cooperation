import keras
import tensorflow as tf
import numpy as np
from commons import util as u


class MADDPGAgent(object):
    def __init__(self, obs_shape_n, act_shape_n, agent_index, lr, no_layers, num_units, path,
                 tau, use_gumbel):
        self.result_path = path
        self.policy = MADDPGPolicyNetwork(no_layers, num_units, lr, obs_shape_n, act_shape_n[agent_index], use_gumbel,
                                          agent_index)
        self.policy_target = MADDPGPolicyNetwork(no_layers, num_units, lr, obs_shape_n, act_shape_n[agent_index],
                                                 use_gumbel, agent_index)
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

    def update(self, obs_n, acts_n, critic, train_step):
        # Train the policy.
        policy_loss = self.policy.train(obs_n, acts_n, critic)
        # Update target networks.
        self.update_target_networks(self.tau)
        self.save(self.result_path, train_step)

        return policy_loss

    def save(self, fp, train_step):
        episode_model = int(train_step % 50000)
        fp = fp + '/' + str(episode_model)
        tf.saved_model.save(self.policy.model, fp + '/policy')

    def load(self, fp):
        self.policy.model = keras.models.load_model(fp + '/policy')


class MADDPGPolicyNetwork(object):
    def __init__(self, no_layers, units_per_layer, lr, obs_n_shape, act_shape, use_gumbel, agent_index):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        """
        self.no_layers = no_layers
        self.lr = lr
        self.obs_n_shape = obs_n_shape
        self.act_shape = act_shape
        self.use_gumbel = use_gumbel
        self.agent_index = agent_index
        self.clip_norm = 0.5

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

    @tf.function
    def get_action(self, obs):
        outputs = self.forward_pass(obs)
        if self.use_gumbel:
            outputs = u.gumbel_softmax_sample(outputs)
        return outputs

    @tf.function
    # The state and the action that executed in the environment from an agent
    def train(self, obs_n, act_n, q_network):
        with tf.GradientTape() as tape:
            # linear output layer
            x = self.forward_pass(obs_n[self.agent_index])
            act_n = tf.unstack(act_n)
            if self.use_gumbel:
                logits = x
                # log probabilities of the gumbel softmax dist are the output of the network
                act_n[self.agent_index] = u.gumbel_softmax_sample(logits)
            else:
                act_n[self.agent_index] = x

            q_value = q_network.predict(obs_n, act_n)
            policy_regularization = tf.math.reduce_mean(tf.math.square(x))
            loss = -tf.math.reduce_mean(q_value) + 1e-3 * policy_regularization  # gradient plus regularization

        gradients = tape.gradient(loss, self.model.trainable_variables)  # todo not sure if this really works
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        local_clipped = u.clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss


class MADDPGCriticNetwork(object):
    def __init__(self, no_layers, units_per_layer, lr, obs_shape_n, act_shape_n):
        """
        Implementation of a critic to represent the Q-Values. Basically just a fully-connected
        regression ANN.
        """
        self.lr = lr
        self.clip_norm = 0.5
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.no_layers = no_layers
        # set up layers
        # each agent's action and obs are treated as separate inputs
        self.obs_input_n = []
        for idx, shape in enumerate(obs_shape_n):
            self.obs_input_n.append(tf.keras.layers.Input(shape=shape, name='obs_in' + str(idx)))

        self.act_input_n = []
        for idx, shape in enumerate(act_shape_n):
            self.act_input_n.append(tf.keras.layers.Input(shape=shape, name='act_in' + str(idx)))

        self.input_concat_layer = tf.keras.layers.Concatenate()

        self.hidden_layers = []
        for idx in range(self.no_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu')
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

        x = self.input_concat_layer(self.obs_input_n + self.act_input_n)
        for idx in range(self.no_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)

        # connect layers
        self.model = tf.keras.Model(inputs=self.obs_input_n + self.act_input_n,  # list concatenation
                                    outputs=[x])

        # tf.keras.utils.plot_model(self.model, show_shapes=True)
        self.model.compile(self.optimizer, loss='mse')

    def predict(self, obs_n, act_n):
        """
        Predict the value of the input.
        """
        return self._predict_internal(obs_n + act_n)

    @tf.function
    def _predict_internal(self, concatenated_input):
        x = self.input_concat_layer(concatenated_input)
        for idx in range(self.no_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)
        return x

    def train_step(self, obs_n, act_n, target_q):
        """
        Train the critic network with the observations, actions, rewards and next observations, and next actions.
        """
        return self._train_step_internal(obs_n + act_n, target_q)

    @tf.function
    def _train_step_internal(self, concatenated_input, target_q):
        """
        Internal function, because concatenation can not be done inside tf.function
        """
        with tf.GradientTape() as tape:
            x = self.input_concat_layer(concatenated_input)
            for idx in range(self.no_layers):
                x = self.hidden_layers[idx](x)
            q_pred = self.output_layer(x)
            td_loss = tf.math.square(target_q - q_pred)
            loss = tf.reduce_mean(td_loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        local_clipped = u.clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))

        return loss, td_loss

    def save(self, fp):
        tf.saved_model.save(self.model, fp + '/critic')

    def load(self, fp):
        fp = fp + '/' + str(0)
        self.model = keras.models.load_model(fp + '/critic')
