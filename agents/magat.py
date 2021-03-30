import numpy as np
import tensorflow as tf
from gym import Space
from gym.spaces import Discrete
from scipy.spatial import cKDTree
from spektral.layers import GATConv

from agents.AbstractAgent import AbstractAgent
from commons.OUNoise import OUNoise
from commons.util import space_n_to_shape_n, clip_by_local_norm
from commons.weight_decay_optimizers import AdamW


class MAGATAgent(AbstractAgent):
    def __init__(self, no_neighbors, obs_space_n, act_space_n, agent_index, batch_size, buff_size, lr, num_layer,
                 num_critic_neurons, num_actor_neurons, num_gnn_neurons, gamma,
                 tau, prioritized_replay=False, alpha=0.6, max_step=None, initial_beta=0.6, prioritized_replay_eps=1e-6,
                 wd=1e-5, logger=None, noise=0.0, use_ounoise=False):
        """
        An object containing critic, actor and training functions for Multi-Agent DDPG.
        """
        self.logger = logger

        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = space_n_to_shape_n(obs_space_n)
        act_shape_n = space_n_to_shape_n(act_space_n)

        self.no_neighbors = no_neighbors
        self.no_agents = len(obs_shape_n)
        self.no_features = obs_shape_n[0][0]
        self.no_actions = obs_shape_n[0][0]
        self.k_lst = list(range(self.no_neighbors + 2))[2:]
        super().__init__(buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay, alpha, max_step,
                         initial_beta,
                         prioritized_replay_eps=prioritized_replay_eps)

        act_type = type(act_space_n[0])
        self.critic = MADDPGCriticNetwork(no_neighbors, num_layer, num_critic_neurons, num_gnn_neurons, lr, obs_shape_n, act_shape_n, act_type,
                                          wd, agent_index)
        self.critic_target = MADDPGCriticNetwork(no_neighbors, num_layer, num_critic_neurons, num_gnn_neurons, lr, obs_shape_n, act_shape_n,
                                                 act_type, wd, agent_index)
        self.critic_target.model.set_weights(self.critic.model.get_weights())

        self.policy = MADDPGPolicyNetwork(num_layer, num_actor_neurons, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                          self.critic, agent_index, noise, use_ounoise)
        self.policy_target = MADDPGPolicyNetwork(num_layer, num_actor_neurons, lr, obs_shape_n, act_shape_n[agent_index],
                                                 act_type, 1,
                                                 self.critic, agent_index, noise, use_ounoise)
        self.policy_target.model.set_weights(self.policy.model.get_weights())

        self.batch_size = batch_size
        self.agent_index = agent_index
        self.decay = gamma
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

    def preupdate(self):
        pass

    def update_noise(self, reduction_noise):
        self.policy.noise *= reduction_noise

    def update_target_networks(self, tau):
        """
        Implements the updates of the target networks, which slowly follow the real network.
        """

        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            net_weights = np.array(net.get_weights())
            target_net_weights = np.array(target_net.get_weights())
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        update_target_network(self.critic.model, self.critic_target.model)
        update_target_network(self.policy.model, self.policy_target.model)

    @staticmethod
    def get_adj(arr, k_lst, no_agents):
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
        adj = GATConv.preprocess(adj).astype('f4')
        return adj

    def update(self, agents, step):
        """
        Update the agent, by first updating the critic and then the actor.
        Requires the list of the other agents as input, to determine the target actions.
        """
        assert agents[self.agent_index] is self
        adjacency = None
        if self.prioritized_replay:
            obs_n, acts_n, rew_n, next_obs_n, done_n, weights, indices = \
                self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
        else:
            obs_n, acts_n, rew_n, next_obs_n, done_n = self.replay_buffer.sample(self.batch_size)
            adjacency = [self.get_adj(obs, self.k_lst, self.no_agents) for obs in
                         np.swapaxes(obs_n, 1, 0)]
            adjacency = np.array(adjacency)  # shape: (batch_size, no_agents, no_agents)

        self.update_noise(0.999)
        # Train the critic, using the target actions in the target critic network, to determine the
        # training target (i.e. target in MSE loss) for the critic update.
        target_act_next = [a.target_action(obs) for a, obs in zip(agents, next_obs_n)]
        target_q_next = self.critic_target.predict(next_obs_n, target_act_next, adjacency)
        q_train_target = rew_n[:, None] + self.decay * target_q_next

        td_loss = self.critic.train_step(obs_n, acts_n, adjacency, q_train_target).numpy()[:, 0]

        # Train the policy.
        policy_loss = self.policy.train(obs_n, acts_n, adjacency)

        # Update priorities if using prioritized replay
        if self.prioritized_replay:
            self.replay_buffer.update_priorities(indices, td_loss + self.prioritized_replay_eps)

        # Update target networks.
        self.update_target_networks(self.tau)

        self.logger.save_logger("policy_loss", policy_loss.numpy(), step, self.agent_index)
        self.logger.save_logger("critic_loss", np.mean(td_loss), step, self.agent_index)
        # self.logger.log_scalar('agent_{}.train.policy_loss'.format(self.agent_index), policy_loss.numpy(), step)
        # self.logger.log_scalar('agent_{}.train.q_loss0'.format(self.agent_index), np.mean(td_loss), step)
        return [td_loss, policy_loss]

    def save(self, fp):
        self.critic.model.save_weights(fp + 'critic.h5', )
        self.critic_target.model.save_weights(fp + 'critic_target.h5')
        self.policy.model.save_weights(fp + 'policy.h5')
        self.policy_target.model.save_weights(fp + 'policy_target.h5')
        # tf.saved_model.save(self.critic.model, fp + 'critic')
        # tf.saved_model.save(self.critic_target.model, fp + 'critic_target')
        # tf.saved_model.save(self.policy.model, fp + 'policy')
        # tf.saved_model.save(self.policy_target.model, fp + 'policy_target')

    def load(self, fp):
        # self.critic.model = tf.keras.models.load_model(fp + 'critic')
        # self.critic_target.model = tf.keras.models.load_model(fp + 'critic_target')
        # self.policy.model = tf.keras.models.load_model(fp + 'policy')
        # self.policy_target.model = tf.keras.models.load_model(fp + 'policy_target')

        self.critic.model.load_weights(fp + 'critic.h5')
        self.critic_target.model.load_weights(fp + 'critic_target.h5')
        self.policy.model.load_weights(fp + 'policy.h5')
        self.policy_target.model.load_weights(fp + 'policy_target.h5')


class MADDPGPolicyNetwork(object):
    def __init__(self, num_layers, units_per_layer, lr, obs_n_shape, act_shape, act_type,
                 gumbel_temperature, q_network, agent_index, noise, use_ounoise):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        """
        self.num_layers = num_layers
        self.lr = lr
        self.obs_n_shape = obs_n_shape
        self.act_shape = act_shape
        self.act_type = act_type
        if act_type is Discrete:
            self.use_gumbel = True
        else:
            self.use_gumbel = False
        self.use_ounoise = use_ounoise
        self.gumbel_temperature = gumbel_temperature
        self.q_network = q_network
        self.agent_index = agent_index
        self.clip_norm = 0.5
        self.noise = noise
        self.noise_mode = OUNoise(act_shape[0], scale=1.0)
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        ### set up network structure
        self.obs_input = tf.keras.layers.Input(shape=self.obs_n_shape[agent_index])

        self.hidden_layers = []
        for idx in range(num_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}pol_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        if self.use_gumbel:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='linear',
                                                      name='ag{}pol_out{}'.format(agent_index, idx))
        else:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='tanh',
                                                      name='ag{}pol_out{}'.format(agent_index, idx))

        # connect layers
        x = self.obs_input
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=[self.obs_input], outputs=[x])

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

    def forward_pass(self, obs):
        """
        Performs a simple forward pass through the NN.
        """
        x = obs
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        outputs = self.output_layer(x)  # log probabilities of the gumbel softmax dist are the output of the network
        return outputs

    @tf.function
    def get_action(self, obs):
        outputs = self.forward_pass(obs)
        if self.use_gumbel:
            outputs = self.gumbel_softmax_sample(outputs)
        elif self.use_ounoise:
            outputs = outputs + self.noise * self.noise_mode.noise()
            outputs = tf.clip_by_value(outputs, -1, 1)
        return outputs

    # @tf.function
    # The state and the action that executed in the environment from an agent
    def train(self, obs_n, act_n, adjacency):
        with tf.GradientTape() as tape:
            # linear output layer
            x = self.forward_pass(obs_n[self.agent_index])
            act_n = tf.unstack(act_n)
            if self.use_gumbel:
                logits = x
                # log probabilities of the gumbel softmax dist are the output of the network
                act_n[self.agent_index] = self.gumbel_softmax_sample(logits)
            elif self.use_ounoise:
                act_n[self.agent_index] = x + self.noise * self.noise_mode.noise()
                act_n[self.agent_index] = tf.clip_by_value(act_n[self.agent_index], -1, 1)
            else:
                act_n[self.agent_index] = x
            # q_value = self.q_network._predict_internal(obs_n + act_n)
            concatenated_input = tf.concat([obs_n, act_n], axis=-1)
            concatenated_input = tf.transpose(concatenated_input, [1, 0, 2])
            q_value = self.q_network.model([concatenated_input, adjacency])

            # policy_regularization = tf.math.reduce_mean(tf.math.square(x))
            policy_regularization = tf.math.reduce_mean(x)
            loss = -tf.math.reduce_mean(q_value) + 1e-3 * policy_regularization  # gradient plus regularization

        gradients = tape.gradient(loss, self.model.trainable_variables)  # todo not sure if this really works
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss


class MADDPGCriticNetwork(object):
    def __init__(self, no_neighbors, num_hidden_layers, units_per_layer, num_gnn_neurons, lr, obs_n_shape, act_shape_n, act_type,
                 wd, agent_index):
        """
        Implementation of a critic to represent the Q-Values. Basically just a fully-connected
        regression ANN.
        """
        self.num_layers = num_hidden_layers
        self.lr = lr
        self.obs_shape_n = obs_n_shape
        self.act_shape_n = act_shape_n
        self.act_type = act_type

        self.clip_norm = 0.5
        # self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        self.no_neighbors = no_neighbors
        self.no_agents = len(self.obs_shape_n)
        self.no_features = self.obs_shape_n[0][0]
        self.no_actions = self.act_shape_n[0][0]
        # GAT
        self.k_lst = list(range(self.no_neighbors + 2))[2:]

        self.graph_input = tf.keras.layers.Input((self.no_agents, self.no_features + self.no_actions),
                                                 name="graph_input")
        self.adj = tf.keras.layers.Input(shape=(self.no_agents, self.no_agents), name="adj")
        # (2, (None, 15))
        self.gat = GATConv(
            num_gnn_neurons,
            activation='relu',
            attn_heads=4,
            concat_heads=True,
        )([self.graph_input, self.adj])

        self.hidden_layers = []
        for idx in range(2):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu')
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

        # Try ResNet Alternative
        # self.flatten = tf.keras.layers.Flatten()(self.gat)
        self.concat = tf.keras.layers.Concatenate(axis=2)([self.graph_input, self.gat])
        self.flatten = tf.keras.layers.Flatten()(self.concat)

        x = self.flatten
        for idx in range(2):
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
        Shapes:
        obs_n: (list no_agents, ndarray(batch_size, no_features))
        act_n: (list no_agents, EagerTensor: batch_size, no_actions)
        """
        concatenated_input = tf.concat([obs_n, act_n], axis=-1)
        concatenated_input = tf.transpose(concatenated_input, [1, 0, 2])
        return self._predict_internal(concatenated_input, adjacency)

    def _predict_internal(self, concatenated_input, adjacency):
        """
        Internal function, because concatenation can not be done in tf.function
        """
        x = self.model.predict([concatenated_input, adjacency])
        return x

    def train_step(self, obs_n, act_n, adjacency, target_q):
        """
        Train the critic network with the observations, actions, rewards and next observations, and next actions.
        """
        # return self._train_step_internal(obs_n + act_n, target_q, weights)
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
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return td_loss
