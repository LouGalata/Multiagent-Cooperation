import keras
import tensorflow as tf

from commons import util as u


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
        self.model = keras.models.load_model(fp + '/critic')
