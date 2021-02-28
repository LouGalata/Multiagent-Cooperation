import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Reshape


class ActionValueModel:
    def __init__(self, args, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = args.no_atoms
        self.tau = [(2*(i-1)+1)/(2*self.atoms) for i in range(1, self.atoms+1)]
        self.opt = tf.keras.optimizers.Adam(args.lr)
        self.model = self.create_model()

    def create_model(self):
        return tf.keras.Sequential([
            Input([self.state_dim, ]),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_dim * self.atoms, activation='linear'),
            Reshape([self.action_dim, self.atoms])
        ])


    def huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = tf.keras.backend.abs(error) < clip_delta

        squared_loss = 0.5 * tf.keras.backend.square(error)
        linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

        return tf.where(cond, squared_loss, linear_loss)


    def quantile_huber_loss(self, target, pred, actions):
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        # This operation creates a new tensor by replicating input multiples times. The output tensor's i'th dimension has input.dims(i) * multiples[i] elements, and the values of input are replicated multiples[i] times along the 'i'th dimension. For example, tiling [a b c d] by [2] produces [a b c d a b c d].
        pred_tile = tf.tile(tf.expand_dims(pred, axis=2), [1, 1, self.atoms])
        target_tile = tf.tile(tf.expand_dims(target, axis=1), [1, self.atoms, 1])
        huber_loss = self.huber_loss(target_tile, pred_tile)
        tau = tf.reshape(np.array(self.tau), [1, self.atoms])
        inv_tau = 1.0 - tau
        tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.atoms, 1])
        inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.atoms, 1])
        error_loss = tf.math.subtract(target_tile, pred_tile)
        loss = tf.where(tf.less(error_loss, 0.0), inv_tau *
                        huber_loss, tau * huber_loss)
        loss = tf.reduce_mean(tf.reduce_sum(
            tf.reduce_mean(loss, axis=2), axis=1))
        return loss

    def train(self, states, target, actions):
        with tf.GradientTape() as tape:
            theta = self.model(states)
            # theta: shape(batch_size, no_actions, atoms)
            # actions: shape(batch_size, no_actions) one-hot
            # target: list of batch_size --> ndarray(shape = no_atoms)
            loss = self.quantile_huber_loss(target, theta, actions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state, ep):
        state = np.reshape(state, [1, self.state_dim])
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)

    def get_optimal_action(self, state):
        z = self.model.predict(state)[0]
        q = np.mean(z, axis=1)
        return np.argmax(q)
