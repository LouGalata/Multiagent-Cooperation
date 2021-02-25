import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow_probability as tfp


class critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        # self.d2 = tf.keras.layers.Dense(32,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        # x = self.d2(x)
        v = self.v(x)
        return v


class actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        # self.d2 = tf.keras.layers.Dense(32,activation='relu')
        self.a = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_data):
        x = self.d1(input_data)
        # x = self.d2(x)
        a = self.a(x)
        return a


class agent():
    def __init__(self, gamma=0.95, lr=1e-2):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.actor = actor()
        self.critic = critic()

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, probs, actions, td):
        probability = []
        log_probability = []
        for pb, a in zip(probs, actions):
            dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)

        # print(probability)
        # print(log_probability)

        p_loss = []
        e_loss = []
        td = td.numpy()
        # print(td)
        for pb, t, lpb in zip(probability, td, log_probability):
            t = tf.constant(t)
            policy_loss = tf.math.multiply(lpb, t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        # print(p_loss)
        # print(e_loss)
        loss = -p_loss - 0.0001 * e_loss
        # print(loss)
        return loss

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            # print(discnt_rewards)
            # print(v)
            # print(td.numpy())
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss
