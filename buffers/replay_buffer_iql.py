import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []  # buffer
        self._maxsize = int(size)  # max buffer size
        self._next_idx = 0  # experiences

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):  # When there is the equality, the _next_idx becomes zero from the mod
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def can_provide_sample(self, batch_size, max_episode_len):
        return len(self._storage) >= batch_size * max_episode_len

    def collect(self):
        return self.sample(-1)


class EfficientReplayBuffer(object):
    def __init__(self, size, n_agents, obs_shape_n, act_shape_n):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._obs_n = []
        self._acts_n = []
        self._obs_tp1_n = []
        self._n_agents = n_agents
        for idx in range(n_agents):
            self._obs_n.append(np.empty([size, obs_shape_n[idx, 0]], dtype=np.float32))
            self._acts_n.append(np.empty([size, act_shape_n[idx][0]], dtype=np.float32))
            self._obs_tp1_n.append(np.empty([size, obs_shape_n[idx, 0]], dtype=np.float32))
        self._done = np.empty([size], dtype=np.float32)
        self._reward = np.empty([size], dtype=np.float32)
        self._maxsize = size
        self._next_idx = 0
        self.full = False
        self.len = 0

    def __len__(self):
        return self.len

    def add(self, obs_t, action, reward, obs_tp1, done):
        for ag_idx in range(self._n_agents):
            self._obs_n[ag_idx][self._next_idx] = obs_t[ag_idx]
            self._acts_n[ag_idx][self._next_idx] = action[ag_idx]
            self._obs_tp1_n[ag_idx][self._next_idx] = obs_tp1[ag_idx]
        self._reward[self._next_idx] = reward
        self._done[self._next_idx] = done

        if not self.full:
            self._next_idx = self._next_idx + 1
            if self._next_idx > self._maxsize - 1:
                self.full = True
                self.len = self._maxsize
                self._next_idx = self._next_idx % self._maxsize
            else:
                self.len = self._next_idx - 1
        else:
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        """
        Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > self.len:
            raise RuntimeError('Too few samples in buffer to generate batch.')

        indices = np.random.randint(self.len, size=[batch_size])

        obs_n = []
        acts_n = []
        next_obs_n = []
        for ag_idx in range(self._n_agents):
            obs_n.append(self._obs_n[ag_idx][indices])
            acts_n.append(self._acts_n[ag_idx][indices].copy())
            next_obs_n.append(self._obs_tp1_n[ag_idx][indices])

        rew = self._reward[indices]
        done = self._done[indices]
        return obs_n, acts_n, rew, next_obs_n, done

