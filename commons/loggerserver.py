import os
import time
import pickle

import numpy as np
import tensorflow as tf

class RLLogger(object):
    def __init__(self, exp_name, n_agents, n_adversaries, save_rate, arglist):
        '''
        Initializes a logger.
        This logger will take care of results, and debug info, but never the replay buffer.
        '''
        # self._run = _run
        # args = _run.config
        self.n_agents = n_agents
        self.n_adversaries = n_adversaries

        # if not os.path.exists(os.path.join(save_dir)):
        #     os.makedirs(save_dir)
        # while os.path.exists(os.path.join(save_dir, exp_name)):
        #     print('WARNING: EXPERIMENT ALREADY EXISTS. APPENDING TO  TRIAL_NAME.')
        #     exp_name = exp_name + '_i'
        print(exp_name)
        self.ex_path = os.path.join('results', str(exp_name))
        os.makedirs(self.ex_path, exist_ok=True)
        self.model_path = os.path.join(self.ex_path, 'models')
        os.makedirs(self.model_path, exist_ok=True)
        self.tb_path = os.path.join(self.ex_path, 'tb_logs')
        os.makedirs(self.tb_path, exist_ok=True)
        self.tb_writer = tf.summary.create_file_writer(self.tb_path)
        self.tb_writer.set_as_default()

        # CSV FILES
        self.path_episode_reward = os.path.join(self.ex_path, "episode_reward.csv")
        self.path_policy_losses = []
        self.path_critic_losses = []
        for i in range(self.n_agents):
            self.path_policy_losses.append(os.path.join(self.ex_path, "ag%dpolicy_loss.csv" % i))
            self.path_critic_losses.append(os.path.join(self.ex_path, "ag%dcritic_loss.csv" % i))

        # save arguments
        args_file_name = os.path.join(self.ex_path, 'args.pkl')
        with open(args_file_name, 'wb') as fp:
            pickle.dump(arglist, fp)

        self.episode_rewards = [0.0]
        self.agent_rewards = [[0.0] for _ in range(n_agents)]
        self.final_ep_rewards = []  # sum of rewards for training curve
        self.final_ep_ag_rewards = []  # agent rewards for training curve
        self.agent_info = [[[]]]  # placeholder for benchmarking info
        self.train_step = 0
        self.episode_step = 0
        self.episode_count = 0
        self.t_start = time.time()
        self.t_last_print = time.time()

        self.save_rate = save_rate

    def record_episode_end(self, agents, testing):
        """
        Records an episode having ended.
        If save rate is reached, saves the models and prints some metrics.
        """
        self.episode_count += 1
        self.episode_step = 0
        self.episode_rewards.append(0.0)
        for ag_idx in range(self.n_agents):
            self.agent_rewards[ag_idx].append(0.0)

        if self.episode_count % (self.save_rate / 10) == 0:
            mean_rew = np.mean(self.episode_rewards[-self.save_rate:])
            self.save_logger("episode_reward", mean_rew, self.train_step)
            # mean_ag_rew_ = []
            # for ag_idx in range(self.n_agents):
            #     mean_ag_rew = np.mean(self.agent_rewards[ag_idx][:-self.save_rate // 10:-1])
            #     mean_ag_rew_.append(mean_ag_rew)

        if self.episode_count % self.save_rate == 0:
            self.print_metrics()
            self.calculate_means()
            if not testing:
                self.save_models(agents, self.episode_count)

    def save_logger(self, fp, value, step, ag_idx=None, td_loss=None):
        if fp == "episode_reward":
            with open(self.path_episode_reward, "a+") as f:
                mes_dict = {"steps": step, "value": value}
                # print(mes_dict)
                for item in list(mes_dict.values()):
                    f.write("%s\t" % item)
                f.write("\n")
                f.close()
        elif fp == "policy_loss":
            path = self.path_policy_losses[ag_idx]
            with open(path, "a+") as f:
                mes_dict = {"steps": step, "value": value}
                # print(mes_dict)
                for item in list(mes_dict.values()):
                    f.write("%s\t" % item)
                f.write("\n")
                f.close()
        elif fp == "critic_loss":
            path = self.path_critic_losses[ag_idx]
            with open(path, "a+") as f:
                mes_dict = {"steps": step, "value": value}
                # print(mes_dict)
                for item in list(mes_dict.values()):
                    f.write("%s\t" % item)
                f.write("\n")
                f.close()


    def experiment_end(self):
        rew_file_name = os.path.join(self.ex_path, 'rewards.pkl')
        with open(rew_file_name, 'wb') as fp:
            pickle.dump(self.final_ep_rewards, fp)
        agrew_file_name = os.path.join(self.ex_path, 'agrewards.pkl')
        with open(agrew_file_name, 'wb') as fp:
            pickle.dump(self.final_ep_ag_rewards, fp)
        print('...Finished total of {} episodes in {} minutes.'.format(self.episode_count,
                                                                       (time.time() - self.t_start) / 60))

    def print_metrics(self):
        if self.n_adversaries == 0:
            print('steps: {}, episodes: {}, mean episode reward: {}, time: {}'.format(
                self.train_step, self.episode_count, np.mean(self.episode_rewards[-self.save_rate:-1]),
                round(time.time() - self.t_last_print, 3)))
        else:
            print('steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}'.format(
                self.train_step, self.episode_count, np.mean(self.episode_rewards[-self.save_rate:-1]),
                [np.mean(rew[-self.save_rate:-1]) for rew in self.agent_rewards], round(time.time() - self.t_last_print, 3)))
        self.t_last_print = time.time()

    def save_models(self, agents, episode):
        for idx, agent in enumerate(agents):
            episode = int(episode // 100)
            fp = os.path.join(self.model_path, 'ep{}'.format(episode))
            agent.save(os.path.join(fp, 'agent_{}'.format(idx)))

    def calculate_means(self):
        self.final_ep_rewards.append(np.mean(self.episode_rewards[-self.save_rate:-1]))
        for ag_rew in self.agent_rewards:
            self.final_ep_ag_rewards.append(np.mean(ag_rew[-self.save_rate:-1]))


    def add_agent_info(self, agent, info):
        raise NotImplementedError()

    def get_sacred_results(self):
        return np.array(self.episode_rewards), np.array(self.agent_rewards)


    @property
    def cur_episode_reward(self):
        return self.episode_rewards[-1]

    @cur_episode_reward.setter
    def cur_episode_reward(self, value):
        self.episode_rewards[-1] = value


