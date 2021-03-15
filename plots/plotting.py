import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_loss(*args):
    fconcat = pd.DataFrame()
    for i in range(len(args)):
        path = os.path.join(os.path.pardir, "results", args[i], args[i] + ".csv")

        loss_name = args[i]
        col_names = ['step', 'episode', 'train_reward', 'eval_reward', loss_name, 'time']
        f = pd.read_csv(path, '\t', names=col_names, index_col=False, skiprows=lambda x: x % 2 == 0)
        fconcat = pd.concat([fconcat, f[loss_name]], axis=1)

    plt.title("Loss Function")
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    sns.lineplot(data=fconcat, legend=args)
    plt.savefig('loss.png')
    plt.close()


def plot_reward_eval(*args):
    fconcat = pd.DataFrame()
    for i in range(len(args)):
        path = os.path.join(os.path.pardir, "results", args[i], args[i] + ".csv")

        rew_name = args[i]
        col_names = ['step', 'episode', 'train_reward', rew_name, 'loss', 'time']
        f = pd.read_csv(path, '\t', names=col_names, index_col=False, skiprows=lambda x: x % 5 == 0)
        fconcat = pd.concat([fconcat, f[rew_name]], axis=1)

    plt.title("Reward")
    plt.xlabel('Episodes')
    plt.ylabel('Evaluation Reward')
    sns.lineplot(data=fconcat, legend=args)
    plt.savefig('reward.png')
    plt.close()


if __name__ == '__main__':
    exp_name1 = "iql2-m"
    exp_name2 = "iql2s-m"

    plot_loss(exp_name1, exp_name2)
    plot_reward_eval(exp_name1, exp_name2)
