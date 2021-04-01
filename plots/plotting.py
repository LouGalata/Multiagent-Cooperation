import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d


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


class Color:
    def __init__(self):
        self.colors = list()
        self.colors.append(['lightblue', 'b'])
        self.colors.append(['lightgreen', 'g'])
        self.colors.append(['navajowhite', 'darkorange'])
        self.colors.append(['lightpink', 'magenta'])
        self.colors.append(['lightgrey', 'darkgrey'])
        self.colors.append(['red', 'darkred'])
        self.colors.append(['lightyellow', 'yellow'])


    def get_color(self, index):
        return self.colors[index]

    def get_list_color(self, index):
        return [(lambda x: x[1])(y) for y in self.colors[:index]]


def plot_training_reward(*args):
    colors = Color()
    patches = []
    for i in range(len(args)):
        rew_name = args[i]
        path = os.path.join(os.pardir, "results", "maddpg", rew_name, "episode_reward.csv")
        col_names = ['steps', rew_name]
        f = pd.read_csv(path, '\t', names=col_names, index_col=False)
        c = colors.get_color(i)

        y = gaussian_filter1d(f[rew_name].head(15000), sigma=2)

        # plt.plot(pd.DataFrame(y), c[0], pd.DataFrame(y).rolling(40).mean(), c[1])
        plt.plot(pd.DataFrame(y).rolling(100).mean(), c[1])
        patches.append(mpatches.Patch(color=c[0], label=args[i]))

    plt.title("Training Reward")
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend(handles=patches)
    plt.savefig('training_reward.png')
    plt.close()


def plot_asymptotic_perf(*args):
    colors = Color()
    patches = []
    for i in range(len(args)):
        rew_name = args[i]
        path = os.path.join(os.pardir, "evaluation", "maddpg", rew_name, "episode_reward.csv")
        col_names = ['step', 'episode', rew_name]
        f = pd.read_csv(path, '\t', names=col_names, index_col=False)
        c = colors.get_color(i)

        y = gaussian_filter1d(f[rew_name], sigma=4)

        # plt.plot(pd.DataFrame(y), c[0], pd.DataFrame(y).rolling(50).mean(), c[1])
        plt.plot(pd.DataFrame(y), c[0])
        patches.append(mpatches.Patch(color=c[0], label=args[i]))

    plt.title("Asymptotic Performance")
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend(handles=patches)
    plt.savefig('asymptotic.png')
    plt.close()


if __name__ == '__main__':
    exp_name1 = "c2a1d"
    exp_name2 = "128dd"
    exp_name3 = "c2a6d"
    exp_name4 = "c5a2d"
    exp_name5 = "c5a6d"
    exp_name6 = "c2a6bd"
    exp_name7 = "c2a1bd"

    # plot_loss(exp_name1, exp_name2)
    # plot_asymptotic_perf(exp_name1, exp_name2)
    plot_training_reward(exp_name1, exp_name2, exp_name3, exp_name4, exp_name5, exp_name6, exp_name7)