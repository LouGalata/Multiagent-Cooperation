import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "results"))
agrew = pd.read_pickle(os.path.join(path, "Nav-Coop-5_agrewards.pkl"))
rew = pd.read_pickle(os.path.join(path, "Nav-Coop-5_rewards.pkl"))

agrew_lengt = len(agrew)
rew_length = len(rew)


plt.xlabel('Episodes')
plt.ylabel('Mean Rewards')
plt.plot(rew, 'ro', color='r', linewidth=1.0)
plt.xticks(np.arange(0, rew_length, 20))
plt.show()
pass
