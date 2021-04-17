import numpy as np
import matplotlib.pyplot as plt


def func(i):
    return min_expl + (max_expl - min_expl) * np.exp(decay * i/25)





# final step
if __name__ == "__main__":
    min_expl = 0.01
    max_expl = 1.0
    decay = -0.003
    episode_size = 25
    x = list(range(1, 30000, 1))
    dt = [func(i) for i in x]
    plt.plot(x, dt)
    plt.title('Exponential E-greedy Exploration',  size = 13)
    plt.xlabel('Environmental Steps',  size = 13)
    plt.ylabel('E-greedy',  size = 13)
    plt.savefig('irnn2-expl.png')
    plt.close()
    pass