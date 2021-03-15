from environments import gridworld as gw
import os

if __name__ == '__main__':
    n_agent = 20
    env = gw.GridWorld("battle", map_size=30)
    render_path = "test/build/render"
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    env.set_render_dir(render_path)

    handles = env.get_handles()
    n = len(handles)
    obs = [[] for _ in range(n)]
    ids = [[] for _ in range(n)]
    action = [[] for _ in range(n)]
    n_actions = env.get_action_space(handles[0])[0]
    print(env.get_action_space(handles[0])[0])
    print(env.get_action_space(handles[1])[0])
    env.add_agents(handles[0], method="random", n=20)
    env.add_agents(handles[1], method="random", n=12)
    action = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    obs[0] = env.get_observation(handles[0])
    pass
