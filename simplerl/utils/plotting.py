from functools import partial
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from simplerl.agents import BaseAgent


def plot_agent_policy(agent, env):
    greedy_actions = [agent.get_greedy_actions(state) for state in range(env.num_states)]
    l = 0.4
    arrows = [(0, -l), (0, l), (-l, 0), (l, 0)]
    # Prepare the plot showing the gridworld's structure.
    gridworld_map = env.gridworld.copy()
    gridworld_map[gridworld_map != "#"] = 1.0
    gridworld_map[gridworld_map == "#"] = 0.0
    gridworld_map = gridworld_map.astype(float)

    # Plot the gridworld's structure.
    fig, ax = plt.subplots()
    cmap = colors.ListedColormap(["black", "white"])
    norm = colors.BoundaryNorm(range(cmap.N), cmap.N)
    ax.imshow(gridworld_map, cmap=cmap, norm=norm, zorder=0)

    # Prepare the heatmap plot
    res = np.copy(gridworld_map)
    alphas = np.zeros_like(res)
    ys, xs = np.nonzero(gridworld_map != 0.0)
    for coord in zip(ys, xs):
        y, x = coord
        index = env._coords_to_index(coord)

        for a in greedy_actions[index]:
            ax.arrow(
                x,
                y,
                *arrows[a],
                length_includes_head=True,
                head_width=0.2,
                head_length=0.2,
                color="k",
                zorder=2,
            )

    cmap = mpl.colormaps["Reds"]
    cmap.set_bad("blue")
    ax.imshow(res, zorder=1, alpha=alphas, cmap=cmap)

    # Set tick labels and gridlines.
    ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2, zorder=1)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticks(np.arange(-0.5, gridworld_map.shape[1], 1))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(-0.5, gridworld_map.shape[0], 1))
    ax.set_yticklabels([])

    plt.show()


def plot_maze_env(env, labels=True, fig=None, ax=None):
    """Render function for Rooms environment."""
    gridworld_map = np.ones_like(env.gridworld, dtype=float)
    gridworld_map[env.gridworld == "#"] = 0.0
    gridworld_map[env.gridworld == "T"] = 2.0

    if fig is None:
        fig, ax = plt.subplots()

    ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2, zorder=1)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticks(np.arange(-0.5, gridworld_map.shape[1], 1))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(-0.5, gridworld_map.shape[0], 1))
    ax.set_yticklabels([])

    cmap = colors.ListedColormap(["black", "white", "grey"])
    norm = colors.BoundaryNorm(range(cmap.N + 1), cmap.N)
    im = ax.imshow(gridworld_map, cmap=cmap, norm=norm)

    ys, xs = np.nonzero(gridworld_map != 0.0)
    for coord in zip(ys, xs):
        y, x = coord
        index = env._coords_to_index(coord)
        if labels:
            text = ax.text(x, y, index, ha="center", va="center", fontsize=6)

    return fig, ax, im


def func(frame, agent, env, data, ax):
    for art in ax.get_children():
        if isinstance(art, mpl.patches.FancyArrow):
            art.remove()
    l = 0.4
    arrows = [(0, -l), (0, l), (-l, 0), (l, 0)]
    vals = np.zeros_like(env.gridworld, dtype=float)
    q_dict = agent.get_q_values(t=frame)
    for state in range(env.num_states):
        max_q_value = max(q_dict[state][action] for action in agent.actions)
        greedy_actions = [action for action, value in q_dict[state].items() if value == max_q_value]
        y, x = env._index_to_coords(state)
        vals[y, x] = max_q_value
        for a in greedy_actions:
            ax.arrow(
                x,
                y,
                *arrows[a],
                length_includes_head=True,
                head_width=0.2,
                head_length=0.2,
                color="k",
                zorder=2,
            )
    data.set_data(vals)
    return (data, ax)


def animate_learning(agent: BaseAgent, env, frames: int):
    gridworld_map = np.ones_like(env.gridworld, dtype=float)
    gridworld_map[env.gridworld == "#"] = 0.0

    fig, ax = plt.subplots()

    ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2, zorder=1)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticks(np.arange(-0.5, gridworld_map.shape[1], 1))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(-0.5, gridworld_map.shape[0], 1))
    ax.set_yticklabels([])

    cmap = colors.ListedColormap(["black", "white", "grey"])
    norm = colors.BoundaryNorm(range(cmap.N + 1), cmap.N)
    walls = ax.imshow(gridworld_map, cmap=cmap, norm=norm, zorder=0)

    alphas = np.copy(gridworld_map)
    data = ax.imshow(gridworld_map, alpha=alphas, zorder=1)

    ani = animation.FuncAnimation(fig, partial(func, agent=agent, env=env, data=data, ax=ax), frames, blit=False)
    return ani
