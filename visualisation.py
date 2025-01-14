from typing import List

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.patches import RegularPolygon
import numpy as np
from person import Person
"""
Design variables
"""
COLORMAP = 'RdYlGn'
FIGURE_SIZE = (15, 15)
DPI = 300


def plot_lattice(lattice, size=1, colormap=COLORMAP):
    """
    Plots heatmap from lattice values stored in offset coordinates.
    For reasons unknown, I can't get matplotlib to just do this in hexbins.
    :param size: Outer circular radius of each hexagon.
    :param colormap: Colormap to use on values.
    :param lattice:
    :return:
    """
    r_max, c_max = lattice.shape

    fig, ax = plt.subplots(1, figsize=FIGURE_SIZE, dpi=DPI)

    # Colours
    cmap = plt.get_cmap(colormap)
    norm = colors.Normalize(vmin=0, vmax=lattice.max())
    color_list = cmap(norm(lattice))

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, orientation='vertical', label='Number of visits', fraction=0.04, pad=0.04)

    # Hexagon geometry
    dx = np.sqrt(3) * size
    dy = 3 / 2 * size
    ax.set_aspect('equal')

    for r in range(r_max):
        for c in range(c_max):
            a = r % 2
            x = (a / 2 + c) * dx
            y = r * dy
            hexagon = RegularPolygon((x, y), numVertices=6, radius=size,
                                     alpha=0.2, edgecolor='k', facecolor=color_list[r, c, :-1])
            ax.add_patch(hexagon)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.autoscale(enable=True)
    plt.show()


def plot_trajectories(members: List[Person], colormap=COLORMAP):
    max_marker_size = 50
    cmap = plt.get_cmap(colormap, len(members))
    for m, member in enumerate(members):
        trajectory = member.trajectory
        site_freq = {trajectory[0].cartesian(): 1}
        for i in range(0, len(trajectory) - 1):
            origin = trajectory[i].cartesian()
            destination = trajectory[i + 1].cartesian()
            if destination in site_freq:
                site_freq[destination] += 1
            else:
                site_freq[destination] = 1

            plt.plot([origin[0], destination[0]], [origin[1], destination[1]], '-', alpha=0.3, color=cmap(m))

        x, y = zip(*list(site_freq.keys()))
        values = np.array(list(site_freq.values()))
        sizes = max_marker_size * values / values.max()
        plt.scatter(x, y, color=cmap(m), s=sizes, marker='o')
    plt.show()
