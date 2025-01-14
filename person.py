import numpy as np

from lattice import Lattice, Coordinate
from truncated_levy import TruncatedLevy

"""
Person is on Hexagonal Lattice space for now, can abstract out when switching to a Voronoi lattice.
"""


class Person:
    lattice: Lattice

    def __init__(self, lattice: Lattice, seed: Coordinate, rho=0.6, gamma=0.21):
        self.lattice = lattice
        self.seed = seed

        self.rho = rho
        self.gamma = gamma
        self.length_distribution = TruncatedLevy(beta=0.9, k=25)
        self.visited_freq = dict()
        self.visited_freq[seed] = 1
        self.trajectory = [seed]

    def visit(self, coords):
        """
        Visit a location and do the bookkeeping.
        :param coords: Coordinates of the location to visit.
        :return:
        """
        if coords in self.visited_freq:
            self.visited_freq[coords] += 1
        else:
            self.visited_freq[coords] = 1
        self.trajectory.append(coords)

    def preferential_return(self):
        """
        Return to a known location according to Zipfs law, bookkeeping included.
        TODO convert to coordinate type.
        :return: Coordinate that has been returned to.
        """
        coords = np.array(list(self.visited_freq.keys()))
        vals = np.array(list(self.visited_freq.values()))
        probabilities = vals / np.sum(vals)
        next_coord = np.random.choice(coords, size=1, p=probabilities).flatten()[0]
        self.visit(next_coord)
        return next_coord

    def explore(self):
        """
        Explore a new location.
        :return:
        """
        step_size = int(self.length_distribution.rvs(size=1))
        patches = self.lattice.funnel(self.trajectory[-1], step_size)
        # This isn't great, since it means the jump is outside the geometry, try and avoid
        while len(patches) == 0:
            step_size = int(self.length_distribution.rvs(size=1))
            patches = self.lattice.funnel(self.trajectory[-1], step_size)

        chosen_coord = np.random.choice(patches)
        self.visit(chosen_coord)
        return chosen_coord

    def step(self):
        """
        Make a single iteration of model.
        :return:
        """
        uniform = np.random.default_rng().uniform()
        p_explore = self.rho * len(self.visited_freq.keys()) ** self.gamma
        if uniform < p_explore:
            self.explore()
        else:
            self.preferential_return()

    def run(self, n=50):
        for _ in range(n):
            self.step()
