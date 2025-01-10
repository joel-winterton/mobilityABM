import numpy as np

from abstract_space import AbstractSpace
from truncated_levy import TruncatedLevy

"""
Person is on Hexagonal Lattice space for now, can abstract out when switching to a Voronoi lattice.
"""


class Person:
    geometry: AbstractSpace

    def __init__(self, geometry, seed, rho=0.6, gamma=0.21):
        self.geometry = geometry
        self.seed = seed

        self.rho = rho
        self.gamma = gamma
        self.length_distribution = TruncatedLevy()
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
        patches = self.geometry.patches_at(self.trajectory[-1], step_size)
        # This isn't great, since it means the jump is outside the geometry, try and avoid
        while len(patches) == 0:
            step_size = int(self.length_distribution.rvs(size=1))
            patches = self.geometry.patches_at(self.trajectory[-1], step_size)

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


class OldPerson:
    def __init__(self, seed_coords, x_length=200, y_length=200, rho=0.6, gamma=0.21):
        # Define boundaries of box
        self.box_boundaries = [-x_length // 2, x_length // 2, -y_length // 2, y_length // 2]

        self.rho = rho
        self.gamma = gamma
        self.length_distribution = TruncatedLevy()
        self.visited_freq = dict()
        self.visited_freq[seed_coords] = 1
        self.trajectory = [seed_coords]

    def rms(self):
        """
        TODO: Add characteristic lengths
        :return:
        """

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
        :return: Coordinate that has been returned to.
        """
        coords = np.array(list(self.visited_freq.keys()), dtype='i,i')
        vals = np.array(list(self.visited_freq.values()))
        probabilities = vals / np.sum(vals)
        next_coord = tuple(np.random.choice(coords, size=1, p=probabilities)[0])
        next_coord = (int(next_coord[0]), int(next_coord[1]))
        self.visit(next_coord)
        return next_coord

    def explore(self):
        """
        Explore a new location.
        :return:
        """
        step_size = self.length_distribution.rvs(size=1)
        step_angle = np.random.default_rng().uniform() * 2 * np.pi

        new_coord = step_size * (np.cos(step_angle), np.sin(step_angle))
        grid_new_coord = (int(round(new_coord[0])), int(round(new_coord[1])))
        self.visit(grid_new_coord)
        return grid_new_coord

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
