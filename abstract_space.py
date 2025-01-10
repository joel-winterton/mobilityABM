"""
ABM needs to use some discrete space,
but doesn't need to know the nitty-gritty about the space just distances and some identifier for each discrete point in the space.
This abstraction defines the only things the ABM needs to use, and then we can plug in different types of spaces (hexagonal lattices, voronoi lattices e.t.c.)
"""
from typing import List, Generic, TypeVar, Type

import numpy as np

Coordinate = TypeVar('Coordinate')


class AbstractSpace(Generic[Coordinate]):
    size: Coordinate
    space: np.ndarray

    def __init__(self, size: Coordinate, coordinate: Type[Coordinate]):
        self.size = size
        self.space = np.zeros(shape=tuple(size.array))
        self.distance_matrix = np.zeros(shape=tuple(size.array) + tuple(size.array))
        self.coordinate = coordinate

    def distance(self, c1: Coordinate, c2: Coordinate) -> float:
        """
        Gives distance between two coordinates in space.
        :param c1:
        :param c2:
        :return:
        """
        pass

    def patches_at(self, c: Coordinate, d: float) -> List[Coordinate]:
        """
        Returns a list of patches that are d distance from coordinate c.
        :param c:
        :param d:
        :return:
        """
        distance_indices = np.argwhere(self.distance_matrix[*c.array]==d)
        distance_coords = [self.coordinate(*c) for c in distance_indices]
        return distance_coords

    def calculate_distance_matrix(self):
        for c1 in np.ndindex(self.space.shape):
            for c2 in np.ndindex(self.space.shape):
                self.distance_matrix[*c1, *c2] = self.distance(self.coordinate(*c1), self.coordinate(*c2))
