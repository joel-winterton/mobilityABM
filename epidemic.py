from typing import List

from lattice import Lattice, Coordinate, Offset
import numpy as np
from person import Person


class Epidemic:
    people: List[Person]

    def __init__(self, geometry: Lattice, people=25, steps=250):
        self.geometry = geometry
        self.n = people
        self.steps = steps


        self.generate_seeds()

    def generate_seeds(self, pattern='uniform'):
        """
        Place individuals on geometry.
        For now, we just randomly seed people.
        :return:
        """
        if pattern == 'uniform':
            seeds_raw = np.array([np.random.randint(0, self.geometry.r, size=self.n),
                                  np.random.randint(0, self.geometry.c, size=self.n)]).T
            seed_coords = [Coordinate(offset=Offset(*i)) for i in seeds_raw]
        else:
            seed_coords = []

        self.people = [Person(lattice=self.geometry, seed=coord) for coord in seed_coords]
