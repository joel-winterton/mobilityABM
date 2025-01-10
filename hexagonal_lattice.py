from typing import List

import numpy as np

from abstract_space import AbstractSpace, Coordinate


class HexagonalCoord:
    a: int
    r: int
    c: int

    def __init__(self, a, r, c):
        self.a = a
        self.r = r
        self.c = c
        # things to make translation neater
        self.array = np.array([a, r, c], dtype=int)
        self.cartesian = [a / 2 + c, np.sqrt(3) * (a / 2 + r)]


def hex_subtract(a: HexagonalCoord, b: HexagonalCoord) -> HexagonalCoord:
    return HexagonalCoord(a.a - b.a, a.r - b.r, a.c - b.c)


def hex_length(a: HexagonalCoord) -> int:
    return int((abs(a.a) + abs(a.r) + abs(a.c)) / 2)


class HexagonalLattice(AbstractSpace[HexagonalCoord]):

    def __init__(self, size: HexagonalCoord):
        if size.a != 2:
            raise ValueError("Using ARC coordinates, so a must be 2")
        super().__init__(size, HexagonalCoord)
        self.calculate_distance_matrix()

    def distance(self, c1: HexagonalCoord, c2: HexagonalCoord) -> float:
        return hex_length(hex_subtract(c1, c2))
