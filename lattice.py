import numpy as np

"""
Coordinate system classes.
"""


class Offset:
    def __init__(self, r, c):
        r, c = int(r), int(c)
        self.r = r
        self.c = c
        self.array = np.array([r, c])
        self.tuple = (r, c)


class Cube:
    def __init__(self, r, q, s):
        r, q, s = int(r), int(q), int(s)
        self.r = r
        self.q = q
        self.s = s
        self.array = np.array([r, q, s])
        self.tuple = (r, q, s)


class Coordinate:
    cube: Cube
    offset: Offset

    def __init__(self, offset: Offset = None, cube: Cube = None):
        if offset is None and cube is None:
            raise ValueError("You need to pass either offset or cube coordinates")

        if offset is not None:
            self.offset = offset
            self.cube = Cube(*oddr_to_cube(*offset.tuple))
        if cube is not None:
            self.cube = cube
            self.offset = Offset(*cube_to_oddr(*cube.tuple))

    def cartesian(self):
        """
        Return cartesian coordinates from offsets.
        This assumes an outer circle radius of 1 to avoid passing parameters everywhere.
        :return:
        """
        dx = np.sqrt(3)
        dy = 3 / 2
        a = self.offset.r % 2
        x = (a / 2 + self.offset.c) * dx
        y = self.offset.r * dy
        return x, y


def oddr_to_cube(r, c):
    """
    Convert (r,c) offset coordinates to cube coordinates (q,r,s).
    :param r:
    :param c:
    :return:
    """
    q = c - (r - r & 1) / 2
    return q, r, -q - r


def cube_to_oddr(q, r, s):
    """
    Convert cube coordinates (q,r,s) to (r,c) offset coordinates.
    :param q:
    :param r:
    :param s:
    :return:
    """
    return q + (r - (r & 1)) / 2, r


"""
Lattice class
"""


class Lattice:
    def __init__(self, width: int):
        assert (width > 0 & width % 2 == 0)
        self.r = width
        self.c = width

        self.grid = np.zeros((self.r, self.c))

    def validate_coord(self, coords: Coordinate):
        r, c = coords.offset.array
        return 0 <= coords.offset.r < self.r - 1 and 0 <= c < self.c - 1

    def funnel(self, center: Coordinate, distance: int):
        """
        Return all coordinates on lattice that are distance from center coordinate.
        :param center:
        :param distance:

        :return:
        """

        r1, q1, s1 = center.cube.array
        result = []

        for dr in range(-distance, distance + 1):
            for dq in range(-distance, distance + 1):
                ds = -dr - dq
                if abs(dr) + abs(dq) + abs(ds) == 2 * distance:
                    coord = Coordinate(cube=Cube(r1 + dr, q1 + dq, s1 + ds))
                    if self.validate_coord(coord):
                        result.append(coord)
        return result

    def count_visits(self, members):
        """
        Count the number of visits to each location for a list of trajectories.
        Each timestep spent in a location is counted as a visit.
        :param members:
        :return:
        """
        counts = np.zeros((self.r, self.c), dtype=int)
        for member in members:
            for coord in member.trajectory:
                counts[*coord.offset.array] += 1
        return counts
