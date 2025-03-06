"""
A lot of this work is comparing models against what this base mode.
"""

def make_foi_function(od_matrix):
    pop_sizes = od_matrix.sum(axis=1)

    def foi(beta, gamma, psi):
        pass

