"""
We want to be able to sample people from each \pi_{ij}
"""


class Sampler:
    """
    Baseplate sampler, infection simulation should rely on methods from this class, and
    each type of sampler should implement these methods.
    """

    def __init__(self):
        """
        """

    def sample(self, i, j):
        """
        Sample a one-hot vector of $N_i$ individuals travelling from $i$ to $j$ from distribution $\pi_{ij}$.
        :param i:
        :param j:
        :return:
        """
        pass


class CommuterSampler(Sampler):
    """
    Perfect commuter sampler.
    """

    def __init__(self, od_matrix, population_sizes):
        super().__init__()
        self.od_matrix = od_matrix
        self.population_sizes = population_sizes
