"""
Simulation code for commuter distinguishability.
"""
import numpy as np


def make_infection_prob_fn(beta, od_matrix, population_sizes, distinguish=True):
    def infection_probabilities_distinguish(infected):
        within = infected.sum(axis=1)
        between = infected.sum(axis=0)
        return 1 - np.exp(-beta / population_sizes * (within + between - infected.diagonal()))

    def infection_probabilities_meanfield(infected):
        within = infected.sum(axis=1)
        infecteds = infected.sum(axis=0)
        pop_sizes = od_matrix.sum(axis=0)
        between = np.dot(od_matrix.T, (infecteds / pop_sizes))
        difference = od_matrix.diagonal() * infected.diagonal() / pop_sizes
        return 1 - np.exp(-beta / population_sizes * (within + between - difference))

    if distinguish:
        return infection_probabilities_distinguish
    return infection_probabilities_meanfield


def simulate(beta, gamma, od_matrix, seed_subpatch, t_max, distinguish=False):
    population_sizes = od_matrix.sum(axis=0)

    infection_probabilities = make_infection_prob_fn(beta, od_matrix, population_sizes, distinguish=distinguish)

    s = np.repeat(od_matrix[None, ...], t_max, axis=0)
    total_shape = (t_max, *od_matrix.shape)

    s[0, *seed_subpatch] -= 1

    i = np.zeros(shape=total_shape, dtype=int)
    i[0, *seed_subpatch] = 1

    r = od_matrix - s - i
    rng = np.random.default_rng()

    for t in range(1, t_max):
        probs = infection_probabilities(i[t - 1, ...])
        deltas = rng.binomial(s[t - 1, ...], probs[:, np.newaxis])
        gammas = rng.binomial(i[t - 1, ...], 1 - np.exp(-np.full(i[t - 1, ...].shape, gamma)))

        # Book keep
        s[t, ...] = s[t - 1, ...] - deltas
        i[t, ...] = i[t - 1, ...] + deltas - gammas
        r[t, ...] = r[t - 1, ...] + gammas

    return s, i, r
