"""
Simulation code for commuter distinguishability.
"""
import numpy as np


def make_foi_fn(beta, od_matrix, collapse=False, distinguish=True):
    population_sizes = od_matrix.sum(axis=0)

    def distinguish_foi(infected):
        within = infected.sum(axis=1)
        between = infected.sum(axis=0)
        return beta / population_sizes * (within + between - infected.diagonal())

    def meanfield_foi(infected):
        within = infected.sum(axis=1)
        infecteds = infected.sum(axis=0)
        between = np.dot(od_matrix.T, (infecteds / population_sizes))
        difference = od_matrix.diagonal() * infected.diagonal() / population_sizes
        return beta / population_sizes * (within + between - difference)

    def collapsed_meanfield_foi(infected):
        if infected.ndim == 1:
            between = np.dot(od_matrix.T, (infected / population_sizes))
            difference = od_matrix.diagonal() * infected / population_sizes
            return beta / population_sizes * (infected + between - difference)
        else:
            t_max = infected.shape[0]
            pop_sizes = np.repeat(population_sizes[None,...], t_max, axis=0)
            commuter_matrix = np.repeat(od_matrix.T[None,...], t_max, axis=0)
            between = np.einsum('ijk,ik->ij', commuter_matrix, infected/pop_sizes)
            difference = commuter_matrix.diagonal(axis1=1, axis2=2) * infected / pop_sizes
            return beta / pop_sizes * (infected + between - difference)

    if collapse:
        return collapsed_meanfield_foi
    if distinguish:
        return distinguish_foi
    else:
        return meanfield_foi


def generate_state_holder(od_matrix, seed_subpatch, t_max, infected=1, collapsed=False):
    if collapsed:
        s = np.repeat(od_matrix.sum(axis=1)[None, ...], t_max, axis=0)
    else:
        s = np.repeat(od_matrix[None, ...], t_max, axis=0)

    total_shape = s.shape
    i = np.zeros(shape=total_shape, dtype=int)

    s[0, *seed_subpatch] -= infected

    i[0, *seed_subpatch] = infected

    r = np.zeros(shape=total_shape, dtype=int)
    return s, i, r


def simulate(beta, gamma, od_matrix, seed_subpatch, t_max, distinguish=False):
    foi = make_foi_fn(beta, od_matrix, distinguish=distinguish)

    s, i, r = generate_state_holder(od_matrix, seed_subpatch, t_max)

    rng = np.random.default_rng()

    for t in range(1, t_max):
        probs = 1 - np.exp(-foi(i[t - 1, ...]))
        deltas = rng.binomial(s[t - 1, ...], probs[:, np.newaxis])
        gammas = rng.binomial(i[t - 1, ...], 1 - np.exp(-np.full(i[t - 1, ...].shape, gamma)))

        # Book keep
        s[t, ...] = s[t - 1, ...] - deltas
        i[t, ...] = i[t - 1, ...] + deltas - gammas
        r[t, ...] = r[t - 1, ...] + gammas

    return s, i, r
