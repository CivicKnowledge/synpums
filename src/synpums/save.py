from synpums.util import *


"""
# CP with no rows with no nonzero weight in the allocations
# Removing zero weight samples only works when the allocated weights
# is initialized before starting the vector walk.

self = deepcopy(tasks[150])
self.init()
self.initialize_weights_set_sample()

aw = pd.Series(self.allocated_weights, name='weight')
nzcp = self.make_cp(self.sample_pop.iloc[aw[aw > 0].index])
cp = nzcp.reset_index().values.astype(int)

from synpums.allocate import vector_walk_opt

sample_pop = self.sample_pop.values
allocated_weights = self.allocated_weights
target_marginals = self.target_marginals.est.values

vector_walk_opt(sample_pop, allocated_weights, cp, target_marginals, self.total_error, N=2000, min_error=100, inner_iter_min=1, inner_iter_max=15,init_inc_rate=.15 )

self.allocated_weights = allocated_weights

self.total_error, self.m90_total_error
"""


def vector_walk_opt(sample_pop, allocated_weights, cp, target_marginals, total_error,
                    N: int = 500, min_error: int = 100, inner_iter_min: int = 5, inner_iter_max: int = 60,
                    init_inc_rate: float = .15, ):
    """Consider the target state and each household to be a vector. For each iteration
    select a household vector with the best cosine similarity to the vector to the
    target and add that household to the population. """

    rng = np.random.default_rng()

    running_allocated_marginals = np.sum(sample_pop * allocated_weights.reshape(-1, 1), axis=0)

    for i in range(N):

        td = target_marginals - running_allocated_marginals

        te = np.sqrt(np.sum(np.square(td)))
        n_iter = int((inner_iter_max - inner_iter_min) * ((N - i) / N) + inner_iter_min)
        r = i / N * init_inc_rate

        try:

            new_marginals = cp.copy()

            # Compute change in each column of the error vector for adding or subtracting in
            # each of the sample population records
            # idx 0 is the index of the row in self.sample_pop
            # idx 1 is the sign, 1 or -1
            # idx 2 and up are the census count columns

            new_marginals[:, 2:] = np.add(new_marginals[:, 2:], td)

            # For each new error vector, compute total error ( via vector length). By
            # removing the current total error, we get the change in total error for
            # adding or removing each row. ( positive values are better )
            total_errors = (np.sqrt(np.square(new_marginals[:, 2:]).sum(axis=1)) - te)

            # These are indexes on cp, so have to re-reference cp to
            # get the index of the sample_pop
            idx = np.concatenate([rng.choice(np.argwhere(total_errors > 0), int(n_iter))[:, 0],  # reducing error
                                  # Random selection of records that increase the error
                                  rng.choice(np.argwhere(total_errors < 0), int(n_iter * r))[:, 0]
                                  ])

            # Columns are : 'index', 'sign', 'delta_err'
            ss = np.hstack(
                [cp[idx][:, 0:2], total_errors[idx].reshape(-1, 1)])  # Return the index and sign columns of cp

        except ValueError:
            # Usually b/c numpy chouce() got an empty array
            continue

        # This is the equivalent of the if statement in the loop version:
        #   if self.allocated_weights[idx] > 0 or sgn > 0:
        ss = ss[np.logical_or(
            np.isin(ss[:, 0], np.nonzero(allocated_weights > 0)),  # self.allocated_weights[idx] > 0
            ss[:, 1] > 0)  # sgn > 0
        ]

        if len(ss):
            idx = ss[:, 0].astype(int)
            sgn = ss[:, 1]

            # Update all weights by the array of signs
            allocated_weights[idx] += sgn

            # Don't allow negative weights
            allocated_weights[allocated_weights < 0] = 0

            # Add in the signed sampled to the running marginal, to save the cost
            # of re-calculating the marginals.
            running_allocated_marginals += \
                np.multiply(sample_pop[idx], sgn.reshape(ss.shape[0], -1)).sum()

        if te < min_error:
            break
