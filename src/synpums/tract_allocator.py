from synpums import *
from synpums.util import *


class TractAllocator(object):

    @classmethod
    def load(cls, path, puma_geoid, region_geoid=None, year=2018, release=5):

        households, marg, m90, tract_geoids = load_allocation_data(path, puma_geoid, year, release)

        if region_geoid is not None:

            return TractAllocator(households, marg, m90, puma_geoid, region_geoid)
        else:
            return [TractAllocator(households.copy(), marg.copy(), m90.copy(), puma_geoid, region_geoid)
                    for region_geoid in list(marg.columns[1:])]

    def __init__(self, households, marg, m90, puma_geoid, region_geoid):

        self.puma_geoid = puma_geoid
        self.region_geoid = region_geoid

        self.columns = list(sorted(set(marg.index) & set(households.iloc[:, 1:])))

        self.puma_marginals = marg.loc[self.columns, :]
        self.puma_errors = m90.loc[self.columns, :]
        self.households = households.loc[:, self.columns]

        self.serialnos = pd.Series(households.index).values
        self.sample_weights = households.loc[:, 'WGTP'].reset_index(drop=True).astype(int).values
        self.sample_pop = households.loc[:, self.columns].reset_index(drop=True).astype(int).values

        self.target_marginals = marg.loc[self.columns, self.region_geoid].values
        self.target_errors = m90.loc[self.columns, self.region_geoid].values

        self.allocated_weights = np.zeros(self.serialnos.shape)
        self.unallocated_weights = self.sample_weights.copy()

        self.running_allocated_marginals = np.zeros(self.target_marginals.shape)

        self.rng = np.random.default_rng()

        #
        # Columns Sets
        #

        # Column sets
        self.gq_cols = ['b26001_001']
        self.sex_age_cols = [c for c in self.households.columns if c.startswith('b01001')]
        self.hh_size_cols = [c for c in self.households.columns if c.startswith('b11016')]

        p = re.compile(r'b11001[^hi]_')
        self.hh_race_type_cols = [c for c in self.households.columns if p.match(c)]

        p = re.compile(r'b11001[hi]_')
        self.hh_eth_type_cols = [c for c in self.households.columns if p.match(c)]

        p = re.compile(r'b19025')
        self.hh_income_cols = [c for c in self.households.columns if p.match(c)]

        #
        # Totals
        #

        self.household_count = marg.loc['b11016_001', self.region_geoid]
        self.population_count = marg.loc['b01003_001', self.region_geoid]
        self.gq_count = marg.loc['b26001_001', self.region_geoid]
        self.total_count = self.household_count + self.gq_count

    def initialize_weights(self, unallocated_weights=None):

        """Initialize the allocated weights proportional to the sample population weights,
        adjusted to the total population. """

        if unallocated_weights is not None:
            self.unallocated_weights = unallocated_weights

            try:
                self.unallocated_weights = self.unallocated_weights.values
            except AttributeError:
                pass

        wgt_norm = (self.sample_weights / (self.sample_weights.sum()))
        self.allocated_weights = (wgt_norm * self.household_count).round(0).astype(int)
        self.unallocated_weights -= self.allocated_weights

    @property
    def allocated_pop(self):
        # return self.sample_pop.mul(self.allocated_weights, axis=0)
        return self.allocated_weights.reshape(len(self.sample_pop), -1) * self.sample_pop

    @property
    def unallocated_pop(self):
        # return self.sample_pop.mul(self.allocated_weights, axis=0)
        return self.unallocated_weights.reshape(len(self.sample_pop), -1) * self.sample_pop

    @property
    def allocated_count(self):
        # return self.sample_pop.mul(self.allocated_weights, axis=0)
        return self.allocated_weights.sum()

    @property
    def allocated_marginals(self):
        return self.allocated_pop.sum(axis=0)

    @property
    def unallocated_marginals(self):
        pop = self.unallocated_weights.reshape(len(self.sample_pop), -1) * self.sample_pop
        return pop.sum(axis=0)

    @property
    def target_diff(self):
        return self.target_marginals - self.allocated_marginals

    @property
    def total_error(self):
        """Magnitude of the error vector"""
        return np.sqrt(np.sum(np.square(self.target_diff)))

    @property
    def m90_error(self):
        """Error that is relative to the m90 limits. Any value within the m90 limits is an error of 0"""

        # There the allocated marginal is withing the m90 range, return the target marginal estimate
        # otherwise, return amount of the  allocated marginals that is outside of the m90 range

        t = self.error_frame
        m90_diff = np.clip(((t.est-t.alloc).abs()-np.abs(t.err)),a_min=0, a_max=None)

        return np.sqrt(np.sum(np.square(m90_diff)))


    @property
    def error_frame(self):
        return pd.DataFrame({
            'idx': self.columns,
            'est': self.target_marginals,
            'err': self.target_errors,
            'alloc': self.allocated_marginals,
            'unalloc': self.unallocated_marginals,
            'dff': self.target_diff,
            'rdff': self.target_diff / (self.target_marginals + 1)  # Avoid divide by zero
        }).set_index('idx')

    def step_schedule(self, n_reductions, n_increases):
        """return a schedule of which household records to add or remove

        The returned schedule is a set of tuples, with these values:

            0: 0 for add a record, 1 for remove
            1: The index of the household record to add or remove
            2: 1 if the record will increase the error, -1 if it will reduce it.

        """

        td = self.target_diff
        te = self.total_error

        # Compute the difference in total error for add in each of the
        # records
        t_add = np.sqrt(np.square(td + self.sample_pop).sum(axis=1)) - te
        # Extract weights for increasing the error ( t_add is positive)
        # or reducing it ( t_add is negative). The base weight is the number
        # of unallocated records, times the magnitude of the change in the error,
        # although for the error increases, the magnitude is not considered,
        w_add_red = self.unallocated_weights * (t_add < 0) * np.abs(t_add)
        w_add_inc = self.unallocated_weights * (t_add > 0)

        # Ditto, but for removing records.
        t_rem = np.sqrt(np.square(td - self.sample_pop).sum(axis=1)) - te
        w_rem_red = self.allocated_weights * (t_rem < 0) * np.abs(t_rem)
        w_rem_inc = self.allocated_weights * (t_rem > 0)

        # The selection frame is the popualtion indexed, doubled up.
        # the first half is for additions, and the second half is for subtractions.
        # When these values are samples, we can use np.divmod to get a flag for
        # whether it is an addition or subtraction, and the original index
        t = np.arange(len(t_add) + len(t_rem))

        # Concatenate the weights to match the sample indexes and
        # normalize the sum to 1.
        p_red = np.clip(np.concatenate([w_add_red, w_rem_red]), a_min=0, a_max=None)
        s_red = p_red.sum()
        if s_red != 0:
            p_red = p_red / s_red
            red = [np.divmod(idx, len(t_add)) + (-1,) for idx in self.rng.choice(t, n_reductions, p=p_red)]

        else:
            red = []

        p_inc = np.clip(np.concatenate([w_add_inc, w_rem_inc]), a_min=0, a_max=None)
        s_inc = p_inc.sum()
        if s_inc != 0:
            p_inc = np.abs(p_inc / s_inc)
            inc = [np.divmod(idx, len(t_add)) + (1,) for idx in self.rng.choice(t, n_increases, p=p_inc)]

        else:
            inc = []

        # There should be a zero in one in every place there is a
        # nonzer0 in the other.
        # assert (p_inc*p_red).sum() == 0

        return red + inc

    def vector_walk(self, N=1800, init_step_size=10):

        rows = []

        for step_size in np.linspace(init_step_size, 1, N):
            step_size = int(step_size)
            rev_size = int(step_size / 25)

            for add_rem, idx, err_change in self.step_schedule(step_size, rev_size):

                sgn = 1 if add_rem == 1 else -1

                if (self.allocated_weights[idx] > 0 and sgn < 0) or (self.unallocated_weights[idx] > 0 and sgn > 0):
                    # self.running_allocated_marginals += (sgn * self.sample_pop[idx])
                    self.allocated_weights[idx] += sgn  # Increment the count column
                    self.unallocated_weights[idx] -= sgn

            rows.append([self.total_error, step_size, rev_size])

        return pd.DataFrame(rows, columns="err fwd_step rev_step".split())