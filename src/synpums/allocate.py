# Copyright (c) 2020 Civic Knowledge. This file is licensed under the terms of the
# MIT license included in this distribution as LICENSE

import logging
import re
from collections import defaultdict, deque
from pathlib import Path
from time import time

import pandas as pd

from synpums.util import *

''
_logger = logging.getLogger(__name__)


def sample_to_sum(N, df, col, weights):
    """Sample a number of records from a dataset, then return the smallest set of
    rows at the front of the dataset where the weight sums to more than N"""

    t = df.sample(n=N, weights=weights, replace=True)
    # Get the number of records that sum to N.

    arg = t[col].cumsum().sub(N).abs().astype(int).argmin()

    return t.iloc[:arg + 1]


def rms(s):
    """Root mean square"""
    return np.sqrt(np.sum(np.square(s)))


def make_acs_target_df(acs, columns, geoid):
    t = acs.loc[geoid]

    target_map = {c + '_m90': c for c in columns if "WGTP" not in columns}

    target_df = pd.DataFrame({
        'est': t[target_map.values()],
        'm90': t[target_map.keys()].rename(target_map)
    })

    target_df['est_min'] = target_df.est - target_df.m90
    target_df['est_max'] = target_df.est + target_df.m90

    target_df.loc[target_df.est_min < 0, 'est_min'] = 0

    return target_df.astype('Int64')


def geoid_path(geoid):
    from pathlib import Path
    from geoid.acs import AcsGeoid

    go = AcsGeoid.parse(geoid)

    try:
        return Path(f"{go.level}/{go.stusab}/{go.county:03d}/{str(go)}.csv")
    except AttributeError:
        return Path(f"{go.level}/{go.stusab}/{str(go)}.csv")


class AllocationTask(object):
    """Represents the allocation process to one tract"""

    def __init__(self, region_geoid, puma_geoid, acs_ref, hh_ref, cache_dir):

        self.region_geoid = region_geoid
        self.puma_geoid = puma_geoid

        self.acs_ref = acs_ref
        self.hh_ref = hh_ref

        self.cache_dir = cache_dir

        self.sample_pop = None
        self.sample_weights = None
        self.target_marginals = None
        self.allocated_weights = None
        self.household_count = None
        self.population_count = None
        self.gq_count = None

        self.gq_cols = None
        self.sex_age_cols = None
        self.hh_size_cols = None
        self.hh_race_type_cols = None
        self.hh_eth_type_cols = None

        self._init = False

        self.running_allocated_marginals = None

        # A version of the sample_pop constructed by map_cp, added as an instance var so
        # the probabilities can be manipulated during the vector walk.
        self.cp_df = None
        self.cp_prob = None

    @property
    def row(self):
        from geoid.acs import AcsGeoid
        tract = AcsGeoid.parse(self.region_geoid)
        return [tract.state, tract.stusab, tract.county, self.region_geoid, self.puma_geoid, str(self.acs_ref),
                str(self.hh_ref)]

    def init(self, use_sample_weights=False):
        """Load all of the data, just before running the allocation"""

        if isinstance(self.hh_ref, pd.DataFrame):
            hh_source = self.hh_ref
        else:
            hh_source = pd.read_csv(self.hh_ref, index_col='SERIALNO', low_memory=False) \
                .drop(columns=['geoid'], errors='ignore').astype('Int64')

        if isinstance(self.acs_ref, pd.DataFrame):
            acs = self.acs_ref
        else:
            acs = pd.read_csv(self.acs_ref, index_col='geoid', low_memory=False)

        self.serialno = hh_source.index

        # Col 0 is the WGTP column
        w_cols = [c for c in hh_source.columns if "WGTP" in c]
        not_w_cols = [c for c in hh_source.columns if "WGTP" not in c]

        # Not actually a sample pop --- populations are supposed to be unweighted
        self.sample_pop = hh_source[['WGTP'] + not_w_cols].iloc[:, 1:].reset_index(drop=True).astype(int)

        self.sample_weights = hh_source.iloc[:, 0].reset_index(drop=True).astype(int)

        assert self.sample_pop.shape[0] == self.sample_weights.shape[0]

        not_w_cols = [c for c in hh_source.columns if "WGTP" not in c]

        self.target_marginals = make_acs_target_df(acs, not_w_cols, self.region_geoid)

        self.household_count = acs.loc[self.region_geoid].b11016_001
        self.population_count = acs.loc[self.region_geoid].b01003_001
        self.gq_count = acs.loc[self.region_geoid].b26001_001

        self.allocated_weights = np.zeros(len(self.sample_pop))

        self.running_allocated_marginals = pd.Series(0, index=self.target_marginals.index)

        # Sample pop, normalized to unit length to speed up cosine similarity
        self.sample_pop_norm = vectors_normalize(self.sample_pop.values)

        # These are only for debugging.
        self.hh_source = hh_source
        self.tract_acs = acs

        # Column sets
        self.gq_cols = ['b26001_001']
        self.sex_age_cols = [c for c in self.hh_source.columns if c.startswith('b01001')]
        self.hh_size_cols = [c for c in self.hh_source.columns if c.startswith('b11016')]

        p = re.compile(r'b11001[^hi]_')
        self.hh_race_type_cols = [c for c in self.hh_source.columns if p.match(c)]

        p = re.compile(r'b11001[hi]_')
        self.hh_eth_type_cols = [c for c in self.hh_source.columns if p.match(c)]

        # We will use this identity in the numpy version of step_scjhedule
        # assert all((self.cp.index / 2).astype(int) == self['index'])

        self.rng = np.random.default_rng()

        self.init_cp(use_sample_weights=use_sample_weights)

        self._init = True

        return acs

    def init_cp(self, use_sample_weights=False):
        self.use_sample_weights = use_sample_weights
        self.cp = self.make_cp(self.sample_pop)

    def make_cp(self, sp):
        """Make a version of the sample population with two records for each
        row, one the negative of the one before it. This is used to generate
        rows that can be used in the vector walk."""

        weights = self.sample_weights if self.use_sample_weights else 1

        sp_pos = sp.copy()
        sp_pos['sign'] = 1
        sp_pos.insert(0, 'select_weight', weights)

        sp_neg = sp.copy() * -1
        sp_neg['sign'] = -1
        sp_neg.insert(0, 'select_weight', weights)

        # Combined sample population, for calculating trial errors
        cp = pd.concat([sp_pos, sp_neg])
        cp.index.name = 'index'
        cp = cp.set_index('sign', append=True)

        # Probability of row being selected in the step_schedule function

        return cp.sort_index()

    def set_cp_prob(self, cp_prob):
        pass

    @property
    def path(self):
        return Path(self.cache_dir).joinpath(geoid_path(str(self.region_geoid))).resolve()

    @property
    def pums(self):
        """Return the PUMS household and personal records for this PUMA"""
        from .pums import build_pums_dfp_dfh
        from geoid.acs import Puma
        puma = Puma.parse(self.puma_geoid)

        dfp, dfh = build_pums_dfp_dfh(puma.stusab, year=2018, release=5)

        return dfp, dfh

    def get_saved_frame(self):

        if self.path.exists():
            return pd.read_csv(self.path.resolve(), low_memory=False)
        else:
            return None

    def save_frame(self):

        self.path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            'serialno': self.serialno,
            'weight': self.allocated_weights
        })

        df = df[df.weight > 0]

        df.to_csv(self.path, index=False)

    def load_frame(self):

        df = pd.read_csv(self.path, low_memory=False)
        self.init()

        aw, _ = df.align(self.sample_weights, axis=0)

        self.allocated_weights = df.set_index('serialno').reindex(self.serialno).fillna(0).values[:, 0]

    def inc(self, rown, n=1):

        if self.allocated_weights[rown] > 0 or n > 0:
            self.allocated_weights[rown] += n  # Increment the count column
            self.running_allocated_marginals += n * self.sample_pop.iloc[rown]

    @property
    def allocated_pop(self):
        return self.sample_pop.mul(self.allocated_weights, axis=0)

    @property
    def allocated_marginals(self):
        t = self.allocated_pop.sum()
        t.name = 'allocated_marginals'
        return t

    def calc_region_sum(self):
        return self.allocated_weights.sum()

    def column_diff(self, column):

        return (self.target_marginals.est[column] - self.allocated_marginals[column])

    @property
    def target_diff(self):
        return self.target_marginals.est - self.allocated_marginals

    @property
    def rel_target_diff(self):
        return ((self.target_marginals.est - self.allocated_marginals) / self.target_marginals.est) \
            .replace({np.inf: 0, -np.inf: 0})

    @property
    def running_target_diff(self):
        return self.target_marginals.est - self.running_allocated_marginals

    @property
    def error_frame(self):
        return self.target_marginals \
            .join(self.allocated_marginals.to_frame('allocated')) \
            .join(self.m90_error.to_frame('m_90')) \
            .join(self.target_diff.to_frame('diff')) \
            .join(self.rel_target_diff.to_frame('rel'))

    @property
    def total_error(self):
        """Magnitude of the error vector"""
        return np.sqrt(np.sum(np.square(self.target_diff)))

    @property
    def running_total_error(self):
        """Magnitude of the error vector"""
        return np.sqrt(np.sum(np.square(self.running_target_diff)))

    @property
    def m90_error(self):
        """Error that is relative to the m90 limits. Any value within the m90 limits is an error of 0"""

        # There the allocated marginal is withing the m90 range, return the target marginal estimate
        # otherwise, return amount of the  allocated marginals that is outside of the m90 range

        t = self.allocated_marginals - self.target_marginals.est
        t[self.allocated_marginals.between(self.target_marginals.est_min, self.target_marginals.est_max)] = 0
        t[t > self.target_marginals.m90] = t - self.target_marginals.m90
        t[t < -1 * self.target_marginals.m90] = t + self.target_marginals.m90

        return t

    @property
    def m90_total_error(self):
        return np.sqrt(np.sum(np.square(self.m90_error)))

    @property
    def m90_rms_error(self):
        """RMS error of the m90 differences. Like m90 total error, but divides
        by the number of marginal value variables"""

        return np.sqrt(np.sum(np.square(self.m90_total_error)) / len(self.target_marginals))

    # Equivalent to cosine similarity when the vectors are both normalized
    def cosine_similarities(self):
        '''Calculate the cosine similaries for all of the sample population records
        to the normalized error vector'''

        return self.sample_pop_norm.dot(vector_normalize(self.target_diff.values).T)

    def sample_multicol(self, columns):

        targets = self.target_marginals.est

        frames = []
        for col in columns:

            target = targets.loc[col]

            if target > 0:
                t = self.sample_pop[self.sample_pop[col] > 0]
                w = self.sample_weights[self.sample_pop[col] > 0]

                if len(t) > 0 and w.sum() > 0:
                    frames.append(sample_to_sum(target, t, col, w))

        if frames:
            return pd.concat(frames)
        else:
            return None

    def _pop_to_weights(self, pop):
        '''Return weights by counting the records in a population'''

        t = pop.copy()
        t.insert(0, 'dummy', 1)
        t = t.groupby(t.index).dummy.count()
        t = t.align(self.sample_weights)[0].fillna(0).values
        return t

    def initialize_weights_set_sample(self, f=0.85):
        """Sample from the sample population one column at a time, in groups of
        columns that describe exclusive measures ( a household can contribute to
        only one marginal column) Then, resample the population to match the correct number of
        households"""

        assert self._init

        if f == 0:
            return

        frames = [
            self.sample_multicol(self.hh_race_type_cols + self.gq_cols),
            self.sample_multicol(self.hh_eth_type_cols),
            self.sample_multicol(self.sex_age_cols),
        ]

        frames = [f for f in frames if f is not None]

        if len(frames) == 0:
            return

        # An initial population, which is of the wrong size, so just
        # convert it to weights
        t = pd.concat(frames)
        initial_weights = self._pop_to_weights(t)

        # These use those weights to re-sample the population.

        target_count = self.household_count + self.gq_count

        # Sample some fraction less than the target count, so we can vector walk to the final value
        target_count = int(target_count * f)

        t = self.sample_pop.sample(target_count, weights=initial_weights, replace=True)
        self.allocated_weights = self._pop_to_weights(t)

        self.running_allocated_marginals = self.allocated_marginals

    def _rake(self, f=1):

        # Sort the columns by the error
        cols = list(self.error_frame.sort_values('diff', ascending=False).index)
        # cols = random.sample(list(self.sample_pop.columns), len(self.sample_pop.columns)):

        for col in cols:

            b = self.sample_pop[col].mul(self.allocated_weights, axis=0).sum()

            if b != 0:
                a = self.target_marginals.loc[col].replace({pd.NA: 0}).est
                r = a / b * f
                self.allocated_weights[self.sample_pop[col] > 0] *= r

        self.allocated_weights = np.round(self.allocated_weights, 0)

    def initialize_weights_raking(self, n_iter=5, initial_weights='sample'):
        """Set the allocated weights to an initial value by 1-D raking, adjusting the
        weights to fit the target marginal value for each column. """

        if initial_weights == 'sample':
            assert self.allocated_weights.shape == self.sample_weights.shape
            self.allocated_weights = self.sample_weights
        else:
            self.allocated_weights = np.ones(self.allocated_weights.shape)

        for i in range(n_iter):

            # Sort the columns by the error
            cols = list(self.error_frame.sort_values('diff', ascending=False).index)
            # cols = random.sample(list(self.sample_pop.columns), len(self.sample_pop.columns)):

            for col in cols:

                b = self.sample_pop[col].mul(self.allocated_weights, axis=0).sum()

                if b != 0:
                    a = self.target_marginals.loc[col].replace({pd.NA: 0}).est
                    r = a / b
                    self.allocated_weights[self.sample_pop[col] > 0] *= r

        self.allocated_weights = np.round(self.allocated_weights, 0)

        try:
            self.allocated_weights = self.allocated_weights.values
        except AttributeError:
            pass

    def initialize_weights_sample(self):

        """Initialize the allocated weights proportional to the sample population weights,
        adjusted to the total population. """

        self.allocated_weights = (self.sample_weights / (self.sample_weights.sum())).multiply(
            self.household_count).values.round(0).astype(float)

    def step_schedule_np(self, i, N, cp, te, td, step_size_max, step_size_min, reversal_rate, p=None):
        """ Return the next set of samples to add or remove

        :param i: Loop index
        :param N: Max number of iterations
        :param cp: Sample population, transformed by make_cp
        :param te: Total error
        :param td: Marginals difference vector
        :param step_size_max: Maximum step size
        :param step_size_min: Minimum step size
        :param reversal_rate: Probability to allow an increase in error
        :param p: Probability to select each sample row. If None, use column 2 of cp

        :return: Records to add or remove from the allocated population

        """

        # Min  number of record to return in this step. The error-increasing records are in
        # addition to this number
        step_size = int((step_size_max - step_size_min) * ((N - i) / N) + step_size_min)

        # Portion of the error-decreasing records to permit as error0increasing records.
        r = i / N * reversal_rate

        new_marginals = cp.copy()

        # Compute change in each column of the error vector for adding or subtracting in
        # each of the sample population records
        # idx 0 is the index of the row in self.sample_pop
        # idx 1 is the sign, 1 or -1
        # idx 2 is the selection weight
        # idx 3 and up are the census count columns
        new_marginals[:, 3:] += td

        # For each new error vector, compute total error ( via vector length). By
        # removing the current total error, we get the change in total error for
        # adding or removing each row. ( positive values are better )
        total_errors = (np.sqrt(np.square(new_marginals[:, 2:]).sum(axis=1)) - te)

        reducing_error = np.argwhere(total_errors > 0)  # Error reducing indexes
        increasing_error = np.argwhere(total_errors < 0)  # Error increasing indexes

        # Prepare the probabilities so they have the correct shape
        # and sum to 1

        if p is None:
            p = new_marginals[:, 2]

        # Extract just the records for increasing or reducing the error, the
        # re-normalize to 1, as is required by np.random.choice
        try:
            pr = p[reducing_error].reshape(-1)
            pr = pr / np.sum(pr)
        except IndexError:
            print(len(reducing_error), np.max(reducing_error))
            pr = None

        try:
            pi = p[increasing_error].reshape(-1)
            pi = pi / np.sum(pi)
        except IndexError:
            print(len(increasing_error), np.max(increasing_error))
            pi = None

        idx = np.concatenate([self.rng.choice(reducing_error, int(step_size), p=pr)[:, 0],
                              self.rng.choice(increasing_error, int(step_size * r), p=pi)[:, 0]])

        # Columns are : 'index', 'sign', 'delta_err'
        return np.hstack([cp[idx][:, 0:2], total_errors[idx].reshape(-1, 1)])  # Return the index and sign columns of cp

    def _vector_walk(self, N=2000, min_iter=750, target_error=0.03, step_size_min=3, step_size_max=15,
                     reversal_rate=.3, max_ssm=250, cb=None, memo=None):
        """Allocate PUMS records to this object's region.

        Args:
            N:
            min_iter:
            target_error:
            step_size_min:
            step_size_max:
            reversal_rate:
            max_ssm:

        """

        assert self._init

        if target_error < 1:
            target_error = self.household_count * target_error

        min_allocation = None  # allocated weights at last minimum
        steps_since_min = 0

        # CP with no rows with no nonzero weight in the allocations
        # Removing zero weight samples only works when the allocated weights
        # is initialized before starting the vector walk.
        aw = pd.Series(self.allocated_weights, name='weight')
        sp = self.sample_pop.iloc[aw[aw > 0].index] if aw.sum() > 0 else self.sample_pop
        self.cp_df = self.make_cp(sp).reset_index()
        cp = self.cp_df.values.astype(int)

        self.cp_prob = self.cp_df.select_weight.values

        min_error = self.total_error

        self.running_allocated_marginals = self.allocated_marginals

        if cb:
            cb(self, memo)

        for i in range(N):

            td = self.running_target_diff
            te = vector_length(td)

            if te < min_error:
                min_error = te
                min_allocation = self.allocated_weights
                steps_since_min = 0
            else:
                steps_since_min += 1

            min_error = min(te, min_error)

            if (i > min_iter and te < target_error) or steps_since_min > max_ssm:
                break

            try:
                ss = self.step_schedule_np(i, N, cp, te, td.values.astype(int),
                                           step_size_min, step_size_max, reversal_rate,
                                           p=self.cp_prob)
            except ValueError:
                # Usually b/c numpy choice() got an empty array
                continue

            # This is the equivalent of the if statement in the loop version in _loop_assignment
            #   if self.allocated_weights[idx] > 0 or sgn > 0:
            # The loop equivalent is:
            #
            # for j, row in enumerate(ss):
            #     idx = int(row[0])
            #     sgn = row[1]
            #     if self.allocated_weights[idx] > 0 or sgn > 0:
            #         self.allocated_weights[idx] += sgn  # Increment the count column
            #         self.running_allocated_marginals += sgn * self.sample_pop.iloc[idx]

            ss = ss[np.logical_or(
                np.isin(ss[:, 0], np.nonzero(self.allocated_weights > 0)),  # self.allocated_weights[idx] > 0
                ss[:, 1] > 0)  # sgn > 0
            ]

            # Assign the steps from the step schedule into the allocated weights
            if len(ss):
                idx = ss[:, 0].astype(int)
                sgn = ss[:, 1]

                # Update all weights by the array of signs
                self.allocated_weights[idx] += sgn

                # Don't allow negative weights
                self.allocated_weights[self.allocated_weights < 0] = 0

                # Add in the signed sampled to the running marginal, to save the cost
                # of re-calculating the marginals.
                self.running_allocated_marginals += \
                    np.multiply(self.sample_pop.iloc[idx], sgn.reshape(ss.shape[0], -1)).sum()

            yield (i, te, min_error, steps_since_min, len(ss))

        if min_allocation is not None:
            self.allocated_weights = min_allocation

    def vector_walk(self, N=2000, min_iter=750, target_error=0.03, step_size_min=3, step_size_max=10,
                    reversal_rate=.3, max_ssm=250, callback=None, init_cb=None, memo=None):
        """Consider the target state and each household to be a vector. For each iteration
        select a household vector with the best cosine similarity to the vector to the
        target and add that household to the population. """

        assert self._init

        rows = []
        ts = time()
        errors = deque(maxlen=20)
        errors.extend([self.total_error] * 20)

        for i, te, min_error, steps_since_min, n_iter in self._vector_walk(
                N=N, min_iter=min_iter, target_error=target_error,
                step_size_min=step_size_min, step_size_max=step_size_max,
                reversal_rate=reversal_rate, max_ssm=max_ssm,
                cb=init_cb, memo=memo):

            d = {'i': i, 'time': time() - ts, 'step_size': n_iter, 'error': te,
                 'target_error': target_error,
                 'total_error': te,
                 'size': np.sum(self.allocated_weights),
                 'ssm': steps_since_min,
                 'min_error': min_error,
                 'mean_error': np.mean(errors),
                 'std_error': np.std(errors),
                 }

            rows.append(d)
            errors.append(te)

            if callback and i % 10 == 0:
                callback(self, d, memo)

        return rows

    @classmethod
    def get_us_tasks(cls, cache_dir, sl='tract', year=2018, release=5, limit=None, ignore_completed=True):
        """Return all of the tasks for all US states"""

        from geoid.censusnames import stusab

        tasks = []

        for state in stusab.values():
            state_tasks = cls.get_state_tasks(cache_dir, state, sl, year, release, limit, ignore_completed)

            tasks.extend(state_tasks)

        return tasks

    @classmethod
    def get_tasks(cls, cache_dir, state, sl='tract', year=2018, release=5,
                  limit=None, use_tqdm=False, ignore_completed=True):

        if state.upper() == 'US':
            return cls.get_us_tasks(cache_dir, sl, year, release, limit, use_tqdm, ignore_completed)
        else:
            return cls.get_state_tasks(cache_dir, state, sl, year, release, limit, ignore_completed)

    @classmethod
    def get_state_tasks(cls, cache_dir, state, sl='tract', year=2018, release=5,
                        limit=None, ignore_completed=True):
        """Fetch ( possibly download) the source data to generate allocation tasks,
        and cache the data if a cache_dir is provided"""

        from .acs import puma_tract_map
        from synpums import build_acs, build_pums_households
        from functools import partial
        import pickle

        _logger.info(f'Loading tasks for {state} from cache {cache_dir}')

        cp = Path(cache_dir).joinpath('tasks', 'source', f"{state}-{year}-{release}/")
        cp.mkdir(parents=True, exist_ok=True)
        asc_p = cp.joinpath("acs.csv")
        hh_p = cp.joinpath("households.csv")
        tasks_p = cp.joinpath("tasks.pkl")

        if limit:
            from itertools import islice
            limiter = partial(islice, limit)
        else:
            def limiter(g, *args, **kwargs):
                yield from g

        if tasks_p and tasks_p.exists():
            with tasks_p.open('rb') as f:
                _logger.debug(f"Returning cached tasks from  {str(tasks_p)}")
                return pickle.load(f)

        # Cached ACS files
        if asc_p and asc_p.exists():
            tract_acs = pd.read_csv(asc_p, index_col='geoid', low_memory=False)
        else:
            tract_acs = build_acs(state, sl, year, release)
            if asc_p:
                tract_acs.to_csv(asc_p, index=True)

        # Cached Households
        if hh_p and hh_p.exists():
            households = pd.read_csv(hh_p, index_col='SERIALNO', low_memory=False)
        else:
            households = build_pums_households(state, year=year, release=release)
            if hh_p:
                households.to_csv(hh_p, index=True)

        hh = households.groupby('geoid')

        hh_file_map = {}

        for key, g in hh:
            puma_p = cp.joinpath(f"pumas/{key}.csv")
            puma_p.parent.mkdir(parents=True, exist_ok=True)

            _logger.debug(f"Write puma file {str(puma_p)}")
            g.to_csv(puma_p)

            hh_file_map[key] = puma_p

        pt_map = puma_tract_map()

        tasks = []
        for tract_geoid, targets in limiter(tract_acs.iterrows(), desc='Generate Tasks'):
            try:
                puma_geoid = pt_map[tract_geoid]

                t = AllocationTask(tract_geoid, puma_geoid, asc_p, hh_file_map[puma_geoid], cache_dir)

                if not t.path.exists() or ignore_completed is False:
                    tasks.append(t)

            except Exception as e:
                print("Error", tract_geoid, type(e), e)

        if tasks_p:
            with tasks_p.open('wb') as f:
                _logger.debug(f"Write tasks file {str(tasks_p)}")
                pickle.dump(tasks, f, pickle.HIGHEST_PROTOCOL)

        return tasks

    def run(self, *args, callback=None, init_callback=None, memo=None, **kwargs):

        self.init()

        self.initialize_weights_sample()

        rows = self.vector_walk(*args, callback=callback, init_cb=init_callback,
                                memo=memo, **kwargs)

        self.save_frame()

        return rows

    def run_2stage(self, *args, callback=None, init_callback=None, memo=None, **kwargs):

        self.init()

        self.init_cp(use_sample_weights=True)

        self.initialize_weights_sample()

        kwargs['N'] = 2000
        rows = self.vector_walk(*args, callback=callback, init_cb=init_callback, memo=memo, **kwargs)

        kwargs['N'] = 200
        self.init_cp(use_sample_weights=False)
        rows.extend(self.vector_walk(*args, callback=callback, init_cb=init_callback, memo=memo, **kwargs))

        self.save_frame()

        return rows


class PumaAllocator(object):
    """Simultaneously allocate all of the tracts in a pums, attempting to reduce the
    error between the sum of the allocated weights and the PUMS weights"""

    def __init__(self, puma_geoid, tasks, cache_dir):

        self.cache_dir = cache_dir
        self.puma_geoid = puma_geoid
        self.tasks = tasks

        pums_files = [task.hh_ref for task in self.tasks]
        assert all([e == pums_files[0] for e in pums_files])

        self.pums = pd.read_csv(pums_files[0], low_memory=False)

        self.weights = pd.DataFrame({
            'allocated': 0,
            'pums': self.pums.WGTP
        })

        self.prob = None

    @classmethod
    def get_tasks(cls, cache_dir, state, year=2018, release=5):

        tasks = AllocationTask.get_state_tasks(cache_dir, state, sl='tract', year=2018, release=5)

        puma_tasks = defaultdict(list)

        for task in tasks:
            puma_tasks[task.puma_geoid].append(task)

        return puma_tasks

    @classmethod
    def get_allocators(cls, cache_dir, state, year=2018, release=5):

        tasks = AllocationTask.get_state_tasks(cache_dir, state, sl='tract', year=2018, release=5)

        puma_tasks = defaultdict(list)

        for task in tasks:
            puma_tasks[task.puma_geoid].append(task)

        return [PumaAllocator(puma_geoid, tasks, cache_dir) for puma_geoid, tasks in puma_tasks.items()]

    def _update_weights(self):

        self.weights['allocated'] = np.sum(np.array([task.allocated_weights for task in self.tasks]), axis=0)

        self.weights['dff'] = self.weights.allocated - self.weights.pums
        self.weights['rdff'] = (self.weights.dff / self.weights.pums).fillna(0)
        self.weights['p'] = self.weights.rdff

    def _update_probabilities(self):
        """Update the running cp_probs, the probabilities for selecting each PUMS
        household from the sample_pop, based on the error in weights for
        the households at the Puma level"""

        self.weights['p_pos'] = -self.weights.p.where(self.weights.p < 0, 0)
        self.weights['p_neg'] = self.weights.p.where(self.weights.p > 0, 0)
        self.prob = np.array(self.weights[['p_neg', 'p_pos']].values.flat)
        return self.prob

    def initialize_weights(self):
        """Initialize the weights of all of the tasks"""
        from tqdm import tqdm
        for task in tqdm(self.tasks):
            task.initialize_weights_set_sample()

        self._update_weights()

    @property
    def rms_error(self):
        """RMS error in all of the individual task marginals"""
        t = pd.concat([task.target_diff for task in self.tasks], axis=1).values
        return np.sqrt(np.mean(np.nan_to_num(np.square(t))))

    @property
    def rms_weight_error(self):
        return np.sqrt(np.mean(np.square(self.weights.dff)))

    def vector_walk(self, N=2000, min_iter=750, target_error=0.03, step_size_min=1, step_size_max=10,
                    reversal_rate=.3, max_ssm=250, callback=None, memo=None):
        """Consider the target state and each household to be a vector. For each iteration
        select a household vector with the best cosine similarity to the vector to the
        target and add that household to the population. """

        rows = []

        from tqdm import tqdm

        ts = time()

        for task in tqdm(self.tasks):

            self.init()

            errors = deque(maxlen=20)
            errors.extend([task.total_error] * 20)

            for i, te, min_error, steps_since_min, n_iter in task._vector_walk(
                    N=N, min_iter=min_iter, target_error=target_error,
                    step_size_min=step_size_min, step_size_max=step_size_max,
                    reversal_rate=reversal_rate, max_ssm=max_ssm):

                d = {'i': i, 'time': time() - ts, 'step_size': n_iter, 'error': te,
                     'target_error': target_error,
                     'size': np.sum(task.allocated_weights),
                     'ssm': steps_since_min,
                     'min_error': min_error,
                     'mean_error': np.mean(errors), 'std_error': np.std(errors),
                     'mean_dev_error': np.mean(np.diff(errors)),
                     'mean_std_dev_error': np.std(np.diff(errors)),
                     'mean_2dev_error': np.mean(np.diff(np.diff(errors))),
                     'mean_std_2dev_error': np.std(np.diff(np.diff(errors))),
                     }

                rows.append(d)
                errors.append(te)

                if callback and i % 10 == 0:
                    callback(self, d, memo)

        self._update_weights()

        return rows

    def tune_puma_allocation(self):
        """Re-run all of the tasks in the puma, trying to reduce the discrepancy
        between the """
        task_iters = [(task, iter(task._vector_walk())) for task in self.tasks]

        for task, task_iter in task_iters:
            try:

                self._update_weights()
                prob = self._update_probabilities()
                task.cp_prob = prob
                row = next(task_iter)
                print(task.region_geoid, self.rms_error, self.rms_weight_error)

            except StopIteration:
                print(task.region_geoid, 'stopped')
