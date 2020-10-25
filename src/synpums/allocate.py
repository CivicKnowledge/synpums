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
        self.unallocated_weights = None  # Initialized to the puma weights, gets decremented
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
        self.hh_income_cols = None

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

    def init(self, use_sample_weights=False, puma_weights=None):
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
        self.total_count = self.household_count + self.gq_count

        self.allocated_weights = np.zeros(len(self.sample_pop))
        self.unallocated_weights = puma_weights if puma_weights is not None else self.sample_weights.copy()
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

        p = re.compile(r'b19025')
        self.hh_income_cols = [c for c in self.hh_source.columns if p.match(c)]

        # We will use this identity in the numpy version of step_scjhedule
        # assert all((self.cp.index / 2).astype(int) == self['index'])

        self.rng = np.random.default_rng()

        self.make_cp(self.sample_pop)

        self._init = True

        return acs


    def make_cp(self, sp):
        """Make a version of the sample population with two records for each
        row, one the negative of the one before it. This is used to generate
        rows that can be used in the vector walk."""

        self.cp = pd.concat([sp, sp]).sort_index().reset_index()
        self.cp.insert(1, 'sign', 1)
        self.cp.insert(2, 'select_weight', 0)

        self.cp.iloc[0::2, 1:] = self.cp.iloc[0::2, 1:] * -1 # flip sign on the marginal counts

        self.update_cp()

        return self.cp

    def update_cp(self):

        self.cp.loc[0::2, 'select_weight'] = self.allocated_weights.tolist()
        self.cp.loc[1::2, 'select_weight'] = self.unallocated_weights.tolist()

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

    @property
    def results_frame(self):
        return pd.DataFrame({
            'geoid': self.region_geoid,
            'serialno': self.serialno,
            'weight': self.allocated_weights
        })

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
        self.unallocated_weights -= self.allocated_weights

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

        self.unallocated_weights -= self.allocated_weights

    def step_schedule_np(self, i, N,  te, td, step_size_max, step_size_min, reversal_rate):
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

        # Compute change in each column of the error vector for adding or subtracting in
        # each of the sample population records
        # idx 0 is the index of the row in self.sample_pop
        # idx 1 is the sign, 1 or -1
        # idx 2 is the selection weight
        # idx 3 and up are the census count columns

        expanded_pop = self.cp.values.astype(int)

        p = expanded_pop[:, 2]

        # For each new error vector, compute total error ( via vector length). By
        # removing the current total error, we get the change in total error for
        # adding or removing each row. ( positive values are better )
        total_errors = (np.sqrt(np.square(expanded_pop[:, 3:] + td).sum(axis=1))) - te

        # For error reducing records, sort them and then mutliply
        # the weights by a linear ramp, so the larger values of
        # reduction get a relative preference over the lower reduction values.

        gt0 = np.argwhere(total_errors > 0).flatten()  # Error reducing records
        srt = np.argsort(total_errors)  # Sorted by error
        reducing_error = srt[np.in1d(srt, gt0)][::-1]  # get the intersection. These are index values into self.cp

        # Selection probabilities, multiply by linear ramp to preference higher values.
        reducing_p = ((p[reducing_error]) * np.linspace(1, 0, len(reducing_error)))
        rps = np.sum(reducing_p)
        if rps > 0:
            reducing_p = np.nan_to_num(reducing_p / rps)
        else:
            reducing_p = []

        increasing_error = np.argwhere(total_errors < 0).flatten()  # Error increasing indexes
        increasing_p = p[increasing_error].flatten().clip(min=0)
        ips = np.sum(increasing_p)
        if ips != 0:
            increasing_p = np.nan_to_num(increasing_p / ips)  # normalize to 1
        else:
            increasing_p =[]

        # Min  number of record to return in this step. The error-increasing records are in
        # addition to this number
        step_size = int((step_size_max - step_size_min) * ((N - i) / N) + step_size_min)

        # Randomly select from each group of increasing or reducing indexes.

        cc = []
        if len(increasing_error) > 0 and ips > 0:
            cc.append(self.rng.choice(increasing_error, int(step_size * reversal_rate), p=increasing_p))

        if len(reducing_error) > 0 and rps > 0:
            cc.append(self.rng.choice(reducing_error, int(step_size), p=reducing_p))

        idx = np.concatenate(cc)

        # Columns are : 'index', 'sign', 'delta_err'
        delta_err = total_errors[idx].reshape(-1, 1).round(0).astype(int)

        return np.hstack([expanded_pop[idx][:, 0:2], delta_err])  # Return the index and sign columns of cp

    def _loop_asignment(self, ss):

        for j, (idx, sgn, *_) in enumerate(ss):
            idx = int(idx)
            if (self.allocated_weights[idx] > 0 and sgn < 0) or \
                    (self.unallocated_weights[idx]>0 and sgn > 0) :
                self.running_allocated_marginals += (sgn * self.sample_pop.iloc[idx])
                self.allocated_weights[idx] += sgn  # Increment the count column
                self.unallocated_weights[idx] -= sgn


    def _numpy_assignment(self, ss):
        # The following code is the numpy equivalent of the loop version of
        # assignment to the allocated marginals. It is about 20% faster than the loop

        # This selection on ss is the equivalent to this if statement in the loop version:
        #   if self.allocated_weights[idx] > 0 or sgn > 0:
        #
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

        min_error = self.total_error

        self.running_allocated_marginals = self.allocated_marginals

        if cb:
            cb(self, memo)

        for i in range(N):

            td = self.running_target_diff.values.astype(int)
            te = vector_length(td)

            # The unallocated weights can be updated both internally and externally --
            # the array can be shared among all tracts in the puma
            self.update_cp()

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
                ss = self.step_schedule_np(i, N, te, td,
                                           step_size_max, step_size_min, reversal_rate)

                self._loop_asignment(ss)

                yield (i, te, min_error, steps_since_min, len(ss))

            except ValueError as e:
                # Usually b/c numpy choice() got an empty array
                pass
                print(e)
                raise

        if min_allocation is not None:
            self.allocated_weights = min_allocation

    def vector_walk(self, N=2000, min_iter=750, target_error=0.03, step_size_min=3, step_size_max=10,
                    reversal_rate=.3, max_ssm=250, callback=None, init_cb=None, memo=None,
                    stats = True):
        """Consider the target state and each household to be a vector. For each iteration
        select a household vector with the best cosine similarity to the vector to the
        target and add that household to the population. """

        assert self._init

        rows = []
        ts = time()
        errors = deque(maxlen=20)
        errors.extend([self.total_error] * 20)

        g = self._vector_walk(
                N=N, min_iter=min_iter, target_error=target_error,
                step_size_min=step_size_min, step_size_max=step_size_max,
                reversal_rate=reversal_rate, max_ssm=max_ssm,
                cb=init_cb, memo=memo)

        if stats is not True:
            list(g)
            return []
        else:

            for i, te, min_error, steps_since_min, n_iter in g :

                d = {'i': i, 'time': time() - ts, 'step_size': n_iter, 'error': te,
                     'target_error': target_error,
                     'total_error': te,
                     'size': np.sum(self.allocated_weights),
                     'ssm': steps_since_min,
                     'min_error': min_error,
                     'mean_error': np.mean(errors),
                     'std_error': np.std(errors),
                     'uw_sum': np.sum(self.unallocated_weights),
                     'total_count': self.total_count
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


class PumaAllocator(object):
    """Simultaneously allocate all of the tracts in a pums, attempting to reduce the
    error between the sum of the allocated weights and the PUMS weights"""

    def __init__(self, puma_geoid, tasks, cache_dir, state, year=2018, release=5):

        self.cache_dir = cache_dir
        self.puma_geoid = puma_geoid
        self.tasks = tasks

        self.year = year
        self.release = release
        self.state = state

        pums_files = [task.hh_ref for task in self.tasks]
        assert all([e == pums_files[0] for e in pums_files])

        self.pums_file = pums_files[0]

        self._puma_target_marginals = None
        self._puma_allocated_marginals = None

        self._puma_max_weights = None
        self._puma_allocated_weights = None
        self._puma_unallocated_weights = None

        self.pums = pd.read_csv(pums_files[0], low_memory=False)

        self.weights = pd.DataFrame({
            'allocated': 0,
            'pums': self.pums.WGTP,  # Original PUMS weights
            'remaining': self.pums.WGTP  # Remaining
        })

        self.prob = None

        self.gq_cols = None
        self.sex_age_cols = None
        self.hh_size_cols = None
        self.hh_race_type_cols = None
        self.hh_eth_type_cols = None
        self.hh_income_cols = None

        self.replicate = 0

    def init(self, init_method='sample'):
        """Initialize the weights of all of the tasks"""
        from tqdm import tqdm

        self.hh_ref = hh_source = pd.read_csv(self.tasks[0].hh_ref, index_col='SERIALNO', low_memory=False) \
            .drop(columns=['geoid'], errors='ignore').astype('Int64')

        self._puma_max_weights = hh_source.iloc[:, 0].reset_index(drop=True).astype(int)

        self._puma_unallocated_weights = self._puma_max_weights.copy()

        for task in tqdm(self.tasks):
            task.init(puma_weights=self._puma_unallocated_weights)

            if init_method == 'sample':
                self.initialize_weights_sample(task)
            if init_method == 'set':
                task.initialize_weights_set_sample()

        t0 = self.tasks[0]  # Just to copy out some internal info.

        self.gq_cols = t0.gq_cols
        self.sex_age_cols = t0.sex_age_cols
        self.hh_size_cols = t0.hh_size_cols
        self.hh_race_type_cols = t0.hh_race_type_cols
        self.hh_eth_type_cols = t0.hh_eth_type_cols

        p = re.compile(r'b19025')
        self.hh_income_cols = [c for c in t0.hh_source.columns if p.match(c)]

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

        return [PumaAllocator(puma_geoid, tasks, cache_dir, state, year, release) for puma_geoid, tasks in
                puma_tasks.items()]

    def initialize_weights_sample(self, task, frac=.7):

        """Initialize the allocated weights proportional to the sample population weights,
        adjusted to the total population. """

        wf = self.weights_frame

        wn1 = wf.remaining / wf.remaining.sum()  # weights normalized to 1

        task.allocated_weights = rand_round(wn1.multiply(task.household_count).values.astype(float))

        task.unallocated_weights -= task.allocated_weights

    def vector_walk(self, N=1200, min_iter=5000, target_error=0.03, step_size_min=1,
                    step_size_max=10, reversal_rate=.3, max_ssm=150, callback=None, memo=None):
        """Run a vector walk on all of the tracts tasks in this puma """

        from itertools import cycle

        rows = []

        ts = time()

        def make_vw(task):
            return iter(task._vector_walk(
                N=N, min_iter=min_iter, target_error=target_error,
                step_size_min=step_size_min, step_size_max=step_size_max,
                reversal_rate=reversal_rate, max_ssm=max_ssm))

        task_iters = [(task, make_vw(task)) for task in self.tasks]

        stopped = set()
        running = set([e[0] for e in task_iters])

        memo['n_stopped'] = len(stopped)
        memo['n_running'] = len(running)
        memo['n_calls'] = 0

        for task, task_iter in cycle(task_iters):

            if task in running:
                try:

                    i, te, min_error, steps_since_min, n_iter = next(task_iter)
                    memo['n_calls'] += 1

                    d = {'i': i, 'time': time() - ts, 'step_size': n_iter, 'error': te,
                         'target_error': target_error,
                         'size': np.sum(task.allocated_weights),
                         'ssm': steps_since_min,
                         'min_error': min_error,
                         'task': task
                         }

                    rows.append(d)

                    if callback and i % 10 == 0:
                        callback(self, task, d, memo)

                except StopIteration:

                    stopped.add(task)
                    running.remove(task)

                    if len(running) == 0:
                        return rows

                    memo['n_stopped'] = len(stopped)
                    memo['n_running'] = len(running)

            callback(self, None, None, memo)

        return rows

    def run(self, *args, callback=None, init_callback=None, memo=None, **kwargs):

        self.init(init_method='sample')

        rows = self.vector_walk(*args, callback=callback, init_cb=init_callback,
                                memo=memo, **kwargs)

        self.save_frame()

        return rows

    def get_task(self, geoid):
        for task in self.tasks:
            if geoid == task.region_geoid:
                return task

        return None

    def tune_puma_allocation(self):
        """Re-run all of the tasks in the puma, trying to reduce the discrepancy
        between the """
        task_iters = [(task, iter(task._vector_walk())) for task in self.tasks]

        for task, task_iter in task_iters:
            try:
                task.cp_prob = self._update_probabilities()
                row = next(task_iter)
                print(task.region_geoid, self.rms_error, self.rms_weight_error, np.sum(task.cp_prob))

            except StopIteration:
                print(task.region_geoid, 'stopped')

    @property
    def weights_frame(self):
        self.weights[
            'allocated'] = self.puma_allocated_weights  # np.sum(np.array([task.allocated_weights for task in self.tasks]), axis=0)
        self.weights['remaining'] = self.weights.pums - self.weights.allocated
        self.weights['dff'] = self.weights.allocated - self.weights.pums
        self.weights['rdff'] = (self.weights.dff / self.weights.pums).fillna(0)
        self.weights['p'] = self.weights.rdff

        return self.weights

    def _update_probabilities(self):
        """Update the running cp_probs, the probabilities for selecting each PUMS
        household from the sample_pop, based on the error in weights for
        the households at the Puma level"""

        w = self.weights_frame

        w['p_pos'] = - w.p.where(w.p < 0, 0)
        w['p_neg'] = w.p.where(w.p > 0, 0)
        self.prob = np.array(w[['p_neg', 'p_pos']].values.flat)
        return self.prob

    @property
    def puma_target_marginals(self):
        from .acs import build_acs

        if self._puma_target_marginals is None:
            _puma_marginals = build_acs(state=self.state, sl='puma', year=self.year, release=self.release)

            cols = self.tasks[
                0].target_marginals.index  # [c for c in _puma_marginals.columns if c.startswith('b') and not c.endswith('_m90')]
            self._puma_target_marginals = _puma_marginals.loc[self.puma_geoid][cols]

        return self._puma_target_marginals

    @property
    def puma_allocated_marginals(self):

        return self.allocated_marginals.sum()

    @property
    def allocated_marginals(self):

        series = {task.region_geoid: task.allocated_marginals for task in self.tasks}

        return pd.DataFrame(series).T

    @property
    def allocated_weights(self):

        series = {task.region_geoid: task.allocated_weights for task in self.tasks}

        return pd.DataFrame(series).T

    @property
    def puma_allocated_weights(self):

        return self.allocated_weights.sum()

    @property
    def target_marginals(self):

        series = {task.region_geoid: task.target_marginals.est for task in self.tasks}

        return pd.DataFrame(series).T

    @property
    def target_errors(self):

        series = {task.region_geoid: task.total_error for task in self.tasks}

        return pd.Series(series)

    @property
    def target_diff(self):

        series = {task.region_geoid: task.target_diff for task in self.tasks}

        return pd.DataFrame(series).T

    @property
    def rms_error(self):
        """RMS error in all of the individual task marginals"""
        t = pd.concat([task.target_diff for task in self.tasks], axis=1).values
        return np.sqrt(np.mean(np.nan_to_num(np.square(t))))

    @property
    def rms_weight_error(self):
        return np.sqrt(np.mean(np.square(self.weights_frame.dff)))

    @property
    def file_name(self):
        return f"{self.state}/{self.year}-{self.release}-{self.replicate}/{self.puma_geoid}.csv"

    @property
    def path(self):
        return Path(self.cache_dir).joinpath(self.file_name)

    def save_frame(self, path=None):

        if path is None:
            path = self.path
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        frames = [task.results_frame for task in self.tasks]
        df = pd.concat(frames)

        df.to_csv(path, index=False)
