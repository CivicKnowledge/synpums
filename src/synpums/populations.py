# Copyright (c) 2020 Civic Knowledge. This file is licensed under the terms of the
# MIT license included in this distribution as LICENSE

import logging
from pathlib import Path

import pandas as pd
import rowgenerators as rg
from geoid.acs import AcsGeoid, County, Puma, State, Tract

from synpums import build_pums_households

from .acs import stusab

_logger = logging.getLogger(__name__)

class StateFactory(object):
    def __init__(self,  cache_dir):
        self.cache_dir = Path(cache_dir)

    @property
    def available_states(self):
        return [e.name for e in self.cache_dir.joinpath('tract').iterdir()]

    def get_state(self, state, year=2018, release=5):
        return StateTracts(state, cache_dir=self.cache_dir, year=year, release=release)

class StateTracts(object):

    def __init__(self,  state, cache_dir, year=2018, release=5 ):

        self.cache_dir = Path(cache_dir)
        self.year = year
        self.release = release

        try:
            int(state)
            self.state = stusab[state]
        except ValueError:
            self.state = state
            assert self.state in stusab.values(), f'Unknown state abbreviation {st}'

    @property
    def tract_files(self):

        tracts = {}

        for f in self.cache_dir.glob(f'**/tract/{self.state}/**/*.csv'):
            tract = AcsGeoid.parse(f.stem)
            tracts[tract] = f

        return tracts

    @property
    def weights_frame(self):
        """Return a Pandas dataframe with one row for each
        combination of tract and PUMS serialno and the weight for that
        combination"""

        frames = []
        for tract, f in self.tract_files.items():
            df = pd.read_csv(f)
            df.insert(0, 'county', tract.county)
            df.insert(0, 'stusab', tract.stusab)
            df.insert(0, 'tract', str(tract))

            df['weight'] = df.weight.fillna(0).astype(int)

            frames.append(df)

        return pd.concat(frames)

    @property
    def allocation_frame(self):
        """Return a dataframe with the weights merged to the allocation frame,
        which has one record per hold, with  columns for census variable names. This
        is the form of the records that are used in the allocation algorithms. """

        pums = build_pums_households(self.state, year=2018, release=5)

        return self.weights_frame.merge(pums, left_on='serialno', right_index=True)

    @property
    def household_frame(self):
        """Weights merged to the PUMS household records"""

        _logger.debug(f"Load household file pums:{self.state}/h/{self.year}/{self.release}")
        return rg.dataframe(f'pums:{self.state}/h/{self.year}/{self.release}', low_memory=False).set_index('SERIALNO')

    @property
    def allocated_household_frame(self):
        """Weights merged to the PUMS household records"""

        return self.weights_frame.merge(self.household_frame.reset_index(), left_on='serialno', right_on='SERIALNO')

    @property
    def person_frame(self):

        _logger.debug(f"Load person file pums:{self.state}/h/{self.year}/{self.release}")
        return rg.dataframe(f'pums:{self.state}/p/{self.year}/{self.release}', low_memory=False)\
            .set_index(['SERIALNO', 'SPORDER'])


    @property
    def allocated_person_frame(self):

        return self.weights_frame.merge(self.person_frame.reset_index(), left_on='serialno', right_on='SERIALNO')


    def write(df, state, cache):
        p = Path(cache).joinpath(f'state/{state}.csv.gz')

        p.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(p, compression='infer')
