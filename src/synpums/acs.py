# Copyright (c) 2020 Civic Knowledge. This file is licensed under the terms of the
# MIT license included in this distribution as LICENSE

import pandas as pd
import rowgenerators as rg
from geoid.censusnames import stusab

race_codes = [  # Census iterate code, local name, PUMS codes
    ('A', 'white', 1),
    ('B', 'black', 2),
    ('C', 'aian', (3, 4, 5)),
    ('D', 'asian', 6),
    ('E', 'nhopi', 7),
    ('F', 'other', 8),
    ('G', 'many', 9),
    ('H', 'nhwhite', None),
    ('I', 'hisp', None)
]

race_map = {
    'A': 'white',
    'B': 'black',
    'C': 'aian',
    'D': 'asian',
    'E': 'nhopi',
    'F': 'other',
    'G': 'many',
    'H': 'nhwhite',
    'I': 'hisp'
}

def make_puma_tract_map():
    # To build the tract map from the download
    import geoid.acs

    tp_rel = rg.dataframe('https://www2.census.gov/geo/docs/maps-data/data/rel/2010_Census_Tract_to_2010_PUMA.txt#target_format=csv') # PUMA <-> Tracts relationships

    tp_rel['tract'] = tp_rel.apply(lambda r: str(geoid.acs.Tract(r.STATEFP, r.COUNTYFP, r.TRACTCE)), axis=1)
    tp_rel['puma'] = tp_rel.apply(lambda r: str(geoid.acs.Puma(r.STATEFP, r.PUMA5CE)), axis=1)

    return tp_rel[['puma', 'tract']].set_index('tract')


def puma_tract_df():  # To actually use the tract map
    import synpums.data as sd
    from pathlib import Path
    return pd.read_csv(Path(sd.__file__).parent.joinpath('puma_tract_map.csv'))

def puma_tract_map(): # To actually use the tract map
    import synpums.data as sd
    from pathlib import Path

    return { e['tract']:e['puma'] for k, e in puma_tract_df().to_dict(orient='index').items()}


def get_table_raceiter(table, state, sl, year=2018, release=5):
    """Get all of the race iterations for a censys table for a given summary level.  """

    acs_frames = [rg.dataframe(f'census://{year}/{release}/{state}/{sl}/{table.upper()}{race}').drop(
        columns=['stusab', 'county', 'name'])
        for race in ['']+list(race_map.keys())]

    return acs_frames[0].join(acs_frames[1:])


def make_age_sex_map(b01001):
    """Make a map from age and sex to columns in Census table b01001

    Generally, you'll want to use the compiled version in age_sex.py
    """

    t=b01001.stacked(add_dimensions=True)

    t = t[['column','sex','min_age','max_age']].drop_duplicates()

    as_map = {}

    for idx, r in t.iterrows():
        for age in range(r.min_age, r.max_age+1):
            as_map[(r.sex, age)] =  r.column

    return as_map

def build_acs(state, sl='puma', year=2018, release=5):

    b11001 = get_table_raceiter('B11001', state, sl) # Household Type

    def fetch(url):
        return rg.dataframe(url).drop(columns=['stusab', 'county', 'name'])

    b11016 = fetch(f'census://{year}/{release}/{state}/{sl}/B11016') #Household Type by Household Size

    b01001 = fetch(f'census://{year}/{release}/{state}/{sl}/B01001')  # Sex by Age

    b01003 = fetch(f'census://{year}/{release}/{state}/{sl}/B01003') # Total population

    b26001 = fetch(f'census://{year}/{release}/{state}/{sl}/B26001')  # Group Quarters population

    b19025 = get_table_raceiter('B19025', state, sl)   # B19025: Aggregate Household Income
    b19025 = b19025.apply(pd.to_numeric, errors='ignore')/20_000 # Scale to be similar to other vars
    b19025 = b19025.apply(round).astype('Int64')

    b19214 = fetch(f'census://{year}/{release}/{state}/{sl}/B19214')  # B19214: Aggregate Nonfamily Household Income

    acs = b11001.join([b11016,b01001,b01003,b26001, b19025]).sort_index().sort_index(axis=1)

    if sl == 'puma':
        acs = acs.join(puma_tract_df())

    return acs.apply(pd.to_numeric, errors='ignore').astype('Int64', errors='ignore')
