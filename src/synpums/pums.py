# Copyright (c) 2020 Civic Knowledge. This file is licensed under the terms of the
# MIT license included in this distribution as LICENSE

import logging

import numpy as np
import pandas as pd
import rowgenerators as rg
from geoid.acs import Puma

_logger = logging.getLogger(__name__)

hht_fam_map = {
    np.nan: 'nothh',  # '.N/A (GQ/vacant)',
    1: 'fam',  # '.Married couple household',
    2: 'fam',  # '.Other family household: Male householder, no spouse present',
    3: 'fam',  # '.Other family household: Female householder, no spouse present',
    4: 'nonfam',  # '.Nonfamily household: Male householder: Living alone',
    5: 'nonfam',  # '.Nonfamily household: Male householder: Not living alone',
    6: 'nonfam',  # '.Nonfamily household: Female householder: Living alone',
    7: 'nonfam',  # '.Nonfamily household: Female householder: Not living alone'
}

# Map from PUMS HHT codes to census column valules.
hh_size_map = {
    # (None, 'fam'):   'B11016_002',  # Households  - Total  - Family households
    (2, 'fam'): 'b11016_003',  # Households  - Total  - Family households  - 2 - person household
    (3, 'fam'): 'b11016_004',  # Households  - Total  - Family households  - 3 - person household
    (4, 'fam'): 'b11016_005',  # Households  - Total  - Family households  - 4 - person household
    (5, 'fam'): 'b11016_006',  # Households  - Total  - Family households  - 5 - person household
    (6, 'fam'): 'b11016_007',  # Households  - Total  - Family households  - 6 - person household
    (7, 'fam'): 'b11016_008',  # Households  - Total  - Family households  - 7 - or - more person household
    # (None, 'nonfam'):'b11016_009',  # Households  - Total  - Nonfamily households
    (1, 'nonfam'): 'b11016_010',  # Households  - Total  - Nonfamily households  - 1 - person household
    (2, 'nonfam'): 'b11016_011',  # Households  - Total  - Nonfamily households  - 2 - person household
    (3, 'nonfam'): 'b11016_012',  # Households  - Total  - Nonfamily households  - 3 - person household
    (4, 'nonfam'): 'b11016_013',  # Households  - Total  - Nonfamily households  - 4 - person household
    (5, 'nonfam'): 'b11016_014',  # Households  - Total  - Nonfamily households  - 5 - person household
    (6, 'nonfam'): 'b11016_015',  # Households  - Total  - Nonfamily households  - 6 - person household
    (7, 'nonfam'): 'b11016_016',  # Households  - Total  - Nonfamily households  - 7 - or - more person household
}


def assign_B11016(r):
    if not r.is_hh:
        return None

    np = r.NP if r.NP < 7 else 7

    return hh_size_map.get((np, r.hhtype))


# Map from  PUMS HHT values to column numbers in Census table B11001
hht_census_map = {
    # nan: '.N/A (GQ/vacant)',
    1: '003',  # '.Married couple household',
    2: '005',  # '.Other family household: Male householder, no spouse present',
    3: '006',  # '.Other family household: Female householder, no spouse present',
    4: '008',  # '.Nonfamily household: Male householder: Living alone',
    5: '009',  # '.Nonfamily household: Male householder: Not living alone',
    6: '008',  # '.Nonfamily household: Female householder: Living alone',
    7: '009',  # '.Nonfamily household: Female householder: Not living alone'
}

# From PUMS race codes to Census race iterations.
pums_census_race_map = {
    1: 'a',  # '.White alone',
    2: 'b',  # '.Black or African American alone',
    3: 'c',  # '.American Indian alone',
    4: 'c',  # '.Alaska Native alone',
    5: 'c',
    # '.American Indian and Alaska Native tribes specified; or American Indian or Alaska Native, not specified and no other races',
    6: 'd',  # '.Asian alone',
    7: 'e',  # '.Native Hawaiian and Other Pacific Islander alone',
    8: 'f',  # '.Some Other Race alone',
    9: 'g'  # '.Two or More Races'}
}


def race_iter_h(r):
    """Return a census race iteration code for a PUMS records, for just
    the Non-hispanic White / Hispanic distinction"""
    is_hisp = r.HISP != 1

    if r.RAC1P == 1 and not is_hisp:
        return 'h'  # non hispanic white
    elif is_hisp:
        return 'i'
    else:
        return None


def get_pums_age_sex(dfp):
    """
    Convert PUMS person records into age/sex census columns per household.

    Args:
        dfp: PUMS person records.
        as_map: Age/Sex map, from make_age_sex_map()

    Returns:

    """
    from .age_sex import age_sex_map
    from collections import defaultdict

    rows = defaultdict(dict)

    t = dfp[['SERIALNO', 'SEX', 'AGEP']].copy()

    t['SEX'] = t.SEX.replace({1: 'male', 2: 'female'})

    for idx, r in t.iterrows():
        sn = r.SERIALNO
        c = age_sex_map.get((r.SEX, r.AGEP))
        if c:
            rows[sn][c] = rows[sn].get(c, 0) + 1

        rows[sn]['b01003_001'] = rows[sn].get('b01003_001', 0) + 1  # Total Population

    t = pd.DataFrame.from_dict(rows, orient='index').fillna(0)
    return t[sorted(t.columns)]


def build_pums_dfp_dfh(state, year=2018, release=5, replicate=0):
    """Download the PUMS files for persons and houlds and convert to a
    dummy variable format with census column ids for column names. """

    # Harmonize the house holder's race with the census race iterations

    _logger.debug(f"Load housing file pums:{state}/h/{year}/{release}")
    dfh = rg.dataframe(f'pums:{state}/h/{year}/{release}', low_memory=False)

    dfh = dfh[dfh.NP > 0]  # Remove records for vacant housing

    _logger.debug(f"Load person file pums:{state}/h/{year}/{release}")
    dfp = rg.dataframe(f'pums:{state}/p/{year}/{release}', low_memory=False)

    drop_cols = ['ADJINC', 'DIVISION', 'PUMA', 'REGION', 'RT', 'ST']

    # Merge the households file with the reference person for each household.
    dfh = dfh.merge(dfp[dfp.SPORDER == 1].drop(columns=drop_cols), on='SERIALNO', how='left').set_index('SERIALNO')

    # Add a puma geoid
    dfh['geoid'] = dfh.apply(lambda r: str(Puma(r.ST, r.PUMA)), axis=1)

    dfh['race_iter'] = dfh.RAC1P.apply(lambda v: pums_census_race_map.get(v))

    dfh['race_iter_h'] = dfh.apply(race_iter_h, axis=1)

    # Household type, by race
    dfh['b11001'] = dfh.apply(
        lambda r: f"b11001{r.race_iter}_{hht_census_map[r.HHT]}" if r.HHT in hht_census_map else None, axis=1)

    dfh['b19025'] = dfh.apply(
        lambda r: f"b19025{r.race_iter}_001" if r.HHT in hht_census_map else None, axis=1)

    # Household type, hispanic
    dfh['b11001_h'] = dfh.apply(lambda r: f"b11001{r.race_iter_h}_{hht_census_map[r.HHT]}"
    if (r.HHT in hht_census_map and r.race_iter_h is not None) else None, axis=1)

    dfh['hhtype'] = dfh.HHT.map(hht_fam_map)
    dfh['is_hh'] = dfh.hhtype.isin(['fam', 'nonfam'])

    # Household type by household size
    dfh['b11016'] = dfh.apply(assign_B11016, axis=1)





    return dfp, dfh


def _full_pums_weights(dfh, dfp):
    """Create a single file with all of the replicate weights for person and households,
    for each record, person and replicate"""

    wgtp = [c for c in dfh.columns if c.startswith('WGTP')]
    pwgtp = [c for c in dfp.columns if c.startswith('PWGTP')]

    t = dfh[wgtp].rename(columns=dict(zip(wgtp, range(len(wgtp))))).copy().stack().to_frame('wgtp')
    t.insert(0, 'SPORDER', 1)
    t = t.set_index('SPORDER', append=True)
    t.index.names = ['SERIALNO', 'replicate', 'SPORDER']
    t = t.reorder_levels(['SERIALNO', 'SPORDER', 'replicate'])

    tp = dfp.set_index(['SERIALNO', 'SPORDER'])[pwgtp].rename(
        columns=dict(zip(pwgtp, range(len(pwgtp))))).copy().stack().to_frame('pwgtp')
    tp.index.names = ['SERIALNO', 'SPORDER', 'replicate']

    return t.join(tp, how='outer').astype('Int64')


def pums_weights(state, year, release):
    dfp, dfh = build_pums_dfp_dfh(state, year, release)
    return _full_pums_weights(dfh, dfp)

def build_pums_households(state, year=2018, release=5, replicate=0):

    dfp, dfh = build_pums_dfp_dfh(state, year, release, replicate)

    # Add in table b01001, age by set
    a_s = get_pums_age_sex(dfp)

    t = pd.get_dummies(dfh[['b11001', 'b11001_h', 'b11016']], prefix='', prefix_sep='')

    # The household income dummies need more processing:
    # * Divide by 20,000 to compress the range
    # * Multiply by  the inflation adjustment factor ( 6 digit fixed point integer )

    pums = pd.get_dummies(dfh['b19025']).multiply(((dfh.HINCP * dfh.ADJINC / 1e6) / 20_000).round(0), axis=0).fillna(0)

    t = t.join(pums)

    t.insert(0, 'b11001_001', (~dfh.RELP.isin([16, 17])).astype(int))  # Total households
    t.insert(1, 'b26001_001', dfh.RELP.isin([16, 17]).astype(int))  # Group quarters

    # For group quarters, the weight on the housing record will be 0,
    # but the "household" contains only one person, so we just use the person weight
    wgtp = [c for c in dfh.columns if c.startswith('WGTP')]
    pwgtp = [c for c in dfp.columns if c.startswith('PWGTP')]

    dfh.loc[dfh[f'WGTP'] == 0, wgtp] = dfh.loc[:, pwgtp]

    t = dfh[['geoid']+wgtp].join(t, how='left', ).join(a_s, how='left').fillna(0)

    cols = [c for c in t.columns if c not in wgtp[1:]] + wgtp[1:]

    return t[cols]


# Maybe deprecated?

def weight_comp_df(df):
    """Generate a dataframe that has the WGTP column from a households dataset, along with
    the calucalted weights from the households assigned to a region. The calculated weights are
    the counts of the number of times that each household, by sequence number, appears in the region dataset
    """

    raise DeprecationWarning()

    t = df.groupby(df.index).count().WGTP.to_frame('WGTP_by_count')

    return df[['WGTP']].join(t).reset_index().drop_duplicates()


def region_households(df):
    """Convert a region file -- with one record for each household -- into a households
    source file, which has each household once, with a weight. This is the same file format as from
    build_pums_households(), but can be built for any smaller region. """

    wc = weight_comp_df(df)[['index', 'WGTP_by_count']].set_index('index').rename(columns={'WGTP_by_count': 'WGTP'})

    t = df \
        .groupby(df.index) \
        .first() \
        .drop(columns=['WGTP']) \
        .join(wc)

    return t

    cols = ['geoid', 'WGTP'] + list(t.columns[:-2])

    return t[cols]
