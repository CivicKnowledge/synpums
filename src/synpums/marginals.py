from pathlib import Path

from geoid.acs import Puma
import pandas as pd
from geoid.censusnames import stusab

from .acs import build_acs
from .pums import build_pums_households

hhc_rac_cols = [f"b11001{rc}_001" for rc in list('abcdefg')]
hhc_eth_cols = [f"b11001{rc}_001" for rc in list('hi')]


def agg_inc_marginals(marginals):
    rows = [c for c in marginals.index if c.startswith('b19025')]
    return marginals.loc[rows]


def marginals_sum(marginals):
    return pd.DataFrame({
        'puma': marginals.iloc[:, 0],
        'tracts': marginals.iloc[:, 1:].sum(axis=1)
    })

def marginals_diff(marginals):
    return ((marginals.iloc[:, 1:].sum(axis=1) - marginals.iloc[:, 0]) / marginals.iloc[:, 0])

# Fill in missing values in the aggregate income for the puma

def fill_puma_income_marginals(marginals, cols, total_col='b19025_001'):
    """Fill in missing aggregated income marginals for the puma"""

    missing_count = marginals.loc[cols, 'puma'].isnull().sum()

    if missing_count == 0:
        return

    # Leftover from the total aggregate
    agg_inc_left = marginals.loc[total_col, 'puma'] - marginals.loc[cols, 'puma'].sum()

    if pd.isna(agg_inc_left):
        # There are no value puma marginals. Rare, so punt by just copying over the
        # tracts.
        tracts = list(marginals)[1:]
        marginals.loc[cols, 'puma'] = marginals.loc[cols, tracts].sum(axis=1)
        return

    # Divide the difference over the missing records, then fill them in
    filled_puma = marginals.loc[cols, 'puma'].fillna((agg_inc_left / missing_count).astype('Int64'))

    # Check that the filed series is equal to the total
    if total_col == total_col:
        assert (filled_puma.sum() - marginals.loc[total_col, 'puma']).round(-1) == 0

    marginals.loc[cols, 'puma'] = filled_puma

def fill_tract_income_marginals(marginals, tract_geoid, cols, total_col='b19025_001'):
    """Fill in missing income marginals for a tract, based on the proportion of the
    income each race holds in the puma"""

    tract_agg_income = marginals.loc[cols, tract_geoid]
    agg_income = marginals.loc[total_col, tract_geoid]  # Aggregate income for the tract

    if pd.isna(agg_income):
        return

    race_agg_income = marginals.loc[cols, tract_geoid]
    tract_agg_income.sum() / marginals.loc[total_col, 'puma']

    missing = tract_agg_income.isnull()
    missing_idx = tract_agg_income.index[missing]

    left_over = agg_income - race_agg_income[~missing].sum()  # Tract agg income not accounted for

    # What portion of the missing income should we allocate to each of the
    # missing race entries?
    puma_missing_inc = marginals.loc[missing_idx, 'puma']
    missing_prop = puma_missing_inc / puma_missing_inc.sum()

    try:
        marginals.loc[missing_idx, tract_geoid] = (missing_prop * left_over).round(0).astype('Int64')
    except ValueError:
        # Too many nans, so just punt and fill them in with zeros
        marginals.loc[missing_idx, tract_geoid] = 0

    # Check that the result is close enough. THis only works for the race columns, not the
    # eth columns, although the eth columns will be really close. For the eth columns, nhwites+hispanics
    # will be larger than whites, because hispanics includes some non-whites.

    if total_col == 'b19025_001':
        # print (marginals.loc[cols, tract_geoid].sum(), marginals.loc[total_col, tract_geoid])
        assert (marginals.loc[cols, tract_geoid].sum() - marginals.loc[total_col, tract_geoid]).round(-1) == 0

# marginals = make_marginals_frame(pums_acs, tract_acs, puma_geoid)

def make_marginals_frame(pums_acs, tract_acs, puma_geoid, m90=False, use_puma_geoid=False):
    """Make a single marginals dataframe, which has the marginals for the
    puma and all of the tracts in the puma"""

    pacs = pums_acs.loc[puma_geoid]
    tacs = tract_acs[tract_acs.puma == puma_geoid]

    m_cols = [c for c in tacs.columns if '_m90' in c and c.startswith('b')]
    est_cols = [c for c in tacs.columns if '_m90' not in c and c.startswith('b')]

    if m90:
        cols = m_cols
        inc_rac_cols = [f"b19025{rc}_001_m90" for rc in list('abcdefg')]
        inc_eth_cols = [f"b19025{rc}_001_m90" for rc in list('hi')]
        total_col = 'b19025_001_m90'
        total_col_a = 'b19025a_001_m90'
    else:
        cols = est_cols
        inc_rac_cols = [f"b19025{rc}_001" for rc in list('abcdefg')]
        inc_eth_cols = [f"b19025{rc}_001" for rc in list('hi')]
        total_col = 'b19025_001'
        total_col_a = 'b19025a_001'

    marginals = pacs.loc[cols].to_frame('puma').join(tacs[cols].T)

    try:
        fill_puma_income_marginals(marginals, inc_rac_cols, total_col)
    except Exception:
        #marginals.loc[inc_rac_cols, 'puma'] = marginals.loc[inc_rac_cols, tracts].sum(axis=1)
        raise
    try:
        fill_puma_income_marginals(marginals, inc_eth_cols, total_col_a)
    except Exception as e:
        raise

    for tract_geoid in list(marginals.columns)[1:]:
        fill_tract_income_marginals(marginals, tract_geoid, inc_rac_cols, total_col)
        fill_tract_income_marginals(marginals, tract_geoid, inc_eth_cols, total_col_a)

    if use_puma_geoid:
        marginals = marginals.stack().to_frame('est')
        marginals.index.names = ['marginal', 'region']
        marginals.insert(0, 'puma', puma_geoid)
        marginals.set_index('puma', inplace=True, append=True)
        marginals = marginals.reorder_levels(['puma', 'marginal', 'region'])

    marginals = marginals.rename(index=lambda v: v.replace('_m90',''))

    return marginals

def make_state_marginals(state, year, release):
    pums_acs = build_acs(state, sl='puma', year=year, release=release)
    tract_acs = build_acs(state, sl='tract', year=year, release=release, add_puma=True)

    f = [make_marginals_frame(pums_acs, tract_acs, puma_geoid, use_puma_geoid=True) for puma_geoid in pums_acs.index]
    return pd.DataFrame(pd.concat(f))  # extra DataFrame to convert from CensusDataFrame

def hdf_path(path, year, release):
    return Path(path).joinpath(f'synpums-source-{year}-{release}.hdf')


def write_source_data(path, year, release, cb=None, delete=False):
    """Build the households, puma and tract data for all states abd write them to an HDF file"""
    fn = hdf_path(path, year, release)

    if fn.exists() and delete:
        fn.unlink()

    if not fn.parent.exists():
        fn.parent.mkdir(parents=True)

    if fn.exists():
        s = pd.HDFStore(fn)
        extant = list(s.keys())
        s.close()
    else:
        extant = []

    if cb is None:
        cb = lambda state, group, action: ''

    for state in stusab.values():

        if f'/{state}/households' not in extant:
            cb(state, 'households', 'generating')

            ph = build_pums_households(state, year=2018, release=5)
            ph.to_hdf(fn, complevel=5, key=f'/{state}/households', mode='a' if fn.exists() else 'w')

            cb(state, 'households', 'wrote')
        else:
            cb(state, 'households', 'exists')

        if f'/{state}/pumas' not in extant:
            cb(state, 'puma', 'generating')
            pums_acs = build_acs(state, sl='puma', year=year, release=release, output_type=float)
            pums_acs.to_hdf(fn, complevel=5, key=f'/{state}/pumas', mode='a' if fn.exists() else 'w')

            cb(state, 'puma', 'wrote')
        else:
            cb(state, 'puma', 'exists')

        if f'/{state}/tracts' not in extant:
            cb(state, 'tract', 'generating')
            tract_acs = build_acs(state, sl='tract', year=year, release=release, add_puma=True, output_type=float)
            tract_acs.to_hdf(fn, complevel=5, key=f'/{state}/tracts', mode='a' if fn.exists() else 'w')

            cb(state, 'tract', 'wrote')
        else:
            cb(state, 'tract', 'exists')

    write_puma_list(path, year, release)

    return fn

def write_puma_list(path, year, release):

    fn = hdf_path('.', year, release)

    h = pd.HDFStore(fn)

    rows = []

    for k in h.keys():
        if k.endswith('pumas') and k != '/pumas':
            state, _ = k.strip('/').split('/')
            for puma_geoid in list(h.get(k).index):
                rows.append([state, puma_geoid, k, k.replace('pumas', 'households')])

    df = pd.DataFrame(rows, columns='state geoid puma_key households_key'.split())

    h.put('puma_list', df)

    h.close()

def load_households(path, state, year, release):
    return pd.read_hdf(hdf_path(path, year, release), f'/{state}/households')

def load_marginals(path, state_or_puma, year, release):
    fn = hdf_path(path, year, release)

    try:
        puma_geoid = Puma.parse(state_or_puma)
        state = puma_geoid.stusab
    except ValueError:
        state = state_or_puma
        puma_geoid = None

    pums_acs = pd.read_hdf(fn, f'/{state}/pumas')
    tract_acs = pd.read_hdf(fn, f'/{state}/tracts')

    if puma_geoid:
        return make_marginals_frame(pums_acs, tract_acs, puma_geoid, use_puma_geoid=False)
    else:

        f = [make_marginals_frame(pums_acs, tract_acs, puma_geoid, use_puma_geoid=True)
             for puma_geoid in pums_acs.index]
        return pd.DataFrame(pd.concat(f))  # extra DataFrame to convert from CensusDataFrame


def load_allocation_data(path, state_or_puma, year, release):
    fn = hdf_path(path, year, release)

    try:
        puma_geoid = Puma.parse(state_or_puma)
        state = puma_geoid.stusab
    except ValueError:
        state = state_or_puma
        puma_geoid = None

    pums_acs = pd.read_hdf(fn, f'/{state}/pumas')
    tract_acs = pd.read_hdf(fn, f'/{state}/tracts')


    if puma_geoid:
        households = load_households(path, state, year, release)
        households = households[households.geoid==puma_geoid].drop(columns=['geoid'])
        marg = make_marginals_frame(pums_acs, tract_acs, puma_geoid, use_puma_geoid=False)
        m90 = make_marginals_frame(pums_acs, tract_acs, puma_geoid, m90=True, use_puma_geoid=False)
    else:
        households = load_households(path, state, year, release)
        f = [make_marginals_frame(pums_acs, tract_acs, puma_geoid, use_puma_geoid=True)
             for puma_geoid in pums_acs.index]
        marg = pd.DataFrame(pd.concat(f))  # extra DataFrame to convert from CensusDataFrame
        m90 = None

    return households, marg, m90,  list(marg.columns[1:])