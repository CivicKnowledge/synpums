
import unittest
import warnings

import pandas as pd
import rowgenerators as rg

from synpums import *
from synpums.util import *

warnings.filterwarnings("ignore")


state = 'RI'
year = 2018
release = 5

def fetch(url):
    return rg.dataframe(url).drop(columns=['stusab', 'county', 'name'])

class TestACSIncome(unittest.TestCase):

    def x_test_median_incomes(self):
        """Check that the summary stats for the aggregate income of puma,
         roughtly matches between the PUMS and ACS datasets. Both values are
         divided by the median household income of the """

        pums_acs = build_acs(state, sl='puma', year=year, release=release)
        dfp, dfh = build_pums_dfp_dfh(state, year=2018, release=5)

        puma_geoid = '79500US4400104'

        dfh_g = dfh[dfh.geoid == puma_geoid]

        pums = pd.get_dummies(dfh_g['b19025']) \
            .multiply(dfh_g['HINCP'] / dfh_g['HINCP'].median(), axis=0) \
            .multiply(dfh_g['ADJINC'] / 1e6, axis=0) \
            .multiply(dfh_g['WGTP'], axis=0) \
            .fillna(0).sum()

        acs = (pums_acs[list(pums.index)].loc[puma_geoid] / pums_acs['b19013_001'].loc[puma_geoid]).astype(float)

        t = pums.to_frame('pums').join(acs.to_frame('acs')).dropna()

        t['r'] = t.pums / t.acs

        print(t.head())

        self.assertTrue(all(t.r.between(.89, 1.11)))

    def test_acs_incomes_2(self):
        ph = build_pums_households(state, year=2018, release=5)
        pums_acs = build_acs(state, sl='puma', year=year, release=release)

        cols = [c for c in ph.columns if c.startswith('b19025') and c in pums_acs.columns]

        acs_sums = pums_acs[cols].fillna(0)
        pums_sums = ph[cols].multiply(ph.WGTP, axis=0).groupby(ph.geoid).sum()

        t = pums_sums.divide(acs_sums, axis=0).replace({np.inf: 1}).fillna(1)
        self.assertEqual(t.stack().mean().round(1),1)


if __name__ == '__main__':
    unittest.main()
