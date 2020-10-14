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
cache_dir = '/Users/eric/proj/data-projects/syntheus/synpums/cache/'

class TestAllocate(unittest.TestCase):

    def test_basic(self):
        tasks = AllocationTask.get_tasks(cache_dir, 'RI', use_tqdm=True, ignore_completed=False)

        self = tasks[150]
        self.init()
        self.initialize_weights_set_sample()

        print(self.total_error)

        self.vector_walk_opt(N=2000, inner_iter_min=1, inner_iter_max=15, init_inc_rate=.1)

        print(self.total_error)


if __name__ == '__main__':
    unittest.main()
