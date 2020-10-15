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
cache_dir = '/tmp/synpums'

class TestAllocate(unittest.TestCase):

    def test_basic(self):

        tasks = AllocationTask.get_tasks(cache_dir, 'RI', ignore_completed=False)

        task = tasks[24]
        task.init()
        print(task.m90_rms_error)
        task.initialize_weights_sample()

        print(f"te={task.total_error}, rms={task.m90_rms_error}")
        args = dict(N=2000, min_iter=1000, step_size_max=15, step_size_min=1, reversal_rate=.4, max_ssm=150)

        rows = task.vector_walk(**args)

        print(f"te={task.total_error}, rms={task.m90_rms_error}")


if __name__ == '__main__':
    unittest.main()
