# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = synpums.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import logging
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time

from geoid.censusnames import stusab
from tqdm import tqdm

from synpums import AllocationTask, __version__
from synpums.allocate import _logger as alloc_logger

__author__ = "Eric Busboom"
__copyright__ = "Eric Busboom"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """

    from os import getcwd

    parser = argparse.ArgumentParser(
        description="Generate tract allocations for a state")

    parser.add_argument("-d", "--dir", default=getcwd(), help="Output directory")

    parser.add_argument("-t", "--test", type=int, help="Run a number of tasks, for testing")

    parser.add_argument("-T", "--tasks", action='store_true', help="Only generate tasks")

    parser.add_argument("-P", "--parallel", action='store_true', help="Run states in parallel")

    parser.add_argument("state", help="state abbreviation")

    parser.add_argument(
        "--version",
        action="version",
        version="synpums {ver}".format(ver=__version__))

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)

    parser.add_argument(
        "-D",
        "--debug",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)

    return parser.parse_args(args)


def setup_logging(args):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """

    loglevel = args.loglevel

    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=logging.ERROR, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

    if not loglevel:
        loglevel = logging.ERROR

    _logger.setLevel(loglevel)
    alloc_logger.setLevel(loglevel)


def tqdm_task_callback(task, d, memo):

    memo['pbar'].set_description(
        f"{task.region_geoid} {memo['state']} i={d['i']:3d}  err={d['error']:07.2f}/{d['target_error']:07.2f}"
        f" size={int(d['size']):5d} Task {memo['task_i']} of  {memo['task_n']}.  {memo['task_skip']} skipped. {memo['task_rate']} sec/task "
    )

    memo['last_iter_time'] = time()

    if memo['last_task_time']:
        dt = memo['last_task_time'] - memo['start_time']
        memo['task_rate'] = round(dt / memo['task_i'], 2)


def logging_task_callback(task, d, memo):


    _logger.info(
        f"{task.region_geoid} {memo['state']} i={d['i']:3d}  err={d['error']:07.2f}/{d['target_error']:07.2f}"
        f" size={int(d['size']):5d} Task {memo['task_i']} of  {memo['task_n']}.  {memo['task_skip']} skipped. {memo['task_rate']} sec/task "
    )

    memo['last_iter_time'] = time()

    if memo['last_task_time']:
        dt = memo['last_task_time'] - memo['start_time']
        memo['task_rate'] = round(dt / memo['task_i'], 2)

def run_task(task, cb, memo, state='??', log=None, pbar=None):
    t_start = time()
    task.run_2stage(callback=cb, memo=memo)
    dt = time() - t_start

    if memo is not None:
        memo['last_task_time'] = time()
        memo['task_i'] += 1

    if pbar:
        pbar.update()
    elif log:
        log(f'Allocated {task.region_geoid} state {state} error={task.total_error:07.2f}'
            f' rms_error={task.m90_rms_error:07.2f} dt={dt:3.1f}')

def run_state(state, cache_dir, task_limit, progress='tqdm'):
    """Run a single state"""
    if progress == 'mp':
        log = print
    else:
        log = _logger.info

    log(f'Loading tasks for state {state}')

    tasks = AllocationTask.get_tasks(cache_dir, state, limit=task_limit)
    log(f'Loaded {len(tasks)} for state {state}')
    skipped = 0
    non_exists_tasks = []

    for task in tasks:
        if task.path.exists():
            skipped += 1
        else:
            non_exists_tasks.append(task)

    memo = {'start_time': time(), 'last_iter_time': None, 'task_n': len(tasks),
            'last_task_time': None, 'iters': 0, 'task_i': 0, 'task_skip': skipped, 'task_rate': 0,
            'state': state}

    if progress == 'tqdm':
        pbar = tqdm(total=len(non_exists_tasks))
        cb = tqdm_task_callback
        memo['pbar'] = pbar
    elif progress == 'debug':
        cb = logging_task_callback
        pbar = None
    else:
        cb = None
        pbar = None

    for task in non_exists_tasks:
        run_task(task, cb, memo, state=state, log=log, pbar=pbar)

    if pbar:
        pbar.close()

def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """

    args = parse_args(args)

    setup_logging(args)

    cache_dir = Path(args.dir).resolve().absolute()

    _logger.debug(f"Starting with cache {cache_dir}")

    if args.tasks:
        # This will generate and cache all of the tasks
        tasks = AllocationTask.get_tasks(cache_dir, args.state, limit=args.test, use_tqdm=True)
        print(f"Generated {len(tasks)} tasks")
        return

    if args.state == 'US':
        states = stusab.values()
    else:
        states = [args.state]

    if not args.loglevel:
        progress = 'tqdm'
    elif args.loglevel == logging.DEBUG:
        progress = 'debug'
    else:
        progress = None  # No progress logging, just regular info logging

    if args.parallel:

        n_cpu = int(cpu_count() * .75)

        multi_tasks = [(state, cache_dir, args.test, 'mp') for state in states]

        if len(states) > 1:
            with Pool(n_cpu) as p:
                _logger.info(f'Running {len(states)} states with {n_cpu} processes')
                p.starmap(run_state, multi_tasks)
        else:
            state = states[0]
            non_exists_tasks = [task for task in AllocationTask.get_tasks(cache_dir, state) if not task.path.exists()]

            # run_task(task, cb, memo, state='??', log=None, pbar=None):
            multi_tasks = [(task, None, None, state, _logger.info, None) for task in non_exists_tasks]

            _logger.info(f'Running 1 states with {len(non_exists_tasks)} tasks on {n_cpu} processes')

            with Pool(n_cpu) as p:
                p.starmap(run_task, multi_tasks)


    else:
        for state in states:
            run_state(state, cache_dir, task_limit=args.test, progress=progress)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
