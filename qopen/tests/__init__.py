"""
Tests for the qopen package.

qopen-runtests [-h] [-v] [-a] [-p] [-d] [-f] [-n num]

-h    short help
-v    be verbose
-a    run all tests
-p    use a permanent tempdir
-d    empty permanent tempdir at start
-f    fail fast
-n num   maximal number of cores to use (default: all)
"""

from importlib import resources
import sys
import unittest


def run():
    args = sys.argv[1:]
    if '-h' in args:
        print(__doc__)
        return
    loader = unittest.TestLoader()
    test_pkg = resources.files('qopen') / 'tests'
    with resources.as_file(test_pkg) as test_dir:
        suite = loader.discover(str(test_dir))
    runner = unittest.runner.TextTestRunner(failfast='-f' in args)
    ret = not runner.run(suite).wasSuccessful()
    sys.exit(ret)
