"""
Tests for the qopen package.

qopen-runtests [-h] [-v] [-a] [-p] [-d]

-h    short help
-v    be verbose
-a    run all tests
-d    use a permanent tempdir
-d    empty permanent tempdir at start
"""

from pkg_resources import resource_filename
import sys
import unittest

import matplotlib
matplotlib.use('agg')


def run():
    if '-h' in sys.argv[1:]:
        print(__doc__)
        return
    loader = unittest.TestLoader()
    test_dir = resource_filename('qopen', 'tests')
    suite = loader.discover(test_dir)
    runner = unittest.runner.TextTestRunner()
    ret = not runner.run(suite).wasSuccessful()
    sys.exit(ret)
