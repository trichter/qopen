"""
Tests for the qopen package.
"""

from pkg_resources import resource_filename
import sys
import unittest

import matplotlib
matplotlib.use('agg')


def run():
    loader = unittest.TestLoader()
    test_dir = resource_filename('qopen', 'tests')
    suite = loader.discover(test_dir)
    runner = unittest.runner.TextTestRunner()
    ret = not runner.run(suite).wasSuccessful()
    sys.exit(ret)
