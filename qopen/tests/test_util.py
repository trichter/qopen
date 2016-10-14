# Copyright 2015-2016 Tom Eulenfeld, MIT license
"""
Tests for util module.
"""

# The following lines are for Py2/Py3 support with the future module.
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (  # analysis:ignore
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)

import numpy as np
import unittest

from qopen.util import gmean, gerr, smooth


class TestCase(unittest.TestCase):

    def test_gmean_gerr1(self):
        """Results are same for no weight and equal weights"""
        x = np.array([3, 5, 8, 10, np.nan, 3, 4, -np.inf])
        w = np.ones(len(x))
        self.assertEqual(gmean(x), gmean(x, weights=w))
        self.assertEqual(gerr(x), gerr(x, weights=w))
        x = x.reshape((2, 4))
        w = np.ones(x.shape)
        self.assertEqual(len(gmean(x, axis=0)), x.shape[1])
        np.testing.assert_array_equal(gmean(x, axis=0),
                                      gmean(x, axis=0, weights=w))
        np.testing.assert_array_equal(gerr(x, axis=0),
                                      gerr(x, axis=0, weights=w))

    def test_gmean_gerr2(self):
        """Results are same for repeated elements and adapted weights"""
        x1 = np.array([1, 2, 2, 5, 5, 5, 5, 5, np.inf])
        x2 = np.array([1, 2, 5, np.inf])
        w = np.array([1, 2, 5, 10])
        self.assertEqual(gmean(x1), gmean(x2, weights=w))

        # errors can only be compared for biased std
        np.testing.assert_array_equal(gerr(x1, unbiased=False)[1:],
                                      gerr(x2, weights=w, unbiased=False)[1:])
        # weighted errors are bigger than unweighted
        np.testing.assert_array_less(gerr(x1)[1:], gerr(x2, weights=w)[1:])
        # biased errors are smaller than unbiased
        np.testing.assert_array_less(gerr(x2, weights=w, unbiased=False)[1:],
                                     gerr(x2, weights=w)[1:])

    def test_gmean_gerr3(self):
        """Difference between biased and unbiased error is small for large N"""
        x = np.linspace(0.5, 1.5, 100)
        w = np.linspace(0.5, 1.5, 100)
        error = (0.01, 0.01)
        gerrwbias = np.array(gerr(x, weights=w, unbiased=False)[1:])
        gerrwunbias = np.array(gerr(x, weights=w, unbiased=True)[1:])
        gerrbias = np.array(gerr(x, unbiased=False)[1:])
        gerrunbias = np.array(gerr(x, unbiased=True)[1:])
        np.testing.assert_array_less(gerrwunbias - gerrwbias, error)
        np.testing.assert_array_less(gerrunbias - gerrbias, error)

    def test_robust(self):
        x = [1, 5, 6, 6, 6, 10000]
        m, err1, err2 = gerr(x, robust=True)
        m_, err1_, err2_ = gerr(x)
        self.assertLess(m, m_)
        self.assertLess(err1, err1_)
        self.assertLess(err2, err2_)

        x = np.reshape([3, 4, 5, 4, 5, 100, 1, 5, 6, 6, 6, 10000], (2, 6))
        m, err1, err2 = gerr(x, axis=1, robust=True)
        m_, err1_, err2_ = gerr(x, axis=1)
        np.testing.assert_array_less(m, m_)
        np.testing.assert_array_less(err1, err1_)
        np.testing.assert_array_less(err2, err2_)

    def test_smooth(self):
        # test window lenght of smoothed signal
        x = np.arange(100)
        self.assertEqual(len(smooth(x, 11, method='zeros')), len(x))
        self.assertEqual(len(smooth(x, 10, method='zeros')), len(x))
        self.assertEqual(len(smooth(x, 11, method='reflect')), len(x))
        self.assertEqual(len(smooth(x, 10, method='reflect')), len(x))
        self.assertEqual(len(smooth(x, 10, method=None)), len(x) - 10 + 1)
        x = np.arange(101)
        self.assertEqual(len(smooth(x, 11, method='zeros')), len(x))
        self.assertEqual(len(smooth(x, 10, method='zeros')), len(x))
        self.assertEqual(len(smooth(x, 11, method='reflect')), len(x))
        self.assertEqual(len(smooth(x, 10, method='reflect')), len(x))
        # test different handling of border effects against each other
        np.testing.assert_array_equal(smooth(x, 10, method='zeros')[4:-5],
                                      smooth(x, 10, method=None))
        np.testing.assert_array_equal(smooth(x, 11, method='reflect')[5:-5],
                                      smooth(x, 11, method=None))


if __name__ == '__main__':
    unittest.main()
