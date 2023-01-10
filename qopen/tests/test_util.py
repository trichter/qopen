# Copyright 2015-2017 Tom Eulenfeld, MIT license
"""
Tests for util module.
"""

import numpy as np
import unittest
import warnings

from qopen.core import collect_results
from qopen.util import gmean, gerr, smooth, gmeanlist


class TestCase(unittest.TestCase):

    def test_gmean_gerr1(self):
        """Results are same for no weight and equal weights"""
        x = np.array([3, 5, 8, 10, np.nan, 3, 4, -np.inf])
        w = np.ones(len(x))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(gmean(x), gmean(x, weights=w))
            self.assertEqual(gerr(x), gerr(x, weights=w))
            x = x.reshape((2, 4))
            w = np.ones(x.shape)
            self.assertEqual(len(gmean(x, axis=0)), x.shape[1])
            np.testing.assert_allclose(
                gmean(x, axis=0),
                gmean(x, axis=0, weights=w))
            np.testing.assert_allclose(
                gerr(x, axis=0),
                gerr(x, axis=0, weights=w))

    def test_gmean_gerr2(self):
        """Results are same for repeated elements and adapted weights"""
        x1 = np.array([1, 2, 2, 5, 5, 5, 5, 5, np.inf])
        x2 = np.array([1, 2, 5, np.inf])
        w = np.array([1, 2, 5, 10])
        self.assertEqual(gmean(x1), gmean(x2, weights=w))

        # errors can only be compared for biased std
        np.testing.assert_allclose(
            gerr(x1, unbiased=False)[1:],
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
        np.testing.assert_allclose(
            smooth(x, 10, method='zeros')[4:-5],
            smooth(x, 10, method=None))
        np.testing.assert_allclose(
            smooth(x, 11, method='reflect')[5:-5],
            smooth(x, 11, method=None))

    def test_gmean_list_with_nan(self):
        expected = [None, 1.2e-05, 2.4e-06, 1]
        results = {"events": {
            "1": {"g0": [None, 2.4e-05, 2.4e-06, 1]},
            "2": {"g0": [None, 6.0e-06, 2.4e-06, 1]},
            "3": {"g0": [None, None, 2.4e-06, 1]}}}
        collected = collect_results(results, only=['g0'])
        r1 = gmeanlist(collected['g0'], axis=0, robust=False)
        r2 = gmeanlist(collected['g0'], axis=0, robust=True)
        r3 = gmeanlist(collected['g0'], axis=0, robust=True, fall_back=2)
        self.assertIsNone(r1[0])
        self.assertIsNone(r2[0])
        self.assertIsNone(r3[0])
        np.testing.assert_allclose(r1[1:], expected[1:])
        np.testing.assert_allclose(r2[1:], expected[1:])
        np.testing.assert_allclose(r3[1:], expected[1:])


if __name__ == '__main__':
    unittest.main()
