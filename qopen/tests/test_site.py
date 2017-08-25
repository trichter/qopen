# Copyright 2015-2017 Tom Eulenfeld, MIT license
"""
Tests for site module.
"""

from copy import deepcopy
import json
from pkg_resources import resource_filename
import unittest

import numpy as np

from qopen.site import align_site_responses, _Rmean, _Rstd, _collectR


class TestCase(unittest.TestCase):

    def test_align_site_responses(self):
        eps = 1e-7
        r = {'E1': {'R': {'S1': [0.1, 100], 'S2': [1, 4]}, 'W': [10, 4]},
             'E2': {'R': {'S1': [10, 2.0], 'S3': [1, 4]}, 'W': [1, 8]}}
        r = {'events': r}
        r2 = align_site_responses(deepcopy(r), use_sparse=True)
        r3 = align_site_responses(deepcopy(r), use_sparse=False)
        s1 = r2['events']['E1']['R']['S1'][0]
        s2 = r2['events']['E2']['R']['S1'][0]
        self.assertLess(abs(s1 - 1), eps)
        self.assertLess(abs(s2 - 1), eps)

        def _comp(d1, d2):
            if isinstance(d1, float):
                self.assertLess(abs(d1 - d2), eps)
            elif isinstance(d1, list):
                _comp(d1[0], d2[0])
                _comp(d1[1], d2[1])
            else:
                for k in d1:
                    _comp(d1[k], d2[k])

        _comp(r2, r3)

    def test_align_site_responses_offset(self):
        eps = 1e-7
        r = {'E1': {'R': {'S1': [4], 'S2': [4]}, 'W': [1]},
             'E2': {'R': {'S1': [8], 'S2': [8]}, 'W': [2]}}
        r = {'events': r}
        std1 = _Rstd(_collectR(r))
        r2 = align_site_responses(r)
        std2 = _Rstd(_collectR(r2))

        s1 = r2['events']['E1']['R']['S1'][0]
        s2 = r2['events']['E2']['R']['S1'][0]
        w1 = r2['events']['E1']['W'][0]
        w2 = r2['events']['E2']['W'][0]

        self.assertLess(abs(s1 - 1), eps)
        self.assertLess(abs(s2 - 1), eps)
        self.assertLess(abs(w1 - 4), eps)
        self.assertLess(abs(w2 - 16), eps)
        self.assertLess(std2, std1)

    def test_align_site_responses_mean(self):
        eps = 1e-7
        r = {'E1': {'R': {'S1': [4], 'S2': [8]}, 'W': [1]},
             'E2': {'R': {'S1': [8], 'S2': [1000]}, 'W': [2]}}
        r = {'events': r}
        std1 = _Rstd(_collectR(r))
        align_site_responses(r)
        std2 = _Rstd(_collectR(r))

        s1 = r['events']['E1']['R']['S1'][0]
        s2 = r['events']['E2']['R']['S1'][0]
        s3 = r['events']['E1']['R']['S2'][0]
        s4 = r['events']['E2']['R']['S2'][0]

        self.assertLess(abs(s1 * s2 * s3 * s4 - 1), eps)
        self.assertLess(std2, std1)

    def test_align_site_responses_only_largest_area(self):
        eps = 1e-7
        r = {'E1': {'R': {'S1': [4], 'S2': [6], 'S3': [500],
                          'S0': [None]}, 'W': [1]},
             'E2': {'R': {'S1': [8], 'S2': [50], 'S4': [3]}, 'W': [2]},
             'E3': {'R': {'S1': [None], 'S2': [100], 'S3': [10000]}, 'W': [4]},
             'E4': {'R': {'S5': [50000]}, 'W': [8]},
             'E5': {'R': {'S6': [5000], 'S7': [None]}, 'W': [1]},
             'E6': {'R': {'S7': [1], 'S8': [None]}, 'W': [1]},
             'E7': {'R': {'S9': [1], 'S10': [None]}, 'W': [1]},
             }
        r = {'events': r}
        rsp = 100
        std1 = _Rstd(_collectR(r))
        r2 = align_site_responses(deepcopy(r), response=rsp, station='S1')
        std2 = _Rstd(_collectR(r2))
        s11 = r2['events']['E1']['R']['S1'][0]
        s12 = r2['events']['E2']['R']['S1'][0]
        self.assertLess(abs((s11 * s12) ** 0.5 - rsp), eps)
        self.assertLess(std2, std1)
        rsp = 0.001
        align_site_responses(r, response=rsp)
        std3 = _Rstd(_collectR(r))
        s11 = r['events']['E1']['R']['S1'][0]
        s12 = r['events']['E2']['R']['S1'][0]
        s13 = r['events']['E3']['R']['S1'][0]
        s21 = r['events']['E1']['R']['S2'][0]
        s22 = r['events']['E2']['R']['S2'][0]
        s23 = r['events']['E3']['R']['S2'][0]
        s31 = r['events']['E1']['R']['S3'][0]
        s33 = r['events']['E3']['R']['S3'][0]
        s4 = r['events']['E2']['R']['S4'][0]
        s5 = r['events']['E4']['R']['S5'][0]
        w4 = r['events']['E4']['W'][0]
        s = ((s11 * s12) ** 0.5 * (s21 * s22 * s23) ** (1. / 3) *
             (s31 * s33) ** 0.5 * s4) ** 0.25
        Rm = _Rmean(_collectR(r))
        self.assertLess(abs(Rm - np.log(s)), eps)
        self.assertLess(abs(s - rsp), eps)
        self.assertLess(abs(Rm - np.log(rsp)), eps)
        self.assertEqual(s13, None)
        self.assertEqual(s5, None)
        self.assertEqual(w4, None)
        self.assertLess(std3, std1)
        self.assertLess(abs(std2 - std3), eps)

    def test_align_site_responses_large_dataset_usarray(self):
        eps = 1e-5
        fname = resource_filename('qopen', 'tests/data/usarray_dataset.json')
        with open(fname) as f:
            r = json.load(f)
        sta = 'TA.B12A'
        rsp = 10
        std1 = _Rstd(_collectR(r))
        r2 = align_site_responses(deepcopy(r), station=sta, response=rsp)
        std2 = _Rstd(_collectR(r2))
        Rm = _Rmean(_collectR(r2, only={sta}))
        self.assertLess(abs(Rm - np.log(rsp)), eps)
        rsp = 100
        align_site_responses(r, response=rsp)
        std3 = _Rstd(_collectR(r))
        self.assertLess(abs(_Rmean(_collectR(r)) - np.log(rsp)), eps)
        self.assertLess(std2, std1)
        self.assertLess(std3, std1)
        self.assertLess(abs(std2 - std3), eps)


if __name__ == '__main__':
    unittest.main()
