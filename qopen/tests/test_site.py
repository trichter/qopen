# Copyright 2015-2016 Tom Eulenfeld, MIT license
"""
Tests for site module.
"""

from copy import deepcopy
import unittest

from qopen.site import align_site_responses


class TestCase(unittest.TestCase):

    def test_align_site_responses(self):
        r = {'E1':
             {'R': {'S1': [0.1, 100], 'S2': [1, 4]},
              'W': [10, 4]},
             'E2':
             {'R': {'S1': [10, 2.0], 'S3': [1, 4]},
              'W': [1, 8]},
             }
        r = {'events': r}
        r2 = align_site_responses(deepcopy(r), use_sparse=True)
        r3 = align_site_responses(deepcopy(r), use_sparse=False)
        s1 = r2['events']['E1']['R']['S1'][0]
        s2 = r2['events']['E2']['R']['S1'][0]
        self.assertLess(abs(s1 - 1), 0.001)
        self.assertLess(abs(s2 - 1), 0.001)

        def _comp(d1, d2):
            if isinstance(d1, float):
                self.assertLess(abs(d1 - d2), 0.001)
            elif isinstance(d1, list):
                _comp(d1[0], d2[0])
                _comp(d1[1], d2[1])
            else:
                for k in d1:
                    _comp(d1[k], d2[k])

        _comp(r2, r3)


if __name__ == '__main__':
    unittest.main()
