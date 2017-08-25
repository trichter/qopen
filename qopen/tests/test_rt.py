# Copyright 2015-2017 Tom Eulenfeld, MIT license
"""
Tests for rt module.
"""

import numpy as np
import unittest
from pkg_resources import load_entry_point

from qopen.rt import G, Gcoda_red, Gdirect
from qopen.tests.util import tempdir, quiet


class TestCase(unittest.TestCase):

    def test_Paasschens(self):
        """Test 3 values of figure 2 of Paasschens (1997)"""
        # r, t, c, l, P, dP
        tests = [(2, 3, 1, 1, 0.04 / 4, 5e-4),  # r=2.0l, (3, 0.04)
                 # r=2.8l, (4, 0.03)
                 (2.8 * 2, 4.0, 2, 2, 0.03 / (2.8 * 2) ** 2 / 2, 2e-5),
                 (4, 1, 6, 1, 0.02 / 4 ** 2, 2e-4)]  # r=4.0l, (6, 0.02)
        for r, t, c, l, P, dP in tests:
            # print(r, t, c, l, P, G(r, t, c, 1/l) / FS, dP)
            self.assertLess(abs(G(r, t, c, 1 / l) - P), dP)

    def test_preservation_of_total_energy(self):
        """Volume integral over direct wave and G should be 1"""
        c = 3000
        g0 = 1e-5
        for t in (1, 10, 100):
            r = np.linspace(0, 1.5 * c * t, 1000)
            G_ = G(r, t, c, g0, include_direct=False)
            Gd = 4 * np.pi * c ** 2 * t ** 2 * Gdirect(t, c, g0, var='r')
            G_int = 4 * np.pi * np.sum(r ** 2 * G_) * (r[1] - r[0]) + Gd
            # 2% error are OK for Paaschens solution
#            print(abs(G_int - 1))
            self.assertLess(abs(G_int - 1), 0.02)

    def test_reduced_Sato(self):
        """Test against exact solution in figure 8.4, page 256 of
           Sato, Fehler, Maeda, Second edition (2012) """
        # left side of figure
        self.assertLess(abs(Gcoda_red(1.6, 2.1) - 0.02), 0.002)
        self.assertLess(abs(Gcoda_red(3.2, 3.5) - 0.002), 0.0002)
        # right side of figure
        self.assertLess(abs(Gcoda_red(1, 1.92) - 0.04), 0.004)
        self.assertLess(abs(Gcoda_red(2, 7.68) - 0.004), 0.0004)

    def cmd(self, cmd):
        self.script(cmd.split())

    def test_script(self):
        self.script = load_entry_point('qopen', 'console_scripts', 'qopen-rt')
        with tempdir():
            with quiet():
                self.cmd('calc 1600 500 -t 5 -r 1000')
                self.cmd('calc 1600 500 -t 5 -r 1000 -a 5000')
                self.cmd('calc 1600 500 -t 5 -d')
                self.cmd('plot-t 1600 500 -r 1000')
                self.cmd('plot-t 1600 500 -r 1000 -d')
                self.cmd('plot-r 1600 500 -t 0.5')


if __name__ == '__main__':
    unittest.main()
