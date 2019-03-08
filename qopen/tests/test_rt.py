# Copyright 2015-2017 Tom Eulenfeld, MIT license
"""
Tests for rt module.
"""

import numpy as np
import unittest
from pkg_resources import load_entry_point

from qopen.rt import G, rt3d_coda_reduced
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

    def test_preservation_of_total_energy_3d(self):
        """Volume integral over Green's function should be 1"""
        c = 3000
        g0 = 1e-5
        for t in (1, 10, 100):
            r = np.linspace(0, 1.1 * c * t, 1000)
            G_ = G(r, t, c, g0)
            G_int = 4 * np.pi * np.sum(r ** 2 * G_) * (r[1] - r[0])
            # 2% error are OK for Paaschens solution
            self.assertLess(abs(G_int - 1), 0.02)


    def test_preservation_of_total_energy_2d(self):
        """Area integral over Green's function should be 1"""
        c = 3000
        g0 = 1e-5
        for t in (1, 10, 100):
            r = np.linspace(0, 1.1 * c * t, 1000)
            G_ = G(r, t, c, g0, type='rt2d')
            G_int = 2 * np.pi * np.sum(r * G_) * (r[1] - r[0])
            self.assertLess(abs(G_int - 1), 0.005)


    def test_preservation_of_total_energy_1d(self):
        """Integral over Green's function should be 1"""
        c = 3000
        g0 = 1e-5
        for t in (1, 10, 100):
            r = np.linspace(0, 1.1 * c * t, 1000)
            G_ = G(r, t, c, g0, type='rt1d')
            G_int = 2 * np.sum(G_) * (r[1] - r[0])
            self.assertLess(abs(G_int - 1), 0.005)


    def test_reduced_Sato(self):
        """Test against exact solution in figure 8.4, page 256 of
           Sato, Fehler, Maeda, Second edition (2012) """
        # left side of figure
        self.assertLess(abs(rt3d_coda_reduced(1.6, 2.1) - 0.02), 0.002)
        self.assertLess(abs(rt3d_coda_reduced(3.2, 3.5) - 0.002), 0.0002)
        # right side of figure
        self.assertLess(abs(rt3d_coda_reduced(1, 1.92) - 0.04), 0.004)
        self.assertLess(abs(rt3d_coda_reduced(2, 7.68) - 0.004), 0.0004)

    def cmd(self, cmd):
        self.script(cmd.split())

    def test_script(self):
        self.script = load_entry_point('qopen', 'console_scripts', 'qopen-rt')
        with tempdir():
            with quiet():
                self.cmd('calc 1600 500 -t 5 -r 1000')
                self.cmd('calc 1600 500 -t 5 -r 1000 -a 5000')
                self.cmd('calc-direct 1600 500 -t 5')
                self.cmd('plot-t 1600 500 -r 1000')
                self.cmd('plot-t 1600 500 -r 1000 --no-direct')
                self.cmd('plot-r 1600 500 -t 0.5 --type rt2d')


if __name__ == '__main__':
    unittest.main()
