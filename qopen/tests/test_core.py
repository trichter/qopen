# Copyright 2015-2016 Tom Eulenfeld, MIT license
"""
Tests for core module.
"""

# The following lines are for Py2/Py3 support with the future module.
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (  # analysis:ignore
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)

from glob import glob
import os
from pkg_resources import load_entry_point
import unittest

import numpy as np
from qopen.core import init_data, run, run_cmdline
from qopen.tests.util import tempdir, quiet


TRAVIS = os.environ.get('TRAVIS') == 'true'
PLOT = False


class TestCase(unittest.TestCase):

    def test_entry_point(self):
        script = load_entry_point('qopen', 'console_scripts', 'qopen')
        with quiet():
            try:
                script(['-h'])
            except SystemExit:
                pass

    @unittest.skipIf(not TRAVIS, 'save time')
    def test_cmdline(self):
        parallel = os.environ.get('PARALLEL', 'true') == 'true'
        script = run_cmdline
        msg = ('Only %d plot files (%s) are created.\n\n'
               'Created files are:\n%s\n\n'
               '%s')
        with tempdir(delete=True):
            script(['--create-config', '--tutorial'])
            args = [] if parallel else ['--no-parallel']
            script(args)
            # check if pictures were created
            if os.path.exists('example.log'):
                with open('example.log') as flog:
                    log = 'Content of log file:\n' + flog.read()
            else:
                log = 'Log file does not exist.'
            files = list(glob('plots/*.png'))
            msg2 = msg % (len(files), 'png', files, log)
            self.assertEqual(len(files), 85, msg=msg2)
            files = list(glob('plots/*.pdf'))
            msg2 = msg % (len(files), 'pdf', files, log)
            self.assertEqual(len(files), 4, msg=msg2)

    def test_results_of_tutorial(self):
        """Test against publication of Sens-Schoenfelder and Wegler (2006)"""
        plot = PLOT
        freq = [0.1875, 0.375, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0]  # page 1365
        g0 = [2e-6, 2e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1.5e-6, 2e-6]  # fig 4
        Qi = [2e-3, 2e-3, 1.8e-3, 2e-3, 1.5e-3, 1e-3, 5e-4, 2e-4]  # fig 5
        freq = np.array(freq)
        g0 = np.array(g0)
        b = np.array(Qi) * (2 * np.pi * np.array(freq))
        M0 = {'20010623': 5.4e14, '20020722': 4.1e15, '20030222': 1.5e16,
              '20030322': 8.8e14, '20041205': 6.8e15}  # table 1
        kwargs = {
            'plot_energies': plot, 'plot_optimization': plot,
            'plot_fits': plot, 'plot_eventresult': plot,
            'plot_eventsites': plot, 'plot_results': plot,
            'plot_sites': plot, 'plot_sds': plot, 'plot_mags': plot
        }
        ind = np.logical_and(freq > 0.3, freq < 10)
        freq = freq[ind]
        g0 = np.array(g0)[ind]
        b = np.array(b)[ind]
        with tempdir(delete=not plot):
            run(create_config='conf.json', tutorial=True)
            result = run(conf='conf.json', **kwargs)
            if plot:
                plot_comparison(result['freq'], freq, result['g0'], g0,
                                result['b'], b)
        M0_qopen = {evid.split('_')[0]: r.get('M0')
                    for evid, r in result['events'].items()}
        temp = [(M0_qopen[evid], M0[evid]) for evid in sorted(M0)]
        M0_qopen, M0 = zip(*temp)
        M0 = np.array(M0) / 2 ** 0.5  # wrong surface correction in paper
        # There seems to be a wrong factor of around 1e4 in the observed
        # envelopes (fig. 3). the error could be in the paper or the script.
        # Therefore M0 should be different by a factor of 1e2,
        # but actually they differ by a factor of 10. This corresponds to
        # a magnitude differenc of 0.67. The magnitude comparison mags.pdf
        # suggests that the determined M0s by the script are ok.
        M0 = 10 * M0

#        np.set_printoptions(formatter={'all':lambda x: '%.2g' % x})
#        print('g0 test vs paper')
#        print(np.array(result['g0']))
#        print(g0)
#        print('b test vs paper')
#        print(np.array(result['b']))
#        print(b)
#        print('M0 test vs paper')
#        print(np.array(M0_qopen))
#        print(M0)
#        plot_comparison(result['freq'], freq, result['g0'], g0, result['b'],b)

        np.testing.assert_equal(result['freq'], freq)
#        print(np.log10(result['g0'] / g0))
#        print(np.log10(result['b'] / b))
#        print(np.log10(M0_qopen / M0))
        np.testing.assert_array_less(np.abs(np.log10(result['g0'] / g0)), 0.5)
        np.testing.assert_array_less(np.abs(np.log10(result['b'] / b)), 0.5)
        np.testing.assert_array_less(np.abs(np.log10(M0_qopen / M0)), 0.51)

    def test_plugin_option(self):
        f = init_data('plugin', plugin='qopen.tests.test_core : gw_test')
        self.assertEqual(f(nework=4, station=2), 42)


def plot_comparison(freq1, freq2, g1, g2, b1, b2):
    from pylab import subplot, loglog, savefig, legend
    subplot(121)
    loglog(freq1, g1, label='g0')
    loglog(freq2, g2, label='g0 GJI')
    legend()
    subplot(122)
    loglog(freq1, b1, label='b')
    loglog(freq2, b2, label='b GJI')
    legend()
    savefig('comparison.pdf')


def gw_test(**kwargs):
    return 42

if __name__ == '__main__':
    unittest.main()
