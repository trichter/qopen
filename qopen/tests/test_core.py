# Copyright 2015-2017 Tom Eulenfeld, MIT license
"""
Tests for core module.
"""

from glob import glob
import json
import os
from pkg_resources import load_entry_point
import sys
import unittest

import numpy as np
from qopen.core import init_data, run, run_cmdline
from qopen.tests.util import tempdir, quiet


def _replace_in_file(fname_src, fname_dest, str_src, str_dest):
    with open(fname_src) as f:
        text = f.read()
    text = text.replace(str_src, str_dest)
    with open(fname_dest, 'w') as f:
        f.write(text)


class TestCase(unittest.TestCase):

    def setUp(self):
        args = sys.argv[1:]
        self.verbose = '-v' in args
        self.permanent_tempdir = '-p' in args
        self.delete = '-d' in args
        self.all_tests = '-a' in args
        self.njobs = args[args.index('-n') + 1] if '-n' in args else None

    def test_entry_point(self):
        script = load_entry_point('qopen', 'console_scripts', 'qopen')
        with quiet():
            try:
                script(['-h'])
            except SystemExit:
                pass

    def test_cmdline(self):
        # TODO: add test with picks
        if not self.all_tests:
            raise unittest.SkipTest('save time')
        script = run_cmdline
        msg = ('Only %d plot files (%s) are created.\n\n'
               'Created files are:\n%s\n\n'
               '%s')
        args = []
        if self.njobs:
            args.extend(['--njobs', self.njobs])
        if self.verbose:
            args.append('-vvv')
        tempdirname = 'qopen_test1' if self.permanent_tempdir else None
        with tempdir(tempdirname, self.delete):
            script(['--create-config', '--tutorial'])
            script(args)
            # check if pictures were created
            if os.path.exists('example.log'):
                with open('example.log') as flog:
                    log = 'Content of log file:\n' + flog.read()
            else:
                log = 'Log file does not exist.'
            files = list(glob('plots/*.png'))
            msg2 = msg % (len(files), 'png', files, log)
            # 5*5 energies, optimization, fits
            # + 5 eventresults, eventsites = 85
            self.assertEqual(len(files), 85, msg=msg2)
            files = list(glob('plots/*.pdf'))
            msg2 = msg % (len(files), 'pdf', files, log)
            self.assertEqual(len(files), 4, msg=msg2)
            with open('results.json') as f:
                result1 = json.load(f)
        # now check for "invert_events_simultaniously": true
        tempdirname = 'qopen_test2' if self.permanent_tempdir else None
        with tempdir(tempdirname, self.delete):
            script(['--create-config', '--tutorial'])
            _replace_in_file(
                'conf.json', 'conf2.json',
                '"invert_events_simultaniously": false',
                '"invert_events_simultaniously": true')
            args.extend(['-c', 'conf2.json'])
            script(args)
            # check if pictures were created
            if os.path.exists('example.log'):
                with open('example.log') as flog:
                    log = 'Content of log file:\n' + flog.read()
            else:
                log = 'Log file does not exist.'
            files = list(glob('plots/*.png'))
            msg2 = msg % (len(files), 'png', files, log)
            # 5 energies, optimization, fits = 15
            self.assertEqual(len(files), 15, msg=msg2)
            files = list(glob('plots/*.pdf'))
            msg2 = msg % (len(files), 'pdf', files, log)
            self.assertEqual(len(files), 4, msg=msg2)
            with open('results.json') as f:
                result2 = json.load(f)
        # check similarity of results for invert_events_simultaniously
        np.testing.assert_allclose(result2['freq'], result1['freq'])
        np.testing.assert_allclose(result2['g0'], result1['g0'], rtol=0.3)
        np.testing.assert_allclose(result2['b'], result1['b'], rtol=0.6)
        np.testing.assert_allclose(result2['error'], result1['error'],
                                   rtol=0.3)
        for sta in result1['R']:
            np.testing.assert_allclose(result2['R'][sta], result1['R'][sta],
                                       rtol=0.5)

    def test_results_of_tutorial(self):
        """Test against publication of Sens-Schoenfelder and Wegler (2006)"""
        plot = self.all_tests
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
            'plot_sites': plot, 'plot_sds': plot, 'plot_mags': plot,
        }
        if self.njobs:
            kwargs['njobs'] = int(self.njobs)
        if self.verbose:
            kwargs['verbose'] = 3
        ind = np.logical_and(freq > 0.3, freq < 10)
        freq = freq[ind]
        g0 = np.array(g0)[ind]
        b = np.array(b)[ind]
        tempdirname = 'qopen_test2' if self.permanent_tempdir else None
        with tempdir(tempdirname, self.delete):
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

        # check if the same result for 1 core
        if self.njobs != '1' and self.all_tests:
            kwargs['njobs'] = 1
            with tempdir():
                run(create_config='conf.json', tutorial=True)
                result2 = run(conf='conf.json', **kwargs)
            self.assertEqual(result, result2)

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
