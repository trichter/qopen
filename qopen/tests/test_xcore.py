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
from obspy import read_events
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

    def check_num_images(self, expr, num):
        msg = ('Only %d plot files are created (glob expression %s).\n\n'
               'Created files are:\n%s\n\n'
               '%s')
        if os.path.exists('example.log'):
            with open('example.log') as flog:
                log = 'Content of log file:\n' + flog.read()
        else:
            log = 'Log file does not exist.'
        files = list(glob(expr))
        msg2 = msg % (len(files), expr, files, log)
        self.assertEqual(len(files), num, msg=msg2)

    def test_entry_point(self):
        script = load_entry_point('qopen', 'console_scripts', 'qopen')
        with quiet():
            try:
                script(['-h'])
            except SystemExit:
                pass

    def test_x_cmdline(self):
        # TODO: add test with picks
        if not self.all_tests:
            raise unittest.SkipTest('save time')
        script = run_cmdline

        args = []
        if self.njobs:
            args.extend(['--njobs', self.njobs])
        if self.verbose:
            args.append('-vvv')
        tempdirname = 'qopen_test1' if self.permanent_tempdir else None
        with tempdir(tempdirname, self.delete):
            script(['--create-config', '--tutorial'])
            script(args)
            # 5*5 energies, optimization, fits
            # + 5 eventresults, eventsites = 85
            self.check_num_images('plots/*.png', 85)
            self.check_num_images('plots/*.pdf', 4)
            with open('results.json') as f:
                result1 = json.load(f)
        # now check for "invert_events_simultaneously": true
        tempdirname = 'qopen_test2' if self.permanent_tempdir else None
        with tempdir(tempdirname, self.delete):
            script(['--create-config', '--tutorial'])
            _replace_in_file(
                'conf.json', 'conf2.json',
                '"invert_events_simultaneously": false',
                '"invert_events_simultaneously": true')
            script(args + ['-c', 'conf2.json'])
            # 5 energies, optimization, fits = 15
            self.check_num_images('plots/*.png', 15)
            self.check_num_images('plots/*.pdf', 4)
            with open('results.json') as f:
                result2 = json.load(f)
        # check similarity of results for invert_events_simultaneously
        np.testing.assert_allclose(result2['freq'], result1['freq'])
        np.testing.assert_allclose(result2['g0'], result1['g0'], rtol=0.3)
        np.testing.assert_allclose(result2['b'], result1['b'], rtol=0.6)
        np.testing.assert_allclose(result2['error'], result1['error'],
                                   rtol=0.3)
        for sta in result1['R']:
            np.testing.assert_allclose(result2['R'][sta], result1['R'][sta],
                                       rtol=0.5)

    def test_y_tutorial_full(self):
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
            'plot_optimization': plot,
            'plot_energies': plot,
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
        tempdirname = 'qopen_test3' if self.permanent_tempdir else None
        with tempdir(tempdirname, self.delete):
            run(create_config='conf.json', tutorial=True)
            result = run(conf='conf.json', **kwargs)
            if plot:
                plot_comparison(result['freq'], freq, result['g0'], g0,
                                result['b'], b)
                self.check_num_images('plots/*.png', 85)
                self.check_num_images('plots/*.pdf', 4)

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

    def test_tutorial_codaQ(self):
        plot = self.all_tests
        freq = np.array([0.375, 0.75, 1.5, 3.0, 6.0])
        b = np.array([0.012, 0.019, 0.029, 0.038, 0.047])
#        freq = np.array([3.0, 6.0])
#        b = np.array([0.038, 0.047])
        kwargs = {
            "freqs": {"width": 1, "cfreqs": list(freq)},
            "optimize": None,
            "bulk_window": None,
            "G_plugin": "qopen.rt : G_diffapprox3d",
            "seismic_moment_method": None,
            'plot_optimization': plot,
            'plot_energies': plot, 'plot_fits': plot,
            'plot_eventresult': plot, 'plot_eventsites': plot,
            'plot_results': plot,
            'plot_sites': plot, 'plot_sds': plot, 'plot_mags': plot,
        }
        if self.njobs:
            kwargs['njobs'] = int(self.njobs)
        if self.verbose:
            kwargs['verbose'] = 3
        tempdirname = 'qopen_test4' if self.permanent_tempdir else None
        with tempdir(tempdirname, self.delete):
            run(create_config='conf.json', tutorial=True)
            events = read_events('example_events.xml', 'QUAKEML')[:2]
            result = run(conf='conf.json', events=events, **kwargs)
            if plot:
                plot_comparison(result['freq'], freq, None, None,
                                result['b'], b)
                # 5 * 2 * 2 energies, fits
                # 2 * 2 eventresults, eventsites
                self.check_num_images('plots/*.png', 24)
                self.check_num_images('plots/*.pdf', 3)
        np.testing.assert_equal(result['freq'], freq)
        np.testing.assert_array_less(np.abs(np.log10(result['b'] / b)), 0.5)

    def test_tutorial_codanorm(self):
        plot = self.all_tests
#        freq = np.array([0.375, 0.75, 1.5, 3.0, 6.0])
#        b = np.array([0.012, 0.019, 0.029, 0.038, 0.047])
        freq = np.array([3.0, 6.0])
        b = np.array([0.038, 0.047])
        kwargs = {
            "freqs": {"width": 1, "cfreqs": list(freq)},
            "coda_normalization": [180, 200],
            "seismic_moment_method": None,
            'plot_optimization': plot,
            'plot_energies': plot, 'plot_fits': plot,
            'plot_eventresult': plot, 'plot_eventsites': plot,
            'plot_results': plot,
            'plot_sites': plot, 'plot_sds': plot, 'plot_mags': plot,
        }
        if self.njobs:
            kwargs['njobs'] = int(self.njobs)
        if self.verbose:
            kwargs['verbose'] = 3
        tempdirname = 'qopen_test5' if self.permanent_tempdir else None
        with tempdir(tempdirname, self.delete):
            run(create_config='conf.json', tutorial=True)
            events = read_events('example_events.xml', 'QUAKEML')[:2]
            result = run(conf='conf.json', events=events, **kwargs)
            if plot:
                plot_comparison(result['freq'], freq, None, None,
                                result['b'], b)
                # 3 * 2 * 2 energies, fits, optimization
                # 2 * 2 eventresults, eventsites
                self.check_num_images('plots/*.png', 16)
                self.check_num_images('plots/*.pdf', 3)
        np.testing.assert_equal(result['freq'], freq)
        np.testing.assert_array_less(np.abs(np.log10(result['b'] / b)), 0.5)

    def test_tutorial_everything_with_nans(self):
        """Test scenarios:
            * no results for one station
            * no results for one event
            * no results for single frequency
            * only some results for single frequency

            * test qopen
            * test qopen --fix-params
            * test qopen --align-sites
            * test qopen --calc-source-params
        """
        if not self.all_tests:
            raise unittest.SkipTest('save time')
        plot = self.all_tests
        freq = np.array([0.01, 0.375, 0.75, 1.5, 3.0, 6.0])
        kwargs = {
            "freqs": {"width": 1, "cfreqs": list(freq)},
            #            "coda_normalization": [180, 200],
            #            "seismic_moment_method": None,
            'plot_optimization': plot,
            'plot_energies': plot, 'plot_fits': plot,
            'plot_eventresult': plot, 'plot_eventsites': plot,
            'plot_results': plot,
            'plot_sites': plot, 'plot_sds': plot, 'plot_mags': plot,
        }
        if self.njobs:
            kwargs['njobs'] = int(self.njobs)
        if self.verbose:
            kwargs['verbose'] = 3
        tempdirname = 'qopen_test6' if self.permanent_tempdir else None
        with tempdir(tempdirname, self.delete):
            run(create_config='conf.json', tutorial=True)
            events = read_events('example_events.xml', 'QUAKEML')
            # create one fake event for which no data is present
            events2 = read_events('example_events.xml', 'QUAKEML')
            events = events + events2[:1]
            events[-1].resource_id = 'fake'
            events[-1].origins[0].time += 500
            result = run(conf='conf.json', events=events,
                         coda_window=["S+20s", ["S+150s", "5SNR"]],
                         skip={"coda_window": 120},
                         **kwargs)
            self.assertIsNone(result['g0'][0])
            self.assertIsNone(result['b'][0])
            self.assertNotIn('fake', result['events'])
            result2 = run(conf='conf.json', events=events,
                          coda_window=["S+20s", ["S+150s", "10SNR"]],
                          skip={"coda_window": 120},
                          fix_params='results.json',
                          output='results_fixparams.json',
                          plot_sds_options={"fname": "plots/sds_fix.pdf"},
                          plot_sites_options={"fname": "plots/sites_fix.pdf"},
                          **kwargs)
            self.assertNotIn('b', result2)
            result3 = run(conf='conf.json',
                          align_sites='results_fixparams.json',
                          output='results_aligned.json',
                          plot_sds_options={"fname": "plots/sds_al.pdf"},
                          plot_sites_options={"fname": "plots/sites_al.pdf"},
                          **kwargs)
            self.assertNotIn('b', result3)
            result4 = run(conf='conf.json',
                          calc_source_params='results_aligned.json',
                          output='results_aligned_source_params.json',
                          plot_sds_options={"fname": "plots/sds_al2.pdf"},
                          plot_sites_options={"fname": "plots/sites_al2.pdf"},
                          **kwargs)
            self.assertNotIn('b', result4)
            self.assertEqual(result4, result3)

    def test_plugin_option(self):
        f = init_data('plugin', plugin='qopen.tests.test_xcore : gw_test')
        self.assertEqual(f(nework=4, station=2), 42)


def plot_comparison(freq1, freq2, g1, g2, b1, b2):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    if g1:
        ax1.loglog(freq1, g1, label='g0')
        ax1.loglog(freq2, g2, label='g0 GJI')
        ax1.legend()
    ax2 = fig.add_subplot(122)
    ax2.loglog(freq1, b1, label='b')
    ax2.loglog(freq2, b2, label='b GJI')
    ax2.legend()
    fig.savefig('comparison.pdf')


def gw_test(**kwargs):
    return 42


if __name__ == '__main__':
    unittest.main()
