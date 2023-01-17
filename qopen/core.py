# Copyright 2015-2020 Tom Eulenfeld, MIT license
"""
Qopen command line script and routines

:func:`run_cmdline` is started by the ``qopen`` command line script.
Import and call :func:`run` if you want to use *Qopen* inside Python code:

>>> from qopen import run
>>> run(conf='conf.json')

Qopen will run the following functions top-down:

.. autosummary::
  :nosignatures:

   run_cmdline
   run
   invert_wrapper
   invert
   invert_fb

|
"""

import argparse
from argparse import SUPPRESS
from collections import defaultdict, OrderedDict
from copy import copy, deepcopy
from functools import partial
from importlib import import_module
import json
import logging
import logging.config
import multiprocessing
import os.path
from pkg_resources import resource_filename
import shutil
import sys
import time

import numpy as np
import obspy
from obspy.geodetics import gps2dist_azimuth
import scipy
import scipy.signal
from statsmodels.regression.linear_model import WLS

import qopen
from qopen.site import align_site_responses
from qopen.source import calculate_source_properties, insert_source_properties
from qopen.util import (cache, gmeanlist, smooth as smooth_, smooth_func,
                        LOGGING_DEFAULT_CONFIG)

IS_PY3 = sys.version_info.major == 3

log = logging.getLogger('qopen')
log.addHandler(logging.NullHandler())

LOGLEVELS = {0: 'CRITICAL', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}

DUMP_CONFIG = ['invert_events_simultaneously', 'mean',
               'v0', 'rho0', 'R0', 'free_surface',
               'freqs', 'filter', 'optimize', 'g0_bounds', 'b_bounds',
               'seismic_moment_method', 'seismic_moment_options',
               'bulk_window', 'coda_normalization',
               'coda_window', 'noise_windows',
               'weight', 'remove_noise',
               'adjust_sonset', 'adjust_sonset_options',
               'remove_response', 'remove_response_options',
               'correct_for_elevation', 'skip',
               'G_module']

DUMP_ORDER = ['M0', 'Mw', 'Mcat', 'fc', 'n', 'gamma',
              'freq', 'g0', 'b', 'nstations', 'error',
              'W', 'sds', 'sds_error', 'fit_error',
              'R', 'events', 'v0', 'config']

BOREHOLE_STATION_DEPTH = 10


class QopenError(Exception):
    pass


class ParseError(QopenError):
    pass


class SkipError(QopenError):
    pass


def sort_dict(dict_, order=DUMP_ORDER):
    return OrderedDict(sorted(dict_.items(), key=lambda t: order.index(t[0])))


@cache
def filter_width(sr, freq=None, freqmin=None, freqmax=None, corners=2,
                 zerophase=False, type='bandpass'):
    """Integrate over the squared filter response of a Butterworth filter

    The result corresponds to the filter width, which equals approximately
    the difference of the corner frequencies. The energy density should
    be divided by the result to get the correct spectral energy density.

    :param sr: sampling rate
    :param freq: corner frequencies of low- or highpass filter
    :param freqmin,freqmax: corner frequencies of bandpass filter
    :param corners: number of corners
    :param zerophase: if True number of corners are doubled
    :param type: 'bandpass', 'highpass' or 'lowpass'
    :return: filter width
    """
    if type == 'bandpass':
        fs = (freqmin / (0.5 * sr), freqmax / (0.5 * sr))
        ftext = '%.2gHz-%.2gHz' % (freqmin, freqmax)
    else:
        fs = freq / (0.5 * sr)
        ftext = '%.2gHz' % freq
    b, a = scipy.signal.iirfilter(corners, fs, btype=type.strip('pass'),
                                  ftype='butter', output='ba')
    w, h = scipy.signal.freqz(b, a)
    df = (w[1] - w[0]) / 2 / np.pi * sr
    ret = df * np.sum(np.abs(h) ** (2 * (zerophase + 1)))
    msg = ('%s filter (%s, %d corners, zerophase=%s, sr=%.1fHz) '
           'has a width of %.2gHz')
    log.debug(msg, type, ftext, corners, zerophase, sr, ret)
    return ret


@cache
def get_freqs(max=None, min=None, step=None, width=None, cfreqs=None,
              fbands=None):
    """Determine frequency bands

    :param args: See example configuration file.
    :return: ordered dictionary {central frequency: corner frequencies}"""
    if cfreqs is None and fbands is None:
        max_exp = int(np.log(max / min) / step / np.log(2))
        exponents = step * np.arange(max_exp + 1)[::-1]
        cfreqs = max / 2 ** exponents
    if fbands is None:
        df = np.array(cfreqs) * (2 ** width - 1) / (2 ** width + 1)
        fbands = OrderedDict((f, (f - d, f + d)) for d, f in zip(df, cfreqs))
    else:
        fbands = sorted(fbands)
        cfreqs = [0.5 * (f1 + f2) for f1, f2 in fbands]
        fbands = OrderedDict((0.5 * (f1 + f2), (f1, f2)) for f1, f2 in fbands)
    msg = 'central frequencies: (' + '%s, ' * (len(cfreqs) - 1) + '%s)'
    log.info(msg, *cfreqs)
    msg = ('freq bands: ' + '(%.3f, %.3f), ' * len(cfreqs))[:-2]
    log.info(msg, *np.array(sorted(fbands.values())).flat)
    return fbands


def energy1c(data, rho, df, fs=4):
    """Spectral energy density of one channel

    :param data: velocity data (m/s)
    :param rho: density (kg/m**3)
    :param df: filter width in Hz
    :param fs: free surface correction (default: 4)
    :return: energy density
    """
    hilb = scipy.fftpack.hilbert(data)
    return rho * (data ** 2 + hilb ** 2) / 2 / df / fs


def observed_energy(stream, rho, df, coda_normalization=None, fs=4, tolerance=1):
    """
    Return trace with total spectral energy density of three component stream

    :param stream: stream of a 3 component seismogram
    :param rho: density (kg/m**3)
    :param df: filter width in Hz
    :param fs: free surface correction (default: 4)
    :param tolerance: the number of samples the length of the traces
        in the 3 component stream may differ (default: 1)
    :return: trace with total energy density"""
    data = [energy1c(tr.data, rho, df, fs=fs) for tr in stream]
    Ns = [len(d) for d in data]
    if max(Ns) - min(Ns) > tolerance:
        msg = ('traces for one stream have different lengths %s. Tolerance '
               ' is %d samples') % (Ns, tolerance)
        raise SkipError(msg)
    elif max(Ns) - min(Ns) > 0:
        data = [d[:min(Ns)] for d in data]
    data = np.sum(data, axis=0)
    tr = obspy.Trace(data=data, header=stream[0].stats)
    tr.stats.channel = tr.stats.channel[:2] + 'X'
    if coda_normalization is not None:
        sl = tr.slice(tr.stats.origintime + coda_normalization[0],
                      tr.stats.origintime + coda_normalization[1])
        tr.data = tr.data / np.mean(sl.data)
    return tr


def get_station(seedid):
    """Station name from seed id"""
    st = seedid.rsplit('.', 2)[0]
    if st.startswith('.'):
        st = st[1:]
    return st


def get_eventid(event):
    """Event id from event"""
    return str(event.resource_id).split('/')[-1]


def get_pair(tr):
    """Station and event id from trace"""
    return (tr.stats.eventid, get_station(tr.id))


def get_origin(event):
    """Preferred or first origin from event"""
    return event.preferred_origin() or event.origins[0]


def get_magnitude(event):
    """Preferred or first magnitude of event

    This is not the coda magnitude determined by the script, but the magnitude
    from the original event catalogue.
    """
    try:
        mag = event.preferred_magnitude() or event.magnitudes[0]
    except IndexError:
        return
    return mag.mag


def get_arrivals(event):
    """Arrivals of appropriate origin from event"""
    ar = get_origin(event).arrivals
    if len(ar) > 0:
        return ar
    ar = event.origins[0].arrivals
    if len(ar) > 0:
        msg = ('event %s: Preferred origin has no arrivals, but first '
               'origin has -> take these')
        log.debug(msg, get_eventid(event))
        return ar
    candidates = [o.arrivals for o in event.origins if len(o.arrivals) > 0]
    if len(candidates) == 0:
        msg = 'event %s: No picks availlable for any origin -> skip event'
        log.warning(msg, get_eventid(event))
    elif len(candidates) == 1:
        msg = ('event %s: Preferred origin has no arrivals, but one other '
               'origin has -> take these')
        log.debug(msg, get_eventid(event))
        return candidates[0]
    else:
        msg = ('event %s: Preferred origin has no arrivals and multiple other '
               'origins contain arrivals -> skip event')
        log.warning(msg, get_eventid(event))


def get_picks(arrivals, station):
    """Picks for specific station from arrivals"""
    picks = {}
    for arrival in arrivals:
        phase = arrival.phase.upper()
        if phase in ('PG', 'SG'):
            phase = phase[0]
        if phase not in 'PS':
            continue
        pick = arrival.pick_id.get_referred_object()
        seedid = pick.waveform_id.get_seed_string()
        if station == get_station(seedid):
            if phase in picks:
                msg = 'station %s: %s-onset has multiple picks'
                log.warning(msg, station, phase)
                # if phase == 'P':
                #     log.warning(msg, station, phase)
                # else:
                #     raise SkipError(msg % (station, phase))
            picks[phase] = pick.time
    return picks


def time2utc(time, trace, starttime=None):
    """Convert string with time information to UTCDateTime object

    :param time: can be one of:\n
         "OT+-???s" seconds relative to origin time\n
         "P+-???s" seconds relative to P-onset\n
         "S+-???s" seconds relative to S-onset\n
         "???Ptt" travel time relative to P-onset travel time\n
         "???Stt" travel time relative to S-onset travel time\n
         "???SNR" time after which SNR falls below this value
                  (after time given in starttime)\n
         "time>???SNR" time after which SNR falls below this value
                  (after time given in front of expression)
    :param trace: Trace object with stats entries
    :param starttime: UTCDatetime object for SNR case.
    """
    ot = trace.stats.origintime
    p = trace.stats.ponset
    s = trace.stats.sonset
    time = time.lower()
    if time.endswith('snr'):
        if '>' in time:
            time1, time = time.split('>')
            if time1 != '':
                st = time2utc(time1, trace)
                if st < starttime:
                    msg = "time stated before '<' is before window starttime"
                    log.warning(msg)
                starttime = st
        assert starttime is not None
        tr = trace.slice(starttime=starttime)
        snr = float(time[:-3])
        noise_level = tr.stats.noise_level
        try:
            index = np.where(tr.data < snr * noise_level)[0][0]
        except IndexError:
            index = len(tr.data)
        t = starttime + index * tr.stats.delta
    elif time.endswith('stt') or time.endswith('ptt'):
        rel = p if time[-3] == 'p' else s
        t = ot + float(time[:-3]) * (rel - ot)
    elif ((time.startswith('s') or time.startswith('p') or
           time.startswith('ot')) and time.endswith('s')):
        rel = p if time.startswith('p') else s if time.startswith('s') else ot
        t = rel + float(time[2:-1] if time.startswith('ot') else time[1:-1])
    else:
        raise ValueError('Unexpected value for time window')
    return t


def tw2utc(tw, trace):
    """Convert time window to UTC time window

    :param tw: tuple of two values, both can be a string (see :func:`time2utc`)
        or a list of strings in which case the latest starttime and earliest
        endtime is taken.
    :param trace: Trace object with stats entries
    """
    starttime = None
    for val in tw:
        if isinstance(val, (list, tuple)):
            times = [time2utc(v, trace, starttime=starttime) for v in val]
            t = max(times) if starttime is None else min(times)
        else:
            t = time2utc(val, trace, starttime=starttime)
        if starttime is None:
            starttime = t
    return starttime, t


def collect_results(results, only=None, freqi=None):
    """
    Collect g0, b, error, R, W, eventids and v0 from results of multiple events

    :param results: result dictionary returned by :func:`run`
    :param only: return only some of the above mentioned keys
    :param freqi: return only values at a single frequency
    :return: dictionary
    """
    def freq_getter(r, c):
        if freqi == None or c in ('eventid', 'v0'):
            return r
        else:
            return r[freqi]

    if only is None:
        collect = ('g0', 'b', 'error', 'R', 'W', 'eventid', 'v0')
    else:
        collect = only
    col = defaultdict(list)
    if 'R' in collect:
        col['R'] = defaultdict(list)
    for eventid, res in results['events'].items():
        if res is None:
            continue
        for c in collect:
            if c == 'eventid':
                col[c].append(eventid)
            elif c == 'R' and 'R' in res:
                for sta, Rsta in res['R'].items():
                    col['R'][sta].append(freq_getter(Rsta, 'R'))
            elif c in res:
                col[c].append(freq_getter(res[c], c))
    if 'R' in col:
        col['R'] = dict(col['R'])
    col = dict(col)
    for c in collect:
        if c == 'eventid':
            pass
        elif c == 'R' and 'R' in col:
            for sta in col['R']:
                col['R'][sta] = np.array(col['R'][sta], dtype=float)
        elif c in col:
            col[c] = np.array(col[c], dtype=float)
    return col
#    # old implementation returns list
#    if 'g0' not in list(results['events'].items())[0][1]:
#        g0 = [results['g0']]
#        b = [results['b']]
#        error = [results['error']]
#        v0 = [results.get['v0']]
#        R = {sta: [Rsta] for sta, Rsta in results['R'].items()}
#        W, eventids = [], []
#        for eventid, res in results['events'].items():
#            W.append(res['W'])
#            eventids.append(eventid)
#    else:
#        g0, b, error, R, W, eventids = [], [], [], defaultdict(list), [], []
#        v0 = []
#        for eventid, res in results['events'].items():
#            if res is None:
#                continue
#            g0.append(res['g0'])
#            b.append(res['b'])
#            error.append(res['error'])
#            W.append(res['W'])
#            eventids.append(eventid)
#            v0.append(res.get('v0'))
#            for sta, Rsta in res['R'].items():
#                R[sta].append(Rsta)
#        R = dict(R)
#    g0 = np.array(g0, dtype=np.float)
#    b = np.array(b, dtype=np.float)
#    error = np.array(error, dtype=np.float)
#    W = np.array(W, dtype=np.float)
#    v0 = np.array(v0, dtype=np.float)
#    for sta in R:
#        R[sta] = np.array(R[sta], dtype=np.float)
#    return g0, b, error, R, W, eventids, v0


def _check_times(tr, tw, tol=0.5):
    d1 = tw[0] - tr.stats.starttime
    d2 = tr.stats.endtime - tw[1]
    if d1 + tol > 0 and d2 + tol > 0:
        return
    else:
        return (d1, d2)
#    return (d1 + tol > 0 and d2 + tol > 0) or (d1, d2)
    return tr.stats.starttime > tw[0] + tol or tr.stats.endtime < tw[1] - tol


def Gsmooth(G_func, r, t, v0, g0, smooth=None, smooth_window='flat'):
    """Return smoothed Green's function as a function of time"""
    Gc = smooth_func(lambda t_: G_func(r, t_, v0, g0),
                     t, smooth, window=smooth_window)
    return Gc


def _get_local_minimum(tr, smooth=None, ratio=5, smooth_window='flat'):
    data = tr.data
    if smooth:
        window_len = int(round(smooth * tr.stats.sampling_rate))
        try:
            data = smooth_(tr.data, window_len=window_len, method='clip',
                           window=smooth_window)
        except ValueError:
            pass
    mins = scipy.signal.argrelmin(data)[0]
    maxs = scipy.signal.argrelmax(data)[0]
    if len(mins) == 0 or len(maxs) == 0:
        return
    mins2 = [mins[0]]
    for mi in mins[1:]:
        if data[mi] < data[mins2[-1]]:
            mins2.append(mi)
    mins = np.array(mins2)
    for ma in maxs:
        try:
            mi = np.nonzero(mins < ma)[0][-1]
            mi = mins[mi]
        except IndexError:
            mi = 0
        if data[ma] / data[mi] > ratio:
            return tr.stats.starttime + mi * tr.stats.delta


def _get_slice(energy, tw, pair, energies, bulk=False):
    s = 'bulk' if bulk else 'coda'
    try:
        energyslice = energy.slice(*tw)
        if _check_times(energyslice, tw):
            raise ValueError('not enough data inside %s window' % s)
        return energyslice
    except ValueError as ex:
        msg = '%s: cannot get %s window (%s) -> skip pair'
        log.warning(msg, pair, s, ex)
        energies.remove(energy)


def invert_fb(freq_band, streams, filter, rho0, v0, coda_window,
              R0=1, free_surface=4,
              noise_windows=None, noise_windows_func='min',
              bulk_window=None, weight=None,
              optimize=None, g0_bounds=(1e-8, 1e-3), b_bounds=(1e-5, 10),
              num_points_integration=1000, coda_normalization=None,
              smooth=None, smooth_window='flat',
              remove_noise=False, cut_coda=None, skip=None,
              adjust_sonset=None, adjust_sonset_options={},
              plot_energies=False, plot_energies_options={},
              plot_optimization=False, plot_optimization_options={},
              plot_fits=False, plot_fits_options={},
              ignore_network_code=False, borehole_stations=(),
              G_plugin='qopen.rt : G_rt3d',
              fix=False, fix_params=False,
              fix_sites=False, fix_sites_params=None,
              dump_optpkl=None, dump_fitpkl=None,
              **kwargs):
    """
    Inverst streams in a specific frequency band for attenuation parameters

    :parameters:
        **freq_band**, **streams**, **borehole_stations**,
        **fix**, **fix_params** , **fix_sites**, **fix_sites_params** --
        are determined in :func:`invert`.
        All other options are described in the example configuration file.
    :return: result tuple

    """
    if skip is None:
        skip = {}
    # coda window is forced to have a minimal length of 0.05s
    # this value should be configured much higher
    skip.setdefault('coda_window', 0.05)
    msg = 'freq band (%.2fHz, %.2fHz): start optimization'
    log.debug(msg, *freq_band)
    if len(streams) == 0:
        msg = ('freq band (%.2fHz, %.2fHz): no data availlable '
               '-> skip frequency band')
        log.error(msg, *freq_band)
        return

    def _tw_utc2s(tw_utc, otime):
        tw = []
        for t in tw_utc:
            tw.append(t - otime)
        return '(%.2fs, %.2fs)' % tuple(tw)

    # Filter traces, normalize to preserve energy density
    # and calculate observed energy
    freqmin, freqmax = freq_band
    energies = []
    for stream in streams:
        pair = get_pair(stream[0])
        sr = stream[0].stats.sampling_rate
        if (freqmin + freqmax) > sr:
            msg = ('%s: Central frequency is above Nyquist -> skip pair '
                   'for frequency band')
            log.warning(msg, pair)
            continue
        if fix_sites and np.isnan(
                fix_sites_params[freq_band].get(pair[1], np.nan)):
            msg = ('%s: Reference site amplification not available '
                   '-> skip pair for frequency band')
            log.warning(msg, pair)
            continue
        filter_ = copy(filter)
        if freqmax > 0.495 * sr:
            fu = {'freq': freqmin, 'type': 'highpass'}
        else:
            fu = {'freqmin': freqmin, 'freqmax': freqmax, 'type': 'bandpass'}
        filter_.update(fu)
        stream.detrend('linear')
        stream.filter(**filter_)
        df = filter_width(sr, **filter_)
        fs = free_surface
        if isinstance(fs, (list, tuple)):
            if get_station(stream[0].id) in borehole_stations:
                fs = fs[1]
            else:
                fs = fs[0]
        try:
            energies.append(observed_energy(
                stream, rho0, df, coda_normalization, fs=fs))
        except SkipError as ex:
            msg = '%s: cannot calculate ernergy (%s)'
            log.warning(msg, pair, str(ex))
        except Exception:
            msg = '%s: cannot calculate ernergy'
            log.exception(msg, pair)

    bulkw = {}
    codaw = {}
    time_adjustments = []
    for energy in energies[:]:
        # Calculate noise level
        pair = get_pair(energy)
        noise_levels = []
        otime = energy.stats.origintime
        sonset = energy.stats.sonset
        distance = energy.stats.distance
        for i, nw in enumerate(noise_windows):
            noisew = tw2utc(nw, energy)
            try:
                tr_noisew = energy.slice(*noisew)
            except ValueError:
                continue
            if len(tr_noisew.data) and np.all(np.isfinite(tr_noisew.data)):
                noise_levels.append(np.mean(tr_noisew.data))
        if len(noise_levels) == 0:
            noise_level = None
            msg = '%s: all noise windows outside data'
            if remove_noise:
                msg = msg + ' -> skip pair for frequency band'
            log.warning(msg, pair)
            if remove_noise:
                energies.remove(energy)
                continue
        else:
            _nwfunc = getattr(np, noise_windows_func)
            energy.stats.noise_level = noise_level = _nwfunc(noise_levels)
            msg = '%s: noise level at %.1e, max value %.1e'
            log.debug(msg, pair, noise_level, np.max(energy.data))
        # Optionally remove noise
        if remove_noise:
            energy.data = energy.data - noise_level
            energy.data[energy.data < noise_level / 100] = noise_level / 100
        # Optionally adjust S-onset
        if adjust_sonset == "maximum":
            try:
                max_window = adjust_sonset_options['window']
            except KeyError:
                msg = ('no window for maximum specified -> '
                       'take original bulk window')
                log.error(msg)
                max_window = bulk_window
            mw = tw2utc(max_window, energy)
            energy.stats.sonset_old = sonset_old = sonset
            imax = np.argmax(energy.slice(*mw).data)
            energy.stats.sonset = sonset = mw[0] + imax * energy.stats.delta
            msg = '%s: adjust S-onset from %.2fs to %.2fs'
            ta = sonset - sonset_old
            vnew = distance / (sonset - otime)
            time_adjustments.append((ta, vnew))
            log.debug(msg, pair, sonset_old - otime, sonset - otime)

    # Calculate v0 from picks if necessary
    if v0 is None and len(energies) > 0:
        def _get_velocity(st):
            return st.distance / (st.sonset - st.origintime)
        v0 = float(np.mean([_get_velocity(e.stats) for e in energies]))

    distances = {}
    tcoda = []
    tbulk = []
    tbulk_window = {}
    weights_bulk = []
    Ebulk = []
    Ecoda = []
    for energy in energies[:]:
        pair = get_pair(energy)
        otime = energy.stats.origintime
        sonset = energy.stats.sonset
        distances[pair] = distance = energy.stats.distance
        sr = energy.stats.sampling_rate
        # Calculate bulk windows in UTC
        # Calculate mean energy in bulk window and 'balanced' time of this
        # mean
        if bulk_window:
            bulkw[pair] = tw2utc(bulk_window, energy)
            esl = _get_slice(energy, bulkw[pair], pair, energies, bulk=True)
            if esl is None:
                continue
            data = esl.data
            Ebulk_val = np.mean(data)
            Nb = len(data)
            t_ = np.arange(Nb) / sr + distance / v0 + \
                (bulkw[pair][0] - sonset)
            tbulk_val = np.sum(data * t_) / np.sum(data)
            tbulk_window[pair] = (t_[0], t_[-1])

        # Smooth energies
        if smooth:
            if plot_fits or dump_fitpkl:
                energy.data_unsmoothed = energy.data
            energy.data = smooth_(energy.data, int(round(sr * smooth)),
                                  window=smooth_window, method='zeros')
        # Calculate coda windows in UTC
        codaw[pair] = tw2utc(coda_window, energy)
        s = ''
        if bulk_window:
            s = 'bulk window %s ' % (_tw_utc2s(bulkw[pair], otime),)
        msg = '%s: %scoda window %s'
        log.debug(msg, pair, s, _tw_utc2s(codaw[pair], otime))
        # Optionally skip some stations if specified conditions are met
        cw = codaw[pair]
        if cw[1] - cw[0] < skip['coda_window']:
            msg = ('%s: coda window of length %.1fs shorter than '
                   '%.1f -> skip pair')
            log.debug(msg, pair, cw[1] - cw[0], skip['coda_window'])
            energies.remove(energy)
            continue
        # use only data before detected local minimum in coda
        if cut_coda:
            if cut_coda is True:
                cut_coda = {}
            cw = codaw[pair]
            if cut_coda.get('smooth'):
                seam = 0.5 * cut_coda['smooth']
                cw = (cw[0] - seam, cw[1] + seam)
            esl = _get_slice(energy, cw, pair, energies)
            if esl is None:
                continue
            cut_coda.setdefault('smooth_window', smooth_window)
            tmin = _get_local_minimum(esl, **cut_coda)
            if tmin:
                msg = '%s: cut coda at local minimum detected at %.2fs.'
                log.debug(msg, pair, tmin - otime)
                codaw[pair] = (min(codaw[pair][0], tmin), tmin)
            # Optionally skip some stations if specified conditions are met
            cw = codaw[pair]
            if cw[1] - cw[0] < skip['coda_window']:
                msg = ('%s: coda window of length %.1fs shorter than '
                       '%.1f -> skip pair')
                log.debug(msg, pair, cw[1] - cw[0], skip['coda_window'])
                energies.remove(energy)
                continue
        if skip and skip.get('maximum'):
            max_window = skip['maximum']
            mw = tw2utc(max_window, energy)
            imax = np.argmax(energy.data)
            tmax = energy.stats.starttime + imax / sr
            if not mw[0] < tmax < mw[1]:
                msg = ('%s: maximum at %.1fs not in window around S-onset '
                       '(%.1fs, %.1fs) -> skip pair')
                log.debug(msg, pair, tmax - otime,
                          mw[0] - otime, mw[1] - otime)
                energies.remove(energy)
                continue
        # Get coda data
        esl = _get_slice(energy, codaw[pair], pair, energies)
        if esl is None:
            continue
        data = esl.data
        Nc = len(data)
        # Adjust tcoda to onset of Green's function
        tc = np.arange(Nc) / sr + distance / v0 + (codaw[pair][0] - sonset)
        tcoda.append(tc)
        Ecoda.append(data)
        if bulk_window:
            if weight[1] == 'codawindow':
                weight_unit = Nc
            elif weight[1] == 'bulkwindow':
                weight_unit = Nb
            elif weight[1] == 'samples':
                weight_unit = 1
            else:
                msg = ("Unknown unit for weight. Should be one of "
                       "'codawindow', bulkwindow' or 'samples'")
                raise Exception(msg)
            weights_bulk.append(weight[0] * weight_unit)
            tbulk.append(tbulk_val)
            Ebulk.append(Ebulk_val)

    if adjust_sonset and len(time_adjustments) > 0:
        ta, vnew = np.mean(time_adjustments, axis=0)
        msg = ('mean of time adjustments is %.2fs, corresponds to a velocity '
               'of %dm/s')
        log.debug(msg, ta, vnew)

    if len(Ecoda) == 0 or skip and len(Ecoda) < skip.get('num_pairs', 1):
        msg = ('freq band (%.2f, %.2f): only %d pairs left -> skip')
        log.info(msg, freq_band[0], freq_band[1], len(Ecoda))
        return

    # Start inversion
    # Construct coefficient matrix for the inversion
    event_station_pairs = [get_pair(energy) for energy in energies]
    eventids, stations = zip(*event_station_pairs)
    eventids = list(OrderedDict.fromkeys(eventids))
    stations = list(OrderedDict.fromkeys(stations))

    # Bi = ln(Ei) - ln(Gji)
    # Ai = 1 0 0 0 0 -1
    # Solve |AC-B| -> min
    # Construct part of the linear equation system
#    tc = []
#    Eobsc = []
#    tb = []
#    Eobsb = []
#    tbulkinterval = []

    As = []
    Ns = len(stations)
    Ne = len(eventids)
    if coda_normalization is None and not fix_sites:
        for i, B in enumerate(Ecoda + [[_] for _ in Ebulk]):
            A = np.zeros((Ns + Ne - fix, len(B)))
#            A[i % Ns, :] = 1
            evid, st = event_station_pairs[i % len(event_station_pairs)]
            A[stations.index(st), :] = 1
            idx = eventids.index(evid)
            if idx > 0:
                A[Ns + idx - 1, :] = 1
            As.append(A)
        del st, evid
    elif fix_sites:
        B_fix_sites = []
        R_fix_sites = []
        for i, B in enumerate(Ecoda + [[_] for _ in Ebulk]):
            A = np.zeros((Ne, len(B)))
            evid, st = event_station_pairs[i % len(event_station_pairs)]
            site_resp = fix_sites_params[freq_band][st]
            R_fix_sites.append(site_resp)
            B_fix_sites.append(np.ones(len(B)) * np.log(site_resp))
            idx = eventids.index(evid)
            A[idx, :] = 1
            As.append(A)
        B_fix_sites = np.hstack(B_fix_sites)
        R_fix_sites = R_fix_sites[:Ns]
        assert not np.any(np.isnan(B_fix_sites))
        del st, evid
    else:
        for i, B in enumerate(Ecoda + [[_] for _ in Ebulk]):
            A = np.ones((2, len(B)))
            As.append(A)
    A = np.hstack(As)
    if not fix:
        A[-1, :] = -np.hstack(tcoda + tbulk)
    A = np.transpose(A)

    # Define error function including the inversion for b, Ri and W
    record = []
    record_g0 = []
    recorded_g0 = set()
    max_record = plot_optimization_options.get('num', 7)
    nonlocal_ = {'warn': True}
    G_func = _load_func(G_plugin)

    def lstsq(g0, opt=False, b_fix=None, fix_sites=False):
        """Error for optimization of g0"""
        if opt and g0_bounds and not g0_bounds[0] <= g0 <= g0_bounds[1]:
            return np.inf
        Gcoda = []
        Gbulk = []
        for i, pair in enumerate(event_station_pairs):
            assert len(Ecoda[i]) > 0
            r = distances[pair]
            Gc = Gsmooth(G_func, r, tcoda[i], v0, g0, smooth=smooth,
                         smooth_window=smooth_window)
            Gcoda.append(Gc)
            if bulk_window:
                t1, t2 = tbulk_window[pair]
                tsup = np.linspace(t1, t2, num_points_integration)
                Gb = np.mean(G_func(r, tsup, v0, g0))
                Gbulk.append(Gb)
        E = np.hstack(Ecoda + Ebulk)
        G = np.hstack(Gcoda + Gbulk)
        B = np.log(E) - np.log(G)
        if b_fix:
            B = B + b_fix * np.hstack(tcoda + tbulk)
        if fix_sites:
            B = B - B_fix_sites
        if bulk_window:
            weights = np.hstack((np.ones(len(B) - len(weights_bulk)),
                                 weights_bulk))
        else:
            weights = 1
        if np.any(np.isinf(B)) and nonlocal_['warn']:
            nonlocal_['warn'] = False
            msg = ('%s: log(E/G) has infinite values. These values are droped.'
                   ' Probably G is smaller than machine precision.')
            log.warning(msg, pair)
        # scipy.linalg.lstsq can only be used for ordinary (unweighted) LES
        # C, _, _, _ = scipy.linalg.lstsq(A, B) (with C == results.params)
        wls = WLS(B, A, weights=weights, missing='drop')
        results = wls.fit()
        # err equals approx. (np.sum((B - np.dot(A, C)) ** 2) / len(B)) ** 0.5
        err = results.mse_resid ** 0.5
        C = results.params
        if fix:
            # intrinsic attenuation
            b = b_fix
            assert coda_normalization is None
        else:
            b = C[-1]
            if (b_bounds and not b_bounds[0] < b < b_bounds[1] or
                    g0_bounds and not g0_bounds[0] <= g0 <= g0_bounds[1]):
                err = np.inf
        if fix_sites:
            W = list(np.exp(C))
            R = R_fix_sites
        else:
            N1 = len(C) - Ne + fix
            N2 = len(C) - 1 + fix
            # spectral source energy of 1st ev
            W0 = np.exp(np.mean(C[:N1])) / R0
            # source energies of all events
            W = [W0] + list(np.exp(C[N1:N2]) * W0)
            if coda_normalization is None:
                R = np.exp(C[:N1]) / W0  # amplification factors
            else:
                R = np.ones(Ns)
        info = (tcoda, tbulk, Ecoda, Ebulk, Gcoda, Gbulk)
        if plot_optimization and opt:
            record_g0.append((err, g0))
        if (plot_optimization and g0 not in recorded_g0 and
                (len(record) < max_record - 1 or not opt)):
            recorded_g0.add(g0)
            record.append((err, g0, b, W, R, info))
        if opt:
            return err
        return err, g0, b, W, R, info

    if fix:
        # Invert with fixed g0 and b
        g0_fix, b_fix = fix_params[freq_band]
        if g0_fix is None or b_fix is None:
            msg = 'freq band (%.2f, %.2f): no g0 or b value present -> skip'
            log.warning(msg, *freq_band)
            return
        else:
            err, g0, b, W, R, info = lstsq(g0_fix, b_fix=b_fix,
                                           fix_sites=fix_sites)
            assert g0 == g0_fix
            assert b == b_fix
            msg = 'solved WLS with %d equations and %d unknowns, error: %.1e'
            log.debug(msg, A.shape[0], A.shape[1], err)
    elif optimize is None or optimize is False:
        g0_fix = np.mean(g0_bounds)
        err, g0, b, W, R, info = lstsq(g0_fix)
        msg = ('no optimization - solved WLS with %d equations and %d unknowns'
               ', error: %.1e')
        log.debug(msg, A.shape[0], A.shape[1], err)
    else:
        # Optimize g0, so that inversion yields minimal error
        if optimize is True:
            optimize = {}
        optimize = copy(optimize)
        optimize.setdefault('method', 'golden')
        optimize.setdefault('tol', 1e-8)
        if optimize['method'] in ('brent', 'golden'):
            optimize.setdefault('bracket', g0_bounds)
        elif optimize['method'] == 'bounded':
            optimize.setdefault('bounds', g0_bounds)
        opt = scipy.optimize.minimize_scalar(lstsq, args=(True,), **optimize)
        err, g0, b, W, R, info = lstsq(opt.x)
        msg = ('optimization solved WLS with %d equations and %d unknowns %d '
               'times, minimal error: %.1e')
        log.debug(msg, A.shape[0], A.shape[1], opt.nfev, err)
    if len(kwargs) > 0:
        log.error('unused kwargs: %s', json.dumps(kwargs))
    # Arrange result
    nstations = np.sum(~np.isnan(np.array(R, dtype='float')), axis=0).item()
    if np.isnan(nstations):
        nstations = 0
    R = OrderedDict([(st, Ri) for st, Ri in zip(stations, R)])
    W = OrderedDict([(evid, Wi) for evid, Wi in zip(eventids, W)])
    Rok = not fix_sites and coda_normalization is None
    Wok = coda_normalization is None
    result = {'g0': g0 if not fix else None,
              'b': b if not fix else None,
              'W': W if Wok else None,
              'R': R if Rok else None,
              'nstations': nstations,
              'error': err,
              'v0': v0}
    msg = 'freq band (%.2fHz, %.2fHz): optimization result is %s'
    log.debug(msg, freq_band[0], freq_band[1], json.dumps(sort_dict(result)))
    # Dump pkl files for external plotting
    if dump_optpkl or dump_fitpkl:
        import pickle
        eventid = energies[0].stats.eventid
        l = '%s_%05.2fHz-%05.2fHz' % (eventid, freq_band[0], freq_band[1])
        if dump_optpkl:
            with open(dump_optpkl % l, 'wb') as f:
                pickle.dump((record, record_g0, event_station_pairs), f, 2)
        if dump_fitpkl:
            with open(dump_fitpkl % l, 'wb') as f:
                t = (energies, g0, b, W, R, v0, info, smooth, smooth_window)
                pickle.dump(t, f, 2)
    # Optionally plot result of optimization routine
    label_eventid = (len(eventids) == 1)

    def fname_and_title(fname, evtotitle=False):
        title = 'filter: (%.2fHz, %.2fHz)' % freq_band
        evid = energies[0].stats.eventid if label_eventid else ''
        if label_eventid and evtotitle:
            title = 'event: %s  %s' % (evid, title)
        fname = fname.format(evid=evid, f1=freq_band[0], f2=freq_band[1])
        if not label_eventid:
            fname = fname.replace('__', '_')
        return fname, title

    try:
        if plot_energies and len(energies) > 0:
            pkwargs = copy(plot_energies_options)
            fname = pkwargs.pop(
                    'fname', 'energies_{evid}_{f1:05.2f}Hz-{f2:05.2f}Hz.png')
            fname, title = fname_and_title(fname)
            pkwargs.update({'bulk_window': bulkw, 'coda_window': codaw})
            from qopen.imaging import plot_energies
            plot_energies(energies, fname=fname, title=title, **pkwargs)
            log.debug('create energies plot at %s', fname)
        if plot_optimization and not fix and optimize:
            pkwargs = copy(plot_optimization_options)
            fname = pkwargs.pop(
                    'fname',
                    'optimization_{evid}_{f1:05.2f}Hz-{f2:05.2f}Hz.png')
            fname, title = fname_and_title(fname)
            from qopen.imaging import plot_optimization
            plot_optimization(record, record_g0, event_station_pairs,
                              fname=fname, title=title, **pkwargs)
            log.debug('create optimization plot at %s', fname)
        if plot_fits:
            pkwargs = copy(plot_fits_options)
            fname = pkwargs.pop('fname',
                                'fits_{evid}_{f1:05.2f}Hz-{f2:05.2f}Hz.png')
            fname, title = fname_and_title(fname)
            from qopen.imaging import plot_fits
            plot_fits(energies, g0, b, W, R, v0, info, G_func,
                      smooth=smooth, smooth_window=smooth_window,
                      title=title, fname=fname, **pkwargs)
            log.debug('create fits plot at %s', fname)
    except Exception:
        log.exception('error while creating a plot (invert_fb)')
    if (b_bounds and not 1.01 * b_bounds[0] < b < 0.99 * b_bounds[1] or
            g0_bounds and not 1.01 * g0_bounds[0] < g0 < 0.99 * g0_bounds[1]):
        msg = 'freq band (%.2f, %.2f): b=%.1e or g0=%.1e near bounds -> skip'
        log.warning(msg, *(freq_band + (b, g0)))
        return
    return tuple(result.values())
    #return g0, b, W, R, err, v0


def _remove_response(tr, evid, plot_remove_reponse, fname, **kw):
    if plot_remove_reponse:
        kw.setdefault('plot', fname.format(evid=evid, tr=tr))
    tr.remove_response(**kw)


def invert(events, inventory, get_waveforms,
           request_window, freqs, filter,
           rho0, vp=None, vs=None,
           remove_response=None,
           plot_remove_response=False, remove_response_options={},
           skip=None, use_picks=False,
           correct_for_elevation=False,
           subtract_local_depth=False,
           njobs=None,
           seismic_moment_method=None, seismic_moment_options={},
           plot_eventresult=False, plot_eventresult_options={},
           plot_eventsites=False, plot_eventsites_options={},
           plot_results=False, plot_results_options={},
           plot_sites=False, plot_sites_options={},
           plot_sds=False, plot_sds_options={},
           plot_mags=False, plot_mags_options={},
           cmd='go', input=None,
           coda_normalization=None,
           request_window_tolerance=0.5,
           **kwargs):
    """
    Qopen function to invert events and stations simultaneously

    :param events: is determined in :func:`invert_wrapper`
    :param inventory,get_waveforms: are determined in :func:`run`
        All other options are described in the example configuration file.
    :return: result dictionary
    """
    assert cmd in ('go', 'fixed', 'source')
    assert remove_response in (None, 'full', 'sensitivity')
    if coda_normalization is not None and cmd != 'go':
        raise ValueError('coda_normalization is only allowed for go command')
    msg = 'use %s cores for parallel computation'
    log.debug(msg, 'all available' if njobs is None else njobs)
    # Get origins and arrivals of event
    origins = {}
    event_dict = {}
    if use_picks:
        arrivals = {}
    for event in events:
        event_dict[get_eventid(event)] = event
        origins[get_eventid(event)] = get_origin(event)
        if use_picks:
            ar = get_arrivals(event)
            if ar is not None:
                arrivals[get_eventid(event)] = ar
    # Get frequencies
    freq_bands = get_freqs(**freqs)
    # Get stations
    channels = inventory.get_contents()['channels']
    stations = list(set(get_station(ch) for ch in channels))
    one_channel = {get_station(ch): ch for ch in channels}
    event_station_pairs = [(evid, sta) for evid in origins
                           for sta in stations]
    msg = '%d stations and %d events -> %s pairs'
    log.info(msg, len(stations), len(origins), len(event_station_pairs))
    # Start processing
    # Calculate distances and remove pairs with distance above threshold

    def _get_coordinates(station, time=None):
        return inventory.get_coordinates(one_channel[station], datetime=time)

    borehole_stations = set()

    @cache
    def _get_distance(evid, sta):
        ori = origins[evid]
        try:
            c = _get_coordinates(sta, time=ori.time)
        except Exception:
            raise SkipError('station not installed')
        args = (c['latitude'], c['longitude'], ori.latitude, ori.longitude)
        hdist = gps2dist_azimuth(*args)[0]
        local_elevation = c['elevation']
        if not subtract_local_depth:
            local_elevation = local_elevation + c['local_depth']
        vdist = (ori.depth + local_elevation * correct_for_elevation -
                 c['local_depth'])
        if c['local_depth'] > BOREHOLE_STATION_DEPTH:
            borehole_stations.add(sta)
        return np.sqrt(hdist ** 2 + vdist ** 2)

    distances = {}
    for pair in event_station_pairs[:]:
        try:
            distances[pair] = dist = _get_distance(*pair)
        except SkipError as ex:
            msg = '%s: %s -> skip pair'
            log.debug(msg, pair, str(ex))
            event_station_pairs.remove(pair)
            continue
        except Exception:
            msg = '%s: exception while determining distances -> skip pair'
            log.exception(msg, pair)
            event_station_pairs.remove(pair)
            continue
        if skip and 'distance' in skip:
            val = skip['distance']
            if val and dist / 1000 > val:
                msg = '%s: distance %.1fkm larger than %.1fkm -> skip pair'
                log.debug(msg, pair, dist / 1000, val)
                event_station_pairs.remove(pair)

    # Sort events by origin time and stations by distance
    event_station_pairs = sorted(
        event_station_pairs, key=lambda p: (origins[p[0]].time, distances[p]))
    msg = '%s pairs after distance selection'
    log.info(msg, len(event_station_pairs))
    log.debug('(%s)', event_station_pairs)

    # Calculate onsets
    def _get_onsets(evid, sta):
        ori = origins[evid]
        if use_picks:
            ons = get_picks(arrivals[evid], sta)
        else:
            ons = {'S': ori.time + _get_distance(evid, sta) / vs}
        if 'S' in ons and 'P' not in ons and vp is not None:
            ons['P'] = ori.time + _get_distance(evid, sta) / vp
        return ons

    try:
        onsets = {'P': {}, 'S': {}}
        for pair in event_station_pairs[:]:
            ons = _get_onsets(*pair)
            try:
                onsets['P'][pair] = ons['P']
                onsets['S'][pair] = ons['S']
            except KeyError:
                log.debug('%s: no pick/onset -> skip pair', pair)
                event_station_pairs.remove(pair)
    except SkipError as ex:
        msg = 'exception while determining onsets (%s) -> skip event'
        log.error(msg, str(ex))
        return
    except Exception:
        log.exception('exception while determining onsets -> skip event')
        return
    log.debug('origin station distances: %s', distances)
    log.debug('onsets: %s', onsets)
    if len(borehole_stations) > 0:
        msg = 'identified %d borehole stations: %s'
        borehole_stations = sorted(borehole_stations)
        log.debug(msg, len(borehole_stations), ' '.join(borehole_stations))

    # Check if enough pairs left
    if (len(event_station_pairs) == 0 or skip and
            len(event_station_pairs) <= skip.get('num_pairs', 0)):
        msg = ('only %d pairs left -> return')
        log.info(msg, len(event_station_pairs))
        return
    log.info('%s pairs with determined onsets/picks', len(event_station_pairs))
    log.debug('(%s)', event_station_pairs)

    # Retrieve data
    streams = []
    for pair in event_station_pairs[:]:
        evid, station = pair
        seedid = one_channel[station][:-1] + '?'
        net, sta, loc, cha = seedid.split('.')
        t1 = origins[evid].time + request_window[0]
        t2 = origins[evid].time + request_window[1]
        kwargs2 = {'network': net, 'station': sta, 'location': loc,
                   'channel': cha, 'starttime': t1, 'endtime': t2,
                   'event': event_dict[evid]}
        stream = get_waveforms(**kwargs2)
        # Check for gaps
        if stream:
            gaps = stream.get_gaps(min_gap=1)
            if len(gaps) > 0:
                msg = '%s: %d gaps longer than 1s detected -> skip pair'
                log.warning(msg, pair, len(gaps))
                stream = None
            else:
                stream.merge(method=1, fill_value='interpolate',
                             interpolation_samples=-1)
        # Check if data is complete
        if stream:
            for tr in stream:
                ct = _check_times(tr, (t1, t2), tol=request_window_tolerance)
                if ct:
                    msg = ('%s: data missing at one end of requested time '
                           'window, difference in seconds %s')
                    log.warning(msg, pair, ct)
                    stream = None
                    break
        if stream is None:
            event_station_pairs.remove(pair)
        elif len(stream) != 3:
            msg = ('%s: number of traces with channel %s is %s, '
                   'it should be 3 -> skip pair')
            log.warning(msg, pair, seedid, len(stream))
            event_station_pairs.remove(pair)
        else:
            for tr in stream:
                tr.stats.eventid = evid
                tr.stats.origintime = origins[evid].time
                tr.stats.ponset = onsets['P'][pair]
                tr.stats.sonset = onsets['S'][pair]
                tr.stats.distance = distances[pair]
            streams.append(stream)
    msg = 'succesfully fetched %d streams for %d stations and %d events'
    log.info(msg, len(streams), len(stations), len(origins))
    log.debug('(%s)', event_station_pairs)

    # Optionally remove instrument response
    if remove_response:
        remove_response_options = copy(remove_response_options)
        remove_response_fname = remove_response_options.pop(
            'fname', 'remove_response_{evid}_{tr.id}.png')
        if plot_remove_response:
            remove_response_options.pop('plot', None)
            path = os.path.dirname(remove_response_fname)
            if path != '':
                os.makedirs(path, exist_ok=True)
        for stream in streams[:]:
            pair = get_pair(stream[0])
            fail = stream.attach_response(inventory)
            if len(fail) > 0:
                msg = ('%s: no instrument response availlable for channel %s '
                       '-> skip pair')
                log.error(msg, pair, stream[0].id)
                streams.remove(stream)
                event_station_pairs.remove(pair)
                continue
            try:
                if remove_response == 'full':
                    for tr in stream:
                        _remove_response(
                            tr, evid, plot_remove_response,
                            remove_response_fname, **remove_response_options)
                    # if njobs == 1:
                    # else:
                    #     do_work = partial(
                    #         _remove_response,
                    #         evid=evid, plot=plot_remove_response,
                    #         fname=remove_response_fname,
                    #         **remove_response_options)
                    #     pool = multiprocessing.Pool(njobs)
                    #     rlist = pool.map(do_work, stream.traces)
                    #     pool.close()
                    #     pool.join()
                else:
                    for tr in stream:
                        sens = tr.stats.response.instrument_sensitivity
                        tr.data = tr.data / sens.value
            except Exception as ex:
                msg = ('%s: removing response/sensitivity failed (%s)'
                       '-> skip pair')
                log.error(msg, pair, ex)
                streams.remove(stream)
                event_station_pairs.remove(pair)
                continue
        msg = 'instrument correction (%s) finished for %d streams'
        log.info(msg, remove_response, len(streams))

    # Check if enough pairs left
    if len(streams) == 0 or skip and len(streams) <= skip.get('num_pairs', 0):
        msg = ('only %d pairs left -> return')
        log.info(msg, len(event_station_pairs))
        return

    # Calculate and cache filter width already here, just for the sake of nice
    # logging output
    srs = set()
    for stream in streams:
        srs.add(stream[0].stats.sampling_rate)

    for freqmin, freqmax in freq_bands.values():
        for sr in srs:
            if (freqmin + freqmax) > sr:
                continue
            filter_ = copy(filter)
            fu = {'freqmin': freqmin, 'freqmax': freqmax, 'type': 'bandpass'}
            if freqmax > 0.495 * sr:
                fu = {'freq': freqmin, 'type': 'highpass'}
            filter_.update(fu)
            filter_width(sr, **filter_)

    # Hack for undocumented option: usually station is termed
    # 'network.station'. But if this option is activated, the network part
    # is ignored in the following. This allows for stations operating in
    # different networks at different times to be considered as single
    # stations
    if kwargs.get('ignore_network_code'):
        for stream in streams:
            for tr in stream:
                tr.stats.network = ''
        stations = [st.split('.', 1)[1] for st in stations]
        stations = list(OrderedDict.fromkeys(stations))
    # Construct kwargs for invert_fb call
    kw = copy(kwargs)
    fix = cmd in ('fixed', 'source')
    fix_sites = cmd == 'source'
    kw.update({'rho0': rho0, 'borehole_stations': borehole_stations,
               'skip': skip, 'filter': filter,
               'fix': fix, 'fix_sites': fix_sites,
               'coda_normalization': coda_normalization,
               })
    if fix:
        # if fix_params is used, the inversion for station site responses and
        # energy source terms are done for fixed g0 and b from previous results
        fp = input
        if set(fp['freq']) != set(freq_bands.keys()):
            msg = ('Frequencies for fixed inversion have to be the same '
                   'as in the original inversion')
            raise ValueError(msg)
        kw['fix_params'] = {freq_bands[cfreq]: (g0f, bf) for cfreq, g0f, bf in
                            zip(fp['freq'], fp['g0'], fp['b'])}
    if fix_sites:
        fp = input
        kw['fix_sites_params'] = pars = {}
        for i, cfreq in enumerate(fp['freq']):
            fb = freq_bands[cfreq]
            pars[fb] = {sta: val[i] for sta, val in fp['R'].items()
                        if val[i] is not None}
    # Start invert_fb function
    if njobs == 1:
        # deepcopy only necessary for more than one freq band
        cond = len(freq_bands) > 1
        rlist = [invert_fb(fb, deepcopy(streams) if cond else streams, **kw)
                 for fb in freq_bands.values()]
    else:
        do_work = partial(invert_fb, streams=streams, **kw)
        pool = multiprocessing.Pool(njobs)
        rlist = pool.map(do_work, list(freq_bands.values()))
        pool.close()
        pool.join()

    # Check if any result
    if all([r is None for r in rlist]):
        log.warning('invert: no result for any frequency band')
        return
    # Re-sort results
    Rok = not fix_sites and coda_normalization is None
    Wok = coda_normalization is None
    result = defaultdict(list)
    if Rok:
        result['R'] = defaultdict(list)
    result['events'] = defaultdict(lambda: defaultdict(list))
    for (cfreq, freq_band), res in zip(freq_bands.items(), rlist):
        if res is None:
            msg = 'freq band (%.2f, %.2f): no result'
            log.debug(msg, *freq_band)
            g0opt, b, W, R, error, v0 = 6 * (None,)
            nstations = 0
        else:
            g0opt, b, W, R, nstations, error, v0 = res
        assert not fix or g0opt is None
        assert not fix or b is None
        assert Rok or R is None
        assert Wok or W is None
        result['freq'].append(cfreq)
        if not fix:
            result['g0'].append(g0opt)
            result['b'].append(b)
        # result['v0'].append(v0)
        if v0 is not None:
            result['v0'] = v0
        result['error'].append(error)
        result['nstations'].append(nstations)
        if Rok:
            for st in stations:
                if R is None:
                    result['R'][st].append(None)
                else:
                    result['R'][st].append(R.get(st))
        # result['W'].append(W)
        if Wok:
            for event in events:
                evid = get_eventid(event)
                if W is None:
                    result['events'][evid]['W'].append(None)
                else:
                    result['events'][evid]['W'].append(W.get(evid))
    # Calculate source properties sds, M0 and Mw
    if Wok:
        for event in events:
            evid = get_eventid(event)
            args = (result['freq'], result['events'][evid], result['v0'], rho0,
                    seismic_moment_method, seismic_moment_options,
                    get_magnitude(event_dict[evid]))
            result['events'][evid] = insert_source_properties(*args)
    if Rok:
        result['R'] = OrderedDict(result['R'])
    result['events'] = OrderedDict(result['events'])
    if ('freq' not in result or
                not fix and all([g0 is None for g0 in result['g0']])):
        log.info('no result for event')
        return
    result = sort_dict(result)
    msg = 'result is %s'
    log.debug(msg, json.dumps(result))
    # Optionally plot stuff
    if len(events) == 1:
        if plot_eventresult:
            kw = {'seismic_moment_method': seismic_moment_method,
                  'seismic_moment_options': seismic_moment_options}
            plot_eventresult_options.update(kw)
        try:
            _plot(result, eventid=get_eventid(event),
                  # v0=kwargs.get('v0'),
                  plot_eventresult=plot_eventresult,
                  plot_eventresult_options=plot_eventresult_options,
                  plot_eventsites=plot_eventsites,
                  plot_eventsites_options=plot_eventsites_options)
        except Exception:
            log.exception('error while creating a plot (invert)')
    return result


def invert_wrapper(events,
                   plot_results=False, plot_results_options={},
                   plot_sites=False, plot_sites_options={},
                   plot_sds=False, plot_sds_options={},
                   plot_mags=False, plot_mags_options={},
                   invert_events_simultaneously=False,
                   mean=None, noplots=False, **kwargs):
    """Qopen function for a list or Catalog of events

    Depending on 'invert_events_simultaneously' flag the function
    calls :func:`invert` for each event seperately or for all events once.
    In the first case mean results are calculated.

    :param events: is determined in :func:`run`.
        The rest of the options are described in the example configuration
        file.
    :return: result dictionary
    """
    # use non-interactive backend to circumvent problems with
    # parallel plotting on MacOS
    # see https://lserv.uni-jena.de/pipermail/seistools/2018/000006.html
    import matplotlib
    matplotlib.use('agg')
    # Sort events by origin time
    time_event_pairs = []
    for event in events:
        try:
            origin = get_origin(event)
        except IndexError:
            msg = 'event %s: no associated origin -> ignore event'
            log.error(msg, get_eventid(event))
            continue
        time_event_pairs.append((origin.time, event))
    events = list(zip(*sorted(time_event_pairs)))[1]
    # Start processing
    if invert_events_simultaneously:
        result = invert(events, **kwargs)
    else:
        result = {'events': OrderedDict()}
        for i, event in enumerate(events):
            evid = get_eventid(event)
            o = get_origin(event)
            mag = get_magnitude(event)
            mag = '%.1f' % mag if mag is not None else '?'
            msg = ('event %s (no %d of %d | %+.3f %+.3f %.1fkm M%s | %.19s): '
                   'start processing')
            log.info(msg, evid, i + 1, len(events), o.latitude, o.longitude,
                     o.depth / 1000, mag, o.time)
            res = invert([event], **kwargs)
            msg = 'event %s (no %d of %d): end processing'
            log.info(msg, evid, i + 1, len(events))
            if res:
                result['freq'] = res.pop('freq')
                if evid in res['events']:
                    res.update(res['events'].pop(evid))
                else:
                    assert kwargs.get('coda_normalization') is not None
                del res['events']
                result['events'][evid] = sort_dict(res)
        if len(result['events']) == 0:
            log.warning('invert_wrapper: no result')
            return
        col = collect_results(result, only=('g0', 'b', 'error', 'R'))
        if 'g0' in col and np.all(np.isnan(col['g0'])):
            log.warning('invert_wrapper: no result')
            return
        kw = {'axis': 0, 'robust': mean == 'robust',
              'weights': (1 / np.array(col['error']) if mean == 'weighted'
                          else None)}
        if 'g0' in col:
            result['g0'] = gmeanlist(col['g0'], **kw)
            result['b'] = gmeanlist(col['b'], **kw)
        result['error'] = gmeanlist(col['error'], **kw)
        if len(col['R']) > 0:
            result['R'] = OrderedDict()
            for st, Rst in col['R'].items():
                result['R'][st] = gmeanlist(Rst, **kw)
    result['config'] = {k: kwargs[k] for k in DUMP_CONFIG if k in kwargs}
    result['config'][
        'invert_events_simultaneously'] = invert_events_simultaneously
    result['config']['mean'] = mean
    result['config'] = sort_dict(result['config'], order=DUMP_CONFIG)
    result = sort_dict(result)
    # Optionally plot stuff
    try:
        if not noplots:
            _plot(result, plot_results=plot_results,
                  plot_results_options=plot_results_options,
                  plot_sites=plot_sites, plot_sites_options=plot_sites_options,
                  plot_sds=plot_sds, plot_sds_options=plot_sds_options,
                  plot_mags=plot_mags, plot_mags_options=plot_mags_options)
    except Exception:
        log.exception('error while creating a plot (invert_wrapper)')
    return result


def _plot(result, eventid=None, v0=None,
          plot_results=False, plot_results_options={},
          plot_sites=False, plot_sites_options={},
          plot_sds=False, plot_sds_options={},
          plot_mags=False, plot_mags_options={},
          plot_eventresult=False, plot_eventresult_options={},
          plot_eventsites=False, plot_eventsites_options={},
          **kwargs
          ):
    """Plotting helper function"""
#    M0_freq = M0_freq or result.get('config', {}).get('M0_freq')
    Wok = (any('W' in evres for evres in result.get('events', {}).values()) or
           'W' in result)
    if eventid is None:
        if plot_results and 'g0' in result:
            # only plot results if b key is in result
            # this is not the case for fixed and source command
            pkwargs = copy(plot_results_options)
            fname = pkwargs.pop('fname', 'results.pdf')
            from qopen.imaging import plot_results
            plot_results(result, fname=fname, **pkwargs)
            log.debug('create results plot at %s', fname)
        if plot_sites and 'R' in result:
            pkwargs = copy(plot_sites_options)
            fname = pkwargs.pop('fname', 'sites.pdf')
            from qopen.imaging import plot_sites
            plot_sites(result, fname=fname, **pkwargs)
            log.debug('create sites plot at %s', fname)
        if plot_sds and Wok:
            pkwargs = copy(plot_sds_options)
            fname = pkwargs.pop('fname', 'sds.pdf')
            from qopen.imaging import plot_all_sds
            plot_all_sds(result, fname=fname, **pkwargs)
            log.debug('create sds plot at %s', fname)
        if plot_mags and Wok:
            pkwargs = copy(plot_mags_options)
            fname = pkwargs.pop('fname', 'mags.pdf')
            from qopen.imaging import plot_mags
            plot_mags(result, fname=fname, **pkwargs)
            log.debug('create mags plot at %s', fname)
    else:
        if plot_eventresult:
            qu = ()
            if 'g0' in result:
                qu = qu + ('g0', 'lsc', 'Qsc', 'b', 'li', 'Qi')
            qu = qu + ('error',)
            if Wok:
                qu = qu + ('W', 'sds')
            pkwargs = copy(plot_eventresult_options)
            fname = pkwargs.pop('fname', 'eventresult_{evid}.pdf')
            fname = fname.format(evid=eventid)
            title = 'event {evid}'.format(evid=eventid)
            from qopen.imaging import plot_eventresult
            plot_eventresult(result, title=title, fname=fname,
                             quantities=qu, **pkwargs)
            log.debug('create eventresult plot at %s', fname)
        if plot_eventsites and 'R' in result:
            pkwargs = copy(plot_eventsites_options)
            fname = pkwargs.pop('fname', 'eventsites_{evid}.pdf')
            fname = fname.format(evid=eventid)
            title = 'event {evid}'.format(evid=eventid)
            from qopen.imaging import plot_eventsites
            plot_eventsites(result, title=title, fname=fname, **pkwargs)
            log.debug('create eventsites plot at %s', fname)


def _load_func(plugin):
    """Load and return function from Python module"""
    sys.path.append(os.path.curdir)
    modulename, funcname = plugin.split(':')
    module = import_module(modulename.strip())
    sys.path.pop(-1)
    func = getattr(module, funcname.strip())
    return func


def init_data(data, client_options=None, plugin=None, cache_waveforms=False,
              get_waveforms=None):
    """Return appropriate get_waveforms function

    See example configuration file for a description of the options"""
    client_module = None
    if get_waveforms is None:
        if client_options is None:
            client_options = {}
        try:
            client_module = import_module('obspy.clients.%s' % data)
        except ImportError:
            pass
        if client_module:
            Client = getattr(client_module, 'Client')
            client = Client(**client_options)

            def get_waveforms(event=None, **args):
                return client.get_waveforms(**args)
        elif data == 'plugin':
            get_waveforms = _load_func(plugin)
        else:
            if isinstance(data, str):
                from obspy import read
                stream = read(data)
            else:
                stream = data
            def get_waveforms(network, station, location, channel,
                              starttime, endtime, event=None):
                st = stream.select(network=network, station=station,
                                   location=location, channel=channel)
                st = st.slice(starttime, endtime)
                return st

    def wrapper(**kwargs):
        try:
            return get_waveforms(**kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as ex:
            seedid = '.'.join((kwargs['network'], kwargs['station'],
                               kwargs['location'], kwargs['channel']))
            msg = 'channel %s: error while retrieving data: %s'
            log.debug(msg, seedid, ex)

    use_cache = client_module is not None or data == 'plugin'
    use_cache = use_cache and cache_waveforms
    if use_cache:
        try:
            import joblib
        except ImportError:
            log.warning('install joblib to use cache_waveforms option')
        else:
            log.info('use waveform cache in %s', cache_waveforms)
            memory = joblib.Memory(cachedir=cache_waveforms, verbose=0)
            return memory.cache(wrapper)
    elif use_cache:
        log.warning('install joblib to use cache_waveforms option')
    return wrapper


class ConfigJSONDecoder(json.JSONDecoder):

    """Decode JSON config with comments stripped"""

    def decode(self, s):
        s = '\n'.join(l.split('#', 1)[0] for l in s.split('\n'))
        return super(ConfigJSONDecoder, self).decode(s)


def configure_logging(loggingc, verbose=0, loglevel=3, logfile=None):
    if loggingc is None:
        loggingc = deepcopy(LOGGING_DEFAULT_CONFIG)
        if verbose > 3:
            verbose = 3
        loggingc['handlers']['console']['level'] = LOGLEVELS[verbose]
        if logfile is None or loglevel == 0:
            del loggingc['handlers']['file']
            loggingc['loggers']['qopen']['handlers'] = ['console']
            loggingc['loggers']['py.warnings']['handlers'] = ['console']
        else:
            loggingc['handlers']['file']['level'] = LOGLEVELS[loglevel]
            loggingc['handlers']['file']['filename'] = logfile
    logging.config.dictConfig(loggingc)
    logging.captureWarnings(loggingc.get('capture_warnings', False))


def _load_json_results(args, name, raise_missing=True):
    fname = args.pop(name, None)
    if fname is not None:
        if isinstance(fname, str):
            with open(fname) as f:
                return json.load(f)
        else:
            return fname
    elif raise_missing:
        raise ParseError(f'Missing option: {name}')


def run(cmd='go',
        conf=None, pdb=False, tutorial=False,
        eventid=None,
        get_waveforms=None,
        print_mag=False,
        plots=None,
        **args):
    """Main entry point for a direct call from Python

    Example usage:

    >>> from qopen import run
    >>> run(conf='conf.json')

    :param args: All args correspond to the respective command line and
        configuration options.
        See the example configuration file for help and possible arguments.
        Options in args can overwrite the configuration from the file.
        E.g. ``run(conf='conf.json', events=my_catalogue)`` will ignore
        ``events`` value in the configuration file.

        Exceptions from the description in configuration file:
    :param events: can be filename or ObsPy Catalog object
    :param inventory: can be filename or ObsPy Inventory object
    :param get_waveforms: function, if given the data option will be ignored.
        get_waveforms will be called as described in the example configuration
        file
    :return: result dictionary
    """
    if cmd not in ('create', 'go', 'fixed', 'source',
                   'recalc_source', 'plot', 'rt'):
        raise ValueError(f'Unkown command {cmd}')
    time_start = time.time()
    if pdb:
        import traceback
        import pdb

        def info(type, value, tb):
            traceback.print_exception(type, value, tb)
            print()
            pdb.pm()

        sys.excepthook = info
    if conf in ('None', 'none', 'null', ''):
        conf = None
    # Copy example files if create_config or tutorial
    if cmd == 'create':
        if conf is None:
            conf = 'conf.json'
        srcs = ['conf.json']
        dest_dir = os.path.dirname(conf)
        dests = [conf]
        if tutorial:
            example_files = ['example_events.xml', 'example_inventory.xml',
                             'example_data.mseed']
            srcs.extend(example_files)
            for src in example_files:
                dests.append(os.path.join(dest_dir, src))
        for src, dest in zip(srcs, dests):
            src = resource_filename('qopen', 'example/%s' % src)
            shutil.copyfile(src, dest)
        return
    # Parse config file
    if conf:
        try:
            with open(conf) as f:
                conf = json.load(f, cls=ConfigJSONDecoder)
        except ValueError as ex:
            print('Error while parsing the configuration: %s' % ex)
            return
        except IOError as ex:
            print(ex)
            return
        # Populate args with conf, but prefer args
        conf.update(args)
        args = conf
    wm = 'mean'
    if wm in args:
        args.get('plot_results_options', {}).setdefault(wm, args[wm])
        args.get('plot_sites_options', {}).setdefault(wm, args[wm])
    args['cmd'] = cmd
    # Configure logging
    kw = {'loggingc': args.pop('logging', None),
          'verbose': args.pop('verbose', 0),
          'loglevel': args.pop('loglevel', 3),
          'logfile': args.pop('logfile', None)}
    prefix = args.pop('prefix', None)
    if prefix:
        prefix_path = os.path.dirname(prefix)
        if prefix_path != '':
            os.makedirs(prefix_path, exist_ok=True)
    if isinstance(kw['logfile'], str) and prefix:
        kw['logfile'] = prefix + kw['logfile']
    output = args.pop('output', None)
    indent = args.pop('indent', None)
    plottargets = ['energies', 'optimization', 'fits', 'eventresult',
               'eventsites', 'results', 'sites', 'sds', 'mags',
               'remove_response']
    if prefix:
        if output is not None:
            output = prefix + output
        for t in plottargets:
            key = 'plot_%s_options' % t
            if t == 'remove_response':
                key = '%s_options' % t
            if key in args and 'fname' in args[key]:
                args[key]['fname'] = prefix + args[key]['fname']
    if plots is not None:
        assert isinstance(plots, bool)
        for t in plottargets:
            args['plot_' + t] = plots

    # Optionally plot
    if cmd == 'plot':
        result_plot = _load_json_results(args, 'input')
        if conf:
            result_plot['config'] = args
        else:
            result_plot['config'].update(args)
            args = result_plot['config']
        if eventid is not None:
            result_plot['events'] = {eventid: result_plot['events'][eventid]}
        _plot(result_plot, eventid=eventid, **args)
        return

    configure_logging(**kw)
    log.info('Qopen version %s', qopen.__version__)
    align_sites = args.pop('align_sites', False)
    load_inv = cmd != 'recalc_source' or align_sites
    load_all = cmd != 'recalc_source'
    try:
        if load_inv:
            # Read inventory
            inventory = args.pop('inventory')
            filter_inventory = args.pop('filter_inventory', None)
            if not isinstance(inventory, obspy.Inventory):
                if isinstance(inventory, str):
                    format_ = None
                else:
                    inventory, format_ = inventory
                inventory = obspy.read_inventory(inventory, format_)
                channels = inventory.get_contents()['channels']
                stations = list(set(get_station(ch) for ch in channels))
                log.info('read inventory with %d stations', len(stations))
            if filter_inventory:
                inventory = inventory.select(**filter_inventory)
                channels = inventory.get_contents()['channels']
                stations = list(set(get_station(ch) for ch in channels))
                log.info('filter inventory with %d stations', len(stations))
        if load_all:
            # Read events
            events = args.pop('events')
            filter_events = args.pop('filter_events', None)
            resolve_seedid = args.pop('resolve_seedid', False)
            read_events_kwargs = args.pop('read_events_kwargs', {})
            if (isinstance(events, (tuple, list)) and
                    isinstance(events[0], str)):
                events, format_ = events
                read_events_kwargs['format'] = format_
            if isinstance(events, str):
                if resolve_seedid:
                    read_events_kwargs['resolve_seedid'] = resolve_seedid
                resolve_seedid = read_events_kwargs.pop('resolve_seedid', False)
                if resolve_seedid:
                    read_events_kwargs['inventory'] = inventory
                events = obspy.read_events(events, **read_events_kwargs)
                log.info('read %d events', len(events))
            if filter_events:
                events = events.filter(*filter_events)
                log.info('filter %d events', len(events))
            # Initialize get_waveforms
            keys = ['data', 'client_options', 'plugin', 'cache_waveforms']
            tkwargs = {k: args.pop(k, None) for k in keys}
            get_waveforms = init_data(get_waveforms=get_waveforms,
                                      **tkwargs)
            if tkwargs['data'] is not None:
                log.info('init data from %s', tkwargs['data'])
    except Exception:
        log.exception('cannot read events/stations or initalize data')
        return
    # Optionally select event
    if eventid:
        elist = [ev for ev in events if get_eventid(ev) == eventid]
        if len(elist) == 0:
            msg = ('Did not find any event with id %s.\n'
                   'Example id from file: %s')
            raise ParseError(msg % (eventid, str(events[0].resource_id)))
        log.debug('use only event with id %s', eventid)
        events = obspy.Catalog(elist)
    # Start main routine with remaining args
    log.info('Use Qopen command %s', cmd)
    log.debug('start qopen routine with parameters %s', json.dumps(args))
    if align_sites and cmd == 'source':
        msg = 'align sites not valid for command source -> set to False'
        log.warning(msg)
        align_sites = False
    if align_sites:
        align_sites_kw = {
            'seismic_moment_method': args.pop('seismic_moment_method', None),
            'seismic_moment_options': args.pop('seismic_moment_options', None),
            'station': args.pop('align_sites_station', None),
            'response': args.pop('align_sites_value', 1)}
        if align_sites_kw['station'] and ',' in align_sites_kw['station']:
            align_sites_kw['station'] = align_sites_kw['station'].split(',')
    if load_inv:
        args['inventory'] = inventory
    if load_all:
        args['get_waveforms'] = get_waveforms
        args['events'] = events
    if cmd in ('go', 'fixed', 'source'):
        if cmd in ('fixed', 'source'):
            args['input'] = _load_json_results(args, 'input')
            if 'input_sites' in args and cmd == 'source':
                input_sites = _load_json_results(args, 'input_sites')
                args['input']['R'] = input_sites['R']
        # main inversion
        result = invert_wrapper(noplots=align_sites, **args)
        # Output and return result
        log.debug('inversion results: %s', json.dumps(result))
    elif cmd == 'recalc_source':
        result = _load_json_results(args, 'input')
    if align_sites:
        msg = 'align station site responses and calculate source spectra'
        log.info(msg)
        align_site_responses(result, **align_sites_kw)
        result.setdefault('config', {}).update(align_sites_kw)
        log.debug('results after alignment of site responses: %s',
                  json.dumps(result))
        _plot(result, eventid=eventid, **args)
    elif cmd == 'recalc_source':
        log.info('recalculate source spectra')
        kw = {'seismic_moment_method': args.pop('seismic_moment_method', None),
              'seismic_moment_options': args.pop('seismic_moment_options',
                                                 None)}
        calculate_source_properties(result, **kw)
        result.setdefault('config', {}).update(kw)
        _plot(result, eventid=eventid, **args)
    if output == 'stdout':
        print(json.dumps(result))
    elif output is not None and output.lower() not in ('none', 'null', ''):
        path = os.path.dirname(output)
        if path != '':
            os.makedirs(path, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(result, f, indent=indent)
    if print_mag and 'events' in result:
        for evid, evres in result['events'].items():
            print(evid,
                  '{:.2f}'.format(evres['Mw']) if 'Mw' in evres else 'nan')
    time_end = time.time()
    log.debug('used time: %.1fs', time_end - time_start)
    return result


def _add_bool_argument(parser, feature, help=None, help2=None):
    group = parser.add_mutually_exclusive_group(required=False)
    dest = feature.replace('-', '_')
    group.add_argument('--' + feature, dest=dest,
                        action='store_true', default=SUPPRESS, help=help)
    group.add_argument('--no-' + feature, dest=dest,
                        action='store_false', default=SUPPRESS, help=help2)


def run_cmdline(args=None):
    """Main entry point from the command line"""
    # Define command line arguments

    from qopen import __version__
    version = '%(prog)s ' + __version__
    msg = ('Qopen: Seperation of intrinsic and scattering Q by envelope '
           'inversion')
    epilog = 'To get help on a Qopen command run: qopen command -h'
    mainp = argparse.ArgumentParser(description=msg, epilog=epilog)
    mainp.add_argument('--version', action='version', version=version)
    sub = mainp.add_subparsers(title='Qopen commands', dest='cmd')

    msg = ('Create example configuration in specified file '
           '(default: conf.json)')
    p1 = sub.add_parser('create', help=msg, description=msg)
    msg = ('Estimate intrinsic attenuation and scattering strength, '
           'site responses, event spectra (including source parameters) '
           'by inversion of envelopes')
    p2 = sub.add_parser('go', help=msg, description=msg)
    msg = ('Estimate site responses and event spectra '
           '(including source parameters) with fixed attenuation parameters '
           '(g0, b) by inversion of envelopes')
    desc = msg + '. Specify JSON file with attenuation results with --input.'
    p3 = sub.add_parser('fixed', help=msg, description=desc)
    msg = ('Estimate event spectra and derive source parameters, e.g. '
           'moment magnitude, with fixed attenuation parameters (g0, b) and '
           'fixed site responses by inversion of envelopes')
    desc = msg + (
            '. Specify JSON file with attenuation results and site responses '
            'with --input, if site responses should be taken from a different '
            'file, specify it with --input-sites.')
    p4 = sub.add_parser('source', help=msg, description=desc)
    msg = ('Derive source parameters from source spectra without '
           'new inversion '
           '(possibly changed configuration, e.g. seismic_moment_options)')
    desc = msg + (
            ', specify JSON file with source spectra with --input. '
            'If align-sites is turned on, the JSON file has to contain '
            'site responses, too.')
    p5 = sub.add_parser('recalc_source', help=msg, description=desc)
    msg = ('Replot results. Can be used '
           'together with -e to plot event results')
    desc = msg + (
            '. Specify JSON file with reults with --input.')
    p6 = sub.add_parser('plot', help=msg, description=desc)
    msg = ("Calculate or plot spectral energy densitiy Green's functions, "
           "used in the above inversions, mainly based on radiative transfer")
    from qopen.rt import __doc__ as rtdoc, create_parser
    p7 = sub.add_parser('rt', help=msg, description=rtdoc,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    create_parser(p7)

    msg = 'additionally create data files for working example'
    p1.add_argument('--tutorial', help=msg, action='store_true',
                    default=SUPPRESS)
    msg = 'Configuration file to create (default: conf.json)'
    p1.add_argument('-c', '--conf', default='conf.json', help=msg)
    for p in (p2, p3, p4, p5, p6):
        msg = 'Configuration file to load (default: conf.json)'
        p.add_argument('-c', '--conf', default='conf.json', help=msg)
        msg = 'Set chattiness on command line. Up to 3 -v flags are possible'
        p.add_argument('-v', '--verbose', help=msg, action='count',
                       default=SUPPRESS)
        msg = 'Start the debugger upon exception'
        p.add_argument('--pdb', action='store_true', help=msg)
        msg = 'Process only event with this id'
        p.add_argument('-e', '--eventid', help=msg, default=SUPPRESS)
        msg = 'Add prefix for all output files defined in config'
        p.add_argument('--prefix', help=msg, default=SUPPRESS)
    for p in (p2, p3, p5):
        msg = 'Align site responses and correct source spectra'
        _add_bool_argument(p, 'align-sites', help=msg)
        msg = ('Site response of this station is fixed, it is possible to '
               'sepcify more than one station with "," delimiter '
               '(default: product of station site responses is fixed)')
        p.add_argument('--align-sites-station', help=msg, default=SUPPRESS)
        msg = ('Value of site response for specified station or product of '
               'station site responses (default: 1)')
        p.add_argument('--align-sites-value', help=msg, type=float,
                       default=SUPPRESS)
    p2.add_argument('--dump-optpkl', default=SUPPRESS, help=SUPPRESS)
    for p in (p2, p3, p4):
        p.add_argument('--dump-fitpkl', default=SUPPRESS, help=SUPPRESS)
    for p in (p2, p3, p4, p5):
        msg = ('Use these flags to overwrite values in the config file. '
               'See the example configuration file for a description of '
               'these options. Options representing dictionaries or lists are '
               'expected to be valid JSON. --overwrite-conf allows to set '
               'config options not listed here using a JSON dictionary.')
        features_str = ('events', 'inventory', 'data', 'output',
                        'input', 'input-sites',
                        'seismic-moment-method')
        features_json = ('filter-events', 'filter-inventory',
                         'seismic-moment-options')
        features_bool = ('resolve-seedid', 'invert-events-simultaneously',
                         'print-mag')
        features_plot = ('energies', 'optimization', 'fits',
                         'eventresult', 'eventsites', 'results', 'sites',
                         'sds', 'mags')
        g3 = p.add_argument_group('optional qopen arguments', description=msg)
        for f in features_str:
            g3.add_argument('--' + f, default=SUPPRESS)
        for f in features_json:
            g3.add_argument('--' + f, default=SUPPRESS, type=json.loads)
        g3.add_argument('--njobs', default=SUPPRESS, type=int)
        for f in features_bool:
            _add_bool_argument(g3, f)
        g3.add_argument('--overwrite-conf', default=SUPPRESS, type=json.loads)
    p6.add_argument('--input', default=SUPPRESS)
    for p in (p2, p3, p4, p5, p6):
        msg = ('The plot configuration can be overridden with '
               'the following options: '
               '--plot-{type}, --no-plot-{type}, --plot-{type}-options '
               'in which {type} is one of %s' % (features_plot,))
        g4 = p.add_argument_group('optional qopen plot arguments',
                                  description=msg)
        _add_bool_argument(g4, 'plots', help='turn all plots on',
                   help2='turn all plots off')
        _add_bool_argument(g4, 'plot-remove-response',
                           help='plot response removal',
                           help2=argparse.SUPPRESS)
        for f in features_plot:
            _add_bool_argument(g4, 'plot-' + f, help=argparse.SUPPRESS,
                               help2=argparse.SUPPRESS)
            g4.add_argument(f'--plot-{f}-options', default=SUPPRESS,
                            type=json.loads, help=argparse.SUPPRESS)


    # Get command line arguments and start run function
    args = mainp.parse_args(args)
    if args.cmd == 'rt':
        from qopen.rt import main
        main(args)
    else:
        args = vars(args)
        args.update(args.pop('overwrite_conf', {}))
        try:
            run(**args)
        except ParseError as ex:
            p.error(ex)
