# coding=utf-8
# Author: Tom Eulenfeld
"""
Qopen
=====

Seperation of intrinsic and scattering **Q** by envel\ **ope** inversio\ **n**

Determining scattering parameters and seismic moment.

Based partly upon Sens-Schönfelder and Wegler (2006).

run_cmdline -> run -> invert_wrapper -> invert -> invert_fb
"""
# The following lines are for Py2/Py3 support with the future module.
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (  # analysis:ignore
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super, map, zip)

import argparse
from argparse import SUPPRESS
from collections import defaultdict, OrderedDict
from copy import copy, deepcopy
from importlib import import_module
import json
import logging
import logging.config
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
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.robust.robust_linear_model import RLM

from qopen.util import (cache, gmean, smooth as smooth_, smooth_func,
                        LOGGING_DEFAULT_CONFIG)

try:
    import joblib
    from joblib import Parallel, delayed
except ImportError:
    joblib = None

IS_PY3 = sys.version_info.major == 3

log = logging.getLogger('qopen')
log.addHandler(logging.NullHandler())

LOGLEVELS = {0: 'CRITICAL', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}

DUMP_CONFIG = ['invert_events_simultaniously', 'mean',
               'v0', 'rho0', 'R0', 'free_surface',
               'freqs', 'filter', 'optimize', 'g0_bounds', 'b_bounds',
               'seismic_moment_method', 'seismic_moment_options',
               'bulk_window', 'coda_window', 'noise_windows',
               'weight', 'remove_noise',
               'adjust_sonset', 'adjust_sonset_options',
               'remove_response', 'correct_for_elevation', 'skip',
               'G_module']

DUMP_ORDER = ['M0', 'Mw', 'Mcat', 'fc', 'n', 'gamma', 'freq', 'g0', 'b',
              'error', 'W', 'omM', 'R', 'events', 'v0', 'config']

DUMP_PKL = False


class CustomError(Exception):
    pass


class ParseError(Exception):
    pass


def sort_dict(dict_, order=DUMP_ORDER):
    return OrderedDict(sorted(dict_.items(), key=lambda t: order.index(t[0])))


def linear_fit(y, x, m=None, method='robust'):
    """Linear fit between x and y

    :param x, y: data
    :param m: fix slope at specific value
    :param method: one of ('least_squares', 'robust')
    :return: slope a and intercept b of y = ax + b
    """
    Model = RLM if method == 'robust' else OLS
    if m is None:
        X = np.empty((len(y), 2))
        X[:, 0] = x
        X[:, 1] = 1
        res = Model(y, X).fit()
        return res.params
    else:
        X = np.ones(len(y))
        res = Model(np.array(y) - m * np.array(x), X).fit()
        return m, res.params[0]


@cache
def filter_width(sr, freq=None, freqmin=None, freqmax=None, corners=2,
                 zerophase=False, type='bandpass'):
    """Integrate over the squared filter response of a Butterworth filter

    The result corresponds to the filter width, which equals approximately
    the difference of the corner frequencies. The energy density should
    be divided by the result to get the correct energy spectral density.

    :param sr: sampling rate
    :param freq: corner frequencies of low- or highpass filter
    :param freqmin, freqmax: corner frequencies of bandpass filter
    :param corners: number of corners
    :param zerophase: if True number of corners are doubled
    :param type: 'bandpass', 'highpass' or 'lowpass'
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

    See example configuration file."""
    if cfreqs is None and fbands is None:
        max_exp = int(np.log(max / min) / step / np.log(2))
        exponents = step * np.arange(max_exp + 1)[::-1]
        cfreqs = max / 2 ** exponents
    if fbands is None:
        df = np.array(cfreqs) * (2 ** width - 1) / (2 ** width + 1)
        fbands = OrderedDict((f, (f - d, f + d)) for d, f in zip(df, cfreqs))
    else:
        fbands = sorted(fbands)
        fbands = OrderedDict((0.5 * (f1 + f2), (f1, f2)) for f1, f2 in fbands)
    msg = 'central frequencies: (' + '%s, ' * (len(cfreqs) - 1) + '%s)'
    log.info(msg, *cfreqs)
    msg = ('freq bands: ' + '(%.3f, %.3f), ' * len(cfreqs))[:-2]
    log.info(msg, *np.array(sorted(fbands.values())).flat)
    return fbands


def energy1c(data, rho, df, fs=4):
    """Energy density of one channel

    :param data: velocity data (m/s)
    :param rho: density (kg/m**3)"""
    hilb = scipy.fftpack.hilbert(data)
    return rho * (data ** 2 + hilb ** 2) / 2 / df / fs


def observed_energy(stream, rho, df, fs=4, tolerance=1):
    """Return trace with total energy density of three component stream

    :param stream: stream of a 3 component seismogram
    :param rho: energy density (kg/m**3)
    :param df: filter width in Hz
    :param fs: free surface correction (default: 4)
    :param tolerance: the number of samples the length of the traces
        in the 3 component stream may differ (default: 1)"""
    data = [energy1c(tr.data, rho, df, fs=fs) for tr in stream]
    Ns = [len(d) for d in data]
    if max(Ns) - min(Ns) > tolerance:
        msg = ('traces for one stream have different lengths %s. Tolerance '
               ' is %d samples') % (Ns, tolerance)
        raise CustomError(msg)
    elif max(Ns) - min(Ns) > 0:
        data = [d[:min(Ns)] for d in data]
    data = np.sum(data, axis=0)
    tr = obspy.Trace(data=data, header=stream[0].stats)
    tr.stats.channel = tr.stats.channel[:2] + 'X'
    return tr


def sds(W, f, v, rho):
    """
    Calculate source displacement spectrum ωM from spectral source energy W

    according to Sato & Fehler (2012, p.188)
    :param W: spectral source energy (J/Hz)
    :param f, v, rho: frequency, mean velocity, mean density
    :return: source displacement spectrum (in Nm)"""
    return np.sqrt(W * 2.5 / np.pi * rho * v ** 5 / f ** 2)


def source_model(freq, M0, fc, n=2, gamma=1):
    """Model for source displacement spectrum

    :param freq: frequencies
    :param M0: seismic moment (Nm)
    :param fc: corner frequency
    :param n: high frequency fall-of
    :param gamma: corner sharpness
    """
    return M0 * (1 + (freq / fc) ** (n * gamma)) ** (-1 / gamma)


def _source_model_ab(freq, M0, fc, a=2, b=1):
    return M0 * (1 + (freq / fc) ** a) ** (-b)


def fit_sds(freq, omM, method='mean', fc=None, n=2, gamma=1,
            fc_lim=None, n_lim=(0.5, 10), gamma_lim=(0.5, 10),
            fc0=10, n0=2, gamma0=1, fall_back=5, num_points=None, **opt_kw):
    """Calculate seismic moment M0 from source displacement spectrum

    :param freq, omM: frequencies, source displacement spectrum (same length)
    :param method: 'mean' - take mean of sds of frequencies below fc,
        'fit', 'robust_fit' - fit source model to obtain M0.
        If one or more of fc, n, gamma are None, M0 and these values are
        simultaneously determined.
        Robust version uses a robust linear model (which downweights outliers).
    :param fc, n, gamma: corner frequency and coefficients for source model
    :param fc_lim, gamma_lim: bounds for corner frequency and gamma
        (used for optimization if respective variable is set to None)
    :param fc0, gamma0: starting values of fc and gamma for optimization
        (only used for optimization for fc and gamma)
    :param fall_back: use robust fit only if number of data points >= fall_back
    :param num_points: determine M0 only if number of data points >= num_points
    All other kwargs are passed to scipy.optimization, e.g.
        :param tol: tolerance for optimization
    :return: dictionary with M0 and optimized variables fc, n, and gamma
        if applicable.
        If M0 is not determined the function will return None
    """
    if method == 'mean':
        if fc is None:
            msg = ("Border frequency fc must be given for "
                   "seismic_moment_method 'mean'")
            raise ValueError(msg)
        M0 = [o for f, o in zip(freq, omM) if f < fc and o is not None]
        if num_points is not None and len(M0) < num_points:
            return
        if len(M0) > 0:
            return {'M0': gmean(M0)}
    elif method in ('fit', 'robust_fit'):
        omM = np.array(omM, dtype=float)
        freq = np.array(freq)[~np.isnan(omM)]
        omM = omM[~np.isnan(omM)]
        if len(freq) == 0 or num_points is not None and len(freq) < num_points:
            return
        if method == 'robust_fit' and len(freq) >= fall_back:
            Model = RLM
        else:
            Model = OLS

        def lstsq(fc, n, gamma, opt=False):
            # Inversion for M0
            model = source_model(freq, 1, fc, n, gamma)
            y = np.log(omM) - np.log(model)
            X = np.ones(len(y))
            res = Model(y, X).fit()
            if opt:
                return np.mean(res.resid ** 2)
            return {'M0': np.exp(res.params[0])}

        def lstsqab(fc, a, opt=False):
            # Inversion for M0 and b
            model = _source_model_ab(freq, 1, fc, a, 1)
            y = np.log(omM)
            X = np.empty((len(y), 2))
            X[:, 0] = 1
            X[:, 1] = np.log(model)
            res = Model(y, X).fit()
            if opt:
                return np.mean(res.resid ** 2)
            return {'M0': np.exp(res.params[0]), 'b': res.params[1]}

        unknowns = ((fc is None) * ('fc',) +
                    (n is None) * ('n',) + (gamma is None) * ('gamma',))
        if n is None and gamma is None:
            unknowns = (fc is None) * ('fc',) + ('a',)
        wrapper = {
            'fc': lambda x, opt=False: lstsq(x, n, gamma, opt=opt),
            'n': lambda x, opt=False: lstsq(fc, x, gamma, opt=opt),
            'gamma': lambda x, opt=False: lstsq(fc, n, x, opt=opt),
            'fcn': lambda x, opt=False: lstsq(x[0], x[1], gamma, opt=opt),
            'fcgamma': lambda x, opt=False: lstsq(x[0], n, x[1], opt=opt),
            'a': lambda x, opt=False: lstsqab(fc, x, opt=opt),
            'fca': lambda x, opt=False: lstsqab(x[0], x[1], opt=opt),
        }
        a_lim = None
        if n_lim and gamma_lim:
            a_lim = [n_lim[0] * gamma_lim[0], n_lim[1] * gamma_lim[1]]
        bounds = {'fc': fc_lim or (freq[0], freq[-1]), 'n': n_lim,
                  'gamma': gamma_lim, 'a': a_lim}
        start = {'fc': fc0, 'n': n0, 'gamma': gamma0, 'a': gamma0 * n0}

        result = {}
        if len(unknowns) == 0:
            return lstsq(fc, n, gamma)
        elif len(unknowns) == 1 and len(freq) > 1:
            optimize = scipy.optimize.minimize_scalar
            x = unknowns[0]
            lstsq2 = wrapper[x]
            opt = optimize(lstsq2, args=(True,), bounds=bounds[x],
                           method='bounded', **opt_kw)
            result = {x: opt.x}
            result.update(lstsq2(opt.x))
        elif len(freq) > len(unknowns) >= 2:
            optimize = scipy.optimize.minimize
            lstsq2 = wrapper[''.join(unknowns)]
            bounds = [bounds[u] for u in unknowns]
            x0 = [start[u] for u in unknowns]
            opt = optimize(lstsq2, x0, args=(True,), bounds=bounds, **opt_kw)
            result = {u: opt.x[i] for i, u in enumerate(unknowns)}
            result.update(lstsq2(opt.x))
            msg = 'Optimization for M0 and %s terminated because of %s'
            log.debug(msg, unknowns, opt.message.lower())
        if 'a' in result:
            a = result.pop('a')
            b = result.pop('b')
            result['gamma'] = 1 / b
            result['n'] = a * b
        return result


def moment_magnitude(M0):
    """Moment magnitude Mw from seismic moment M0"""
    return 2 / 3 * np.log10(M0) - 6.07


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
    """Arrivals of appropriate event"""
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
        pick = arrival.pick_id.getReferredObject()
        seedid = pick.waveform_id.getSEEDString()
        if station == get_station(seedid):
            if phase in picks:
                msg = '%s, %s-onset: multiple picks'
                raise CustomError(msg % (station, phase))
            picks[phase] = pick.time
    return picks


def time2utc(time, trace, after=None):
    """Convert string with time information to UTC object

    :param time: can be one of:
         "OT+-???s" seconds relative to origin time
         "P+-???s" seconds relative to P-onset
         "S+-???s" seconds relative to S-onset
         "???Ptt" travel time relative to P-onset travel time
         "???Stt" travel time relative to S-onset travel time
         "???SNR" time after which SNR falls below this value
                  (after time given in after)
         "time>???SNR" time after which SNR falls below this value
                  (after time given in front of expression)
    :param trace: Trace object with stats entries
    :param after: UTC object for SNR case.
    """
    ot = trace.stats.origintime
    p = trace.stats.ponset
    s = trace.stats.sonset
    time = time.lower()
    if time.endswith('snr'):
        if '>' in time:
            time1, time = time.split('>')
            if time1 != '':
                after = time2utc(time1, trace)
        assert after is not None
        tr = trace.slice(starttime=after)
        snr = float(time[:-3])
        noise_level = tr.stats.noise_level
        try:
            index = np.where(tr.data < snr * noise_level)[0][0] - 1
        except IndexError:
            index = len(tr.data) - 1
        t = tr.stats.starttime + index * tr.stats.delta
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

    :param tw: tuple of two values, both can be a string (see :func: time2utc)
        or a list of strings in which case the latest starttime and earliest
        endtime is taken.
    :param trace: Trace object with stats entries
    """
    ret = []
    for val in tw:
        after = ret[0] if len(ret) > 0 else None
        if isinstance(val, (list, tuple)):
            times = [time2utc(v, trace, after=after) for v in val]
            if len(ret) == 0:
                t = max(times)
            else:
                t = min(times)
        else:
            t = time2utc(val, trace, after=after)
        ret.append(t)
    return ret


def collect_results(results):
    """
    Collect g0, b, error, R, W, eventids and v0 from results of multiple events
    """
    if 'g0' not in list(results['events'].items())[0][1]:
        g0 = [results['g0']]
        b = [results['b']]
        error = [results['error']]
        v0 = [results.get['v0']]
        R = {sta: [Rsta] for sta, Rsta in results['R'].items()}
        W, eventids = [], []
        for eventid, res in results['events'].items():
            W.append(res['W'])
            eventids.append(eventid)
    else:
        g0, b, error, R, W, eventids = [], [], [], defaultdict(list), [], []
        v0 = []
        for eventid, res in results['events'].items():
            if res is None:
                continue
            g0.append(res['g0'])
            b.append(res['b'])
            error.append(res['error'])
            W.append(res['W'])
            eventids.append(eventid)
            v0.append(res.get('v0'))
            for sta, Rsta in res['R'].items():
                R[sta].append(Rsta)
        R = dict(R)
    g0 = np.array(g0, dtype=np.float)
    b = np.array(b, dtype=np.float)
    error = np.array(error, dtype=np.float)
    W = np.array(W, dtype=np.float)
    v0 = np.array(b, dtype=np.float)
    for sta in R:
        R[sta] = np.array(R[sta], dtype=np.float)
    return g0, b, error, R, W, eventids, v0


def collect_station_coordinates(inventory):
    coords = {}
    for net in inventory.networks:
        for sta in net.stations:
            cha = sta.channels[0]
            lat = cha.latitude or sta.latitude
            lon = cha.longitude or sta.longitude
            key = '%s.%s' % (net.code, sta.code)
            coords[key] = (lat, lon)
    return coords


# http://stackoverflow.com/a/9400562
def _merge_sets(sets):
    newsets, sets = sets, []
    while len(sets) != len(newsets):
        sets, newsets = newsets, []
        for aset in sets:
            for eachset in newsets:
                if not aset.isdisjoint(eachset):
                    eachset.update(aset)
                    break
            else:
                newsets.append(aset)
    return newsets


def align_site_responses(results, station=None, response=1., use_sparse=True,
                         seismic_moment_method=None,
                         seismic_moment_options=None):
    """(experimental) Align station site responses and correct source params

    Determine best factor for each event so that site response is the same
    for each station and different events."""
    # Ignore not existing event results
    results['events'] = {evid: eres for (evid, eres) in
                         results['events'].items() if eres is not None}
    join_unconnected = None
    inventory = None
    if join_unconnected:
        msg = 'This feature needs more work and tests'
        raise NotImplementedError(msg)
        coordinates = collect_station_coordinates(inventory)
        station_by_coordinate = {c: sta for sta, c in coordinates.items()}
    Ne = len(results['events'])
    if Ne == 1:
        use_sparse = False
    # Determine number of freqs
    Nf = None
    for evid, eres in results['events'].items():
        Nf2 = len(eres['W'])
        if Nf is not None:
            assert Nf == Nf2
        Nf = Nf2
    # Determine number of events at stations for each freq band
    Nstations = [defaultdict(int) for i in range(Nf)]
    for evid, eres in results['events'].items():
        for i in range(Nf):
            for sta, Rsta in eres['R'].items():
                Rsta = Rsta[i]
                if Rsta is None or np.isnan(Rsta):
                    continue
                Nstations[i][sta] += 1

    def construct_ols(coldata, b_val):
        # b, row and Arepr are nonlocal lists
        b.append(b_val)
        if use_sparse:
            for col, data in coldata:
                Arepr[0].append(data)
                Arepr[1][0].append(row[0])
                Arepr[1][1].append(col)
        else:
            Arow = np.zeros(Ne)
            for col, data in coldata:
                Arow[col] = data
            Arepr.append(Arow)
        row[0] += 1

    # calculate best factors for each freq band with OLS A*factor=b
    factors = np.empty((Ne, Nf))
    for i in range(Nf):
        log.info('align sites for freq no. %d', i)
        # find unconnected areas
        areas = []
        for evid in results['events']:
            R = results['events'][evid]['R']
            area = {sta for sta, Rsta in R.items()
                    if Rsta[i] is not None and not np.isnan(Rsta[i])}
            if len(area) > 0:
                areas.append(area)
        areas = _merge_sets(areas)
        areas = {list(a)[0]: a for a in areas}
        log.info('found %d unconnected areas', len(areas))
        for name in areas:
            stations = areas[name]
            log.debug('area "%s" with %d stations', name, len(stations))
        near_stations = []
        near_stations_eq = {}
        if join_unconnected:
            # At the moment, areas are joined by setting the site response of
            # the station pair with smallest distance to 1
            # Often this works, but sometimes it produces undesired results
            # reduce number of coordinates in each area
            hulls = {}
            for name in areas:
                points = np.array([coordinates[sta] for sta in areas[name]])
                hull = scipy.spatial.ConvexHull(points)
                hulls[name] = {station_by_coordinate[tuple(p)]
                               for p in points[hull.vertices, :]}
            # calculated distances between unconnected areas
            distance = {}
            for a1 in areas:
                for a2 in areas:
                    name = frozenset((a1, a2))
                    if name in distance or a1 == a2:
                        continue
                    dists = {}
                    for sta1 in hulls[a1]:
                        for sta2 in hulls[a2]:
                            args = coordinates[sta1] + coordinates[sta2]
                            dist = gps2dist_azimuth(*args)[0]
                            dists[(sta1, sta2)] = dist
                    mink = min(dists, key=dists.get)
                    distance[name] = (dists[mink] / 1e3, mink)
            # join unconnected regions
            while len(distance) > 0:
                nearest_pair = min(distance, key=distance.get)
                dist = distance[nearest_pair][0]
                if dist > join_unconnected:
                    break
                s1, s2 = distance[nearest_pair][1]
                near_stations.extend([s1, s2])
                near_stations_eq[s1] = s2
                near_stations_eq[s2] = s1
                a1, a2 = tuple(nearest_pair)
                msg = 'connect areas %s and %s with distance %.1fkm'
                log.debug(msg, a1, a2, dist)
                distance.pop(nearest_pair)
                areas[a1] |= areas.pop(a2)
                hulls[a1] |= hulls.pop(a2)
                for a3 in areas:
                    if a3 in (a1, a2):
                        continue
                    pair1 = frozenset((a1, a3))
                    pair2 = frozenset((a2, a3))
                    dist1 = distance[pair1]
                    dist2 = distance.pop(pair2)
                    if dist2[0] < dist1[0]:
                        distance[pair1] = dist2
        largest_area = max(areas, key=lambda k: len(areas[k]))
        msg = 'use largest area %s with %d stations'
        log.info(msg, largest_area, len(areas[largest_area]))
        largest_area = areas[largest_area]

        row = [0]
        b = []
        if use_sparse:
            Arepr = [[], [[], []]]
        else:
            Arepr = []
        norm_row_A = defaultdict(float)
        norm_row_b = 0.
        first = {}
        last = {}
        # add pairs of site responses for one station and different events
        for k, item in enumerate(results['events'].items()):
            evid, eres = item
            for sta, Rsta in eres['R'].items():
                Rsta = Rsta[i]
                if Rsta is None or np.isnan(Rsta):
                    continue
                if sta not in largest_area:
                    continue
                if station is None:
                    # collect information if product of station site responses
                    # is to be normalized
                    fac = 1. / Nstations[i][sta] / len(Nstations[i])
                    norm_row_A[k] += fac
                    norm_row_b -= np.log(Rsta) * fac
                if sta == station:
                    # pin site response of specific station
                    b_val = np.log(response) - np.log(Rsta)
                    construct_ols(((k, 1),), b_val)
                elif sta in last:
                    # add pairs of site responses for one station
                    # and 2 different events
                    kl, Rstal = last[sta]
                    b_val = np.log(Rstal) - np.log(Rsta)
                    construct_ols(((k, 1), (kl, -1)), b_val)
                    last[sta] = k, Rsta
                elif sta in near_stations and near_stations_eq[sta] in last:
                    kl, Rstal = last[near_stations_eq[sta]]
                    b_val = np.log(Rstal) - np.log(Rsta)
                    construct_ols(((k, 1), (kl, -1)), b_val)
                    last[sta] = k, Rsta
                else:
                    last[sta] = first[sta] = (k, Rsta)
        if station is None:
            # pin product of station site responses
            norm_row_b += np.log(response)
            construct_ols(norm_row_A.items(), norm_row_b)
        msg = 'constructed %scoefficient matrix with shape (%d, %d)'
        log.debug(msg, 'sparse ' * use_sparse, row[0], Ne)
        # solve least squares system
        b = np.array(b)
        if use_sparse:
            A = scipy.sparse.csr_matrix(tuple(Arepr), shape=(row[0], Ne))
            res = scipy.sparse.linalg.lsmr(A, b)
        else:
            A = np.array(Arepr)
            res = scipy.linalg.lstsq(A, b, overwrite_a=True, overwrite_b=True)
        factors[:, i] = np.exp(res[0])
    # Scale W and R
    log.debug('scale events and site responses')
    for i in range(Nf):
        for k, item in enumerate(results['events'].items()):
            evid, eres = item
            W = eres['W']
            if W[i] is None or np.isnan(W[i]):
                continue
            W[i] /= factors[k, i]
            R = eres['R']
            for sta, Rsta in R.items():
                if Rsta[i] is None or np.isnan(Rsta[i]):
                    continue
                Rsta[i] *= factors[k, i]
    # Calculate omM, M0 and m again
    csp = calculate_source_properties
    csp(results, seismic_moment_method=seismic_moment_method,
        seismic_moment_options=seismic_moment_options)
    return results


def calculate_source_properties(results, rh0=None, v0=None,
                                seismic_moment_method=None,
                                seismic_moment_options=None):
    conf = results.get('config', {})
    rho0 = rh0 or conf.get('rho0')
    v02 = v0 or conf.get('v0')
    smm = seismic_moment_method or conf.get('seismic_moment_method')
    smo = seismic_moment_options or conf.get('seismic_moment_options')
    freq = results.get('freq')
    if rho0:
        for r in results['events'].values():
            v0 = r.get('v0') or v02
            r.pop('M0', None)
            r.pop('fc', None)
            r.pop('n', None)
            r.pop('gamma', None)
            if v0:
                insert_source_properties(freq, r, v0, rho0, smm, smo)
    return results


def insert_source_properties(freq, evresult, v0, rho0, seismic_moment_method,
                             seismic_moment_options, catmag=None):
    """Insert omM, Mw and possibly Mcat in evresult dictionary"""
    if evresult['W'] and rho0 and v0:
        evresult['omM'] = []
    for i, f in enumerate(freq):
        if evresult['W'][i] and rho0 and v0:
            evresult['omM'].append(sds(evresult['W'][i], f, v0, rho0))
        else:
            evresult['omM'].append(None)
    if seismic_moment_method:
        omM = evresult['omM']
        fitresult = fit_sds(freq, omM, method=seismic_moment_method,
                            **seismic_moment_options)
        if fitresult is not None:
            evresult.update(fitresult)
            evresult['Mw'] = moment_magnitude(fitresult['M0'])
            if catmag is not None:
                evresult['Mcat'] = catmag
    return sort_dict(evresult)


def _check_times(tr, tw, tol=0.5):
    return tr.stats.starttime > tw[0] + tol or tr.stats.endtime < tw[1] - tol


def Gsmooth(G_func, r, t, v0, g0, smooth=None, smooth_window='flat'):
    Gc = smooth_func(lambda t_: G_func(r, t_, v0, g0),
                     t, smooth, window=smooth_window)
    return Gc


def _get_local_minimum(tr, smooth=None, ratio=5):
    data = tr.data
    if smooth:
        window_len = int(round(smooth * tr.stats.sampling_rate))
        try:
            data = smooth_(tr.data, window_len=window_len, method='clip')
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
              noise_windows=None, bulk_window=None, weight=None,
              optimize={}, g0_bounds=(1e-8, 1e-3), b_bounds=(1e-5, 10),
              num_points_integration=1000,
              smooth=None, smooth_window='flat',
              remove_noise=False, cut_coda=None, skip=None,
              adjust_sonset=None, adjust_sonset_options={},
              plot_energies=False, plot_energies_options={},
              plot_optimization=False, plot_optimization_options={},
              plot_fits=False, plot_fits_options={},
              ignore_network_code=False, borehole_stations=(),
              G_module='qopen.rt',
              fix=False, fix_params=None,
              **kwargs):
    """
    inversion for given streams and a specific frequency band

    :param freq_band, streams, borehole_stations, fix, fix_params:
        are calculated in invert function.
    All other options are described in the example configuration file.
    """
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
            energies.append(observed_energy(stream, rho0, df, fs=fs))
        except CustomError as ex:
            msg = '%s %s: cannot calculate ernergy (%s)'
            log.warning(msg, pair[0], pair[1], str(ex))
        except Exception as ex:
            msg = '%s %s: cannot calculate ernergy (%s)'
            log.error(msg, pair[0], pair[1], str(ex))

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
            energy.stats.noise_level = noise_level = np.min(noise_levels)
            log.debug('%s: noise level at %.1e', pair, noise_level)
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
            if plot_fits:
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
        if skip and skip.get('coda_window'):
            cw = codaw[pair]
            val = skip['coda_window']
            if val and cw[1] - cw[0] < val:
                msg = ('%s: coda window of length %.1fs shorter than '
                       '%.1f -> skip pair')
                log.debug(msg, pair, cw[1] - cw[0], val)
                energies.remove(energy)
                continue
        # use only data before detected local minimum in coda
        if cut_coda:
            if cut_coda is True:
                cut_coda = {}
            esl = _get_slice(energy, codaw[pair], pair, energies)
            if esl is None:
                continue
            tmin = _get_local_minimum(esl, **cut_coda)
            if tmin:
                msg = '%s: cut coda at local minimum detected at %.2fs.'
                log.debug(msg, pair, tmin - otime)
                codaw[pair][1] = tmin
        # Optionally skip some stations if specified conditions are met
        if skip and skip.get('coda_window') and cut_coda:
            cw = codaw[pair]
            val = skip['coda_window']
            if val and cw[1] - cw[0] < val:
                msg = ('%s: coda window of length %.1fs shorter than '
                       '%.1f -> skip pair')
                log.debug(msg, pair, cw[1] - cw[0], val)
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
    for i, B in enumerate(Ecoda + [[_] for _ in Ebulk]):
        A = np.zeros((Ns + Ne - fix, len(B)))
#        A[i % Ns, :] = 1
        evid, st = event_station_pairs[i % len(event_station_pairs)]
        A[stations.index(st), :] = 1
        idx = eventids.index(evid)
        if idx > 0:
            A[Ns + idx - 1, :] = 1
        As.append(A)
    del st, evid
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
    G_func = load_func(G_module, 'G')

    def lstsq(g0, opt=False, b_fix=None):
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
#        B = np.log(E) - np.log(G)
        if b_fix:
            B = B + b_fix * np.hstack(tcoda + tbulk)
        if bulk_window:
            weights = np.hstack((np.ones(len(B) - len(weights_bulk)),
                                 weights_bulk))
        else:
            weights = 1
        if np.any(np.isinf(B)) and nonlocal_['warn']:
            nonlocal_['warn'] = False
            msg = '%s: log(E/G) has infinite values. These values are droped.'
            log.warning(msg, pair)
        # scipy.linalg.lstsq can only be used for ordinary (unweighted) LES
        # C, _, _, _ = scipy.linalg.lstsq(A, B) (with C == results.params)
        wls = WLS(B, A, weights=weights, missing='drop')
        results = wls.fit()
        # err equals approx. (np.sum((B - np.dot(A, C)) ** 2) / len(B)) ** 0.5
        err = results.mse_resid ** 0.5
        C = results.params
        # intrinsic attenuation
        b = b_fix or C[-1]
        if (b_bounds and not b_bounds[0] < b < b_bounds[1] or
                g0_bounds and not g0_bounds[0] <= g0 <= g0_bounds[1]):
            err = np.inf
        # spectral source energy of 1st ev
        N1 = len(C) - Ne + fix
        N2 = len(C) - 1 + fix
        W0 = np.exp(np.mean(C[:N1])) / R0
        # source energies of all events
        W = [W0] + list(np.exp(C[N1:N2]) * W0)
        R = np.exp(C[:N1]) / W0  # amplification factors
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
        g0_fix, b_fix = fix_params
        err, g0, b, W, R, info = lstsq(g0_fix, b_fix=b_fix)
        assert g0 == g0_fix
        assert b == b_fix
        msg = 'solved WLS with %d equations and %d unknowns, error: %.1e'
        log.debug(msg, A.shape[0], A.shape[1], err)

    else:
        # Optimize g0, so that inversion yields minimal error
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
    R = OrderedDict([(st, Ri) for st, Ri in zip(stations, R)])
    W = OrderedDict([(evid, Wi) for evid, Wi in zip(eventids, W)])
    result = sort_dict({'g0': g0, 'b': b, 'W': W, 'R': R, 'error': err,
                        'v0': v0})
    msg = 'freq band (%.2fHz, %.2fHz): optimization result is %s'
    log.debug(msg, freq_band[0], freq_band[1], json.dumps(result))
    # Dump pkl files for external plotting
    if DUMP_PKL:
        import cPickle as pickle
        with open('opt.pkl', 'wb') as f:
            pickle.dump((record, record_g0), f, 2)
        with open('fit.pkl', 'wb') as f:
            t = (energies, g0, b, W, R, v0, info, smooth, smooth_window)
            pickle.dump(t, f, 2)

    # Optionally plot result of optimization routine
    label_eventid = (len(eventids) == 1)

    def fname_and_title(fname, evtotitle=False):
        part1 = '%05.2fHz-%05.2fHz' % freq_band
        title = 'filter: (%.2fHz, %.2fHz)' % freq_band
        if label_eventid:
            eventid = energies[0].stats.eventid
            part1 = '%s_%s' % (eventid, part1)
            if evtotitle:
                title = 'event: %s  %s' % (eventid, title)
        return (fname % part1), title

    try:
        if plot_energies and len(energies) > 0:
            pkwargs = copy(plot_energies_options)
            fname = pkwargs.pop('fname', 'energies_%s.png')
            fname, title = fname_and_title(fname)
            pkwargs.update({'bulk_window': bulkw, 'coda_window': codaw})
            from qopen.imaging import plot_energies
            plot_energies(energies, fname=fname, title=title, **pkwargs)
            log.debug('create energies plot at %s', fname)
        if plot_optimization and not fix:
            pkwargs = copy(plot_optimization_options)
            fname = pkwargs.pop('fname', 'optimization_%s.png')
            fname, title = fname_and_title(fname)
            from qopen.imaging import plot_optimization
            plot_optimization(record, record_g0, fname=fname, title=title,
                              **pkwargs)
            log.debug('create optimization plot at %s', fname)
        if plot_fits:
            pkwargs = copy(plot_fits_options)
            fname = pkwargs.pop('fname', 'fits_%s.png')
            fname, title = fname_and_title(fname)
            from qopen.imaging import plot_fits
            plot_fits(energies, g0, b, W, R, v0, info, G_func,
                      smooth=smooth, smooth_window=smooth_window,
                      title=title, fname=fname, **pkwargs)
            log.debug('create fits plot at %s', fname)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        log.exception('error while creating a plot (invert_fb)')
    if (b_bounds and not 1.01 * b_bounds[0] < b < 0.99 * b_bounds[1] or
            g0_bounds and not 1.01 * g0_bounds[0] < g0 < 0.99 * g0_bounds[1]):
        msg = 'freq band (%.2f, %.2f): b=%.1e or g0=%.1e near bounds -> skip'
        log.warning(msg, *(freq_band + (b, g0)))
        return
    return g0, b, W, R, err, v0


def invert(events, inventory, get_waveforms,
           request_window, freqs, filter,
           rho0, vp=None, vs=None,
           remove_response=None, skip=None, use_picks=False,
           correct_for_elevation=False,
           parallel=False, njobs=1,
           seismic_moment_method=None, seismic_moment_options={},
           plot_eventresult=False, plot_eventresult_options={},
           plot_eventsites=False, plot_eventsites_options={},
           plot_results=False, plot_results_options={},
           plot_sites=False, plot_sites_options={},
           plot_sds=False, plot_sds_options={},
           plot_mags=False, plot_mags_options={},
           fix_params=None,
           **kwargs):
    """
    qopen function inverting all events and stations simultaneously.

    :param events: is determined in :func: qopens
    :param inventory, get_waveforms: are determined in :func: run
    All other options are described in the example configuration file.
    """
    if joblib and parallel:
        log.debug('use %d cores for parallel computation', njobs)
    elif parallel:
        log.warning('install joblib to use parallel option')
    # Get origins and arrivals of event
    origins = {}
    event_dict = {}
    if use_picks:
        arrivals = {}
    for event in events:
        event_dict[get_eventid(event)] = event
        if use_picks:
            ar = get_arrivals(event)
            if ar is not None:
                arrivals[get_eventid(event)] = ar
        if not use_picks or ar is not None:
            origins[get_eventid(event)] = get_origin(event)
    # Get frequencies
    freq_bands = get_freqs(**freqs)
    # Get stations
    channels = inventory.get_contents()['channels']
    stations = list(set(get_station(ch) for ch in channels))
    one_channel = {get_station(ch): ch for ch in channels}
    event_station_pairs = [(evid, sta) for evid in origins
                           for sta in stations]
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
        except:
            raise CustomError
        args = (c['latitude'], c['longitude'], ori.latitude, ori.longitude)
        hdist = gps2dist_azimuth(*args)[0]
        vdist = (ori.depth + c['elevation'] * correct_for_elevation -
                 c['local_depth'])
        if c['local_depth'] > 0:
            borehole_stations.add(sta)
        return np.sqrt(hdist ** 2 + vdist ** 2)

    distances = {}
    for pair in event_station_pairs[:]:
        try:
            distances[pair] = dist = _get_distance(*pair)
        except CustomError:
            msg = '%s: station not installed -> skip pair'
            log.debug(msg, pair)
            event_station_pairs.remove(pair)
            continue
        except:
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

    # Calculate onsets
    def _get_onsets(evid, sta):
        if use_picks:
            onsets = get_picks(arrivals[evid], sta)
        else:
            ori = origins[evid]
            onsets = {'P': ori.time + _get_distance(evid, sta) / vp,
                      'S': ori.time + _get_distance(evid, sta) / vs}
        return onsets

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
    except CustomError as ex:
        msg = 'exception while determining onsets (%s) -> skip event'
        log.error(msg, str(ex))
        return
    except:
        log.exception('exception while determining onsets -> skip event')
        return
    log.debug('origin station distances: %s', distances)
    log.debug('onsets: %s', onsets)
    if len(borehole_stations) > 0:
        msg = 'identified borehole stations: %s'
        log.debug(msg, ' '.join(borehole_stations))

    # Check if enough pairs left
    if (len(event_station_pairs) == 0 or skip and
            len(event_station_pairs) <= skip.get('num_pairs', 0)):
        msg = ('only %d pairs left -> return')
        log.info(msg, len(event_station_pairs))
        return

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
                if _check_times(tr, (t1, t2)):
                    msg = ('%s: data missing at one end of requested time '
                           'window')
                    log.warning(msg, pair)
                    stream = None
                    break
        if stream is None:
            event_station_pairs.remove(pair)
        elif len(stream) != 3:
            msg = '%s: number of traces with channel %s is not 3 -> skip pair'
            log.warning(msg, pair, seedid)
            event_station_pairs.remove(pair)
        else:
            for tr in stream:
                tr.stats.eventid = evid
                tr.stats.origintime = origins[evid].time
                tr.stats.ponset = onsets['P'][pair]
                tr.stats.sonset = onsets['S'][pair]
                tr.stats.distance = distances[pair]
            streams.append(stream)
    msg = 'succesfully requested %d streams for %d stations and %d events'
    log.info(msg, len(streams), len(stations), len(origins))

    # Optionally remove instrument response
    if remove_response:
        for stream in streams[:]:
            pair = get_pair(stream[0])
            fail = stream.attach_response(inventory)
            if len(fail) > 0:
                msg = '%s: no instrument response availlable -> skip pair'
                log.error(msg, pair)
                streams.remove(stream)
                event_station_pairs.remove(pair)
                continue
            try:
                if remove_response == 'full':
                    stream.remove_response()
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
        msg = ('instrument correction (%s) finished for %d streams of %d '
               'stations and %d events')
        log.info(msg, remove_response, len(streams), len(stations),
                 len(origins))

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
    kw.update({'rho0': rho0, 'borehole_stations': borehole_stations,
               'skip': skip, 'filter': filter})
    if fix_params:
        # if fix_params is used, the inversion for station site responses and
        # energy source terms are done for fixed g0 and b from previous results
        kw['fix'] = True
        if set(fix_params['freq']) != set(freq_bands.keys()):
            msg = ('Frequencies for fixed inversion have to be the same '
                   'as in the original inversion')
            raise ValueError(msg)
        fp = fix_params
        fix_params = {freq_bands[cfreq]: (g0f, bf) for cfreq, g0f, bf in
                      zip(fp['freq'], fp['g0'], fp['b'])}
    else:
        kw['fix'] = False
        fix_params = defaultdict(lambda: None)
    # Start invert_fb function
    if joblib and parallel:
        rlist = Parallel(n_jobs=njobs, pre_dispatch='1.5*n_jobs')(
            delayed(invert_fb)(fb, deepcopy(streams),
                               fix_params=fix_params[fb], **kw)
            for fb in freq_bands.values())
    else:
        # deepcopy only necessary for more than one freq band
        cond = len(freq_bands) > 1
        rlist = [invert_fb(fb, deepcopy(streams) if cond else streams,
                           fix_params=fix_params[fb], **kw)
                 for fb in freq_bands.values()]

    # Check if any result
    if all([r is None for r in rlist]):
        log.warning('invert: no result for any frequency band')
        return

    # Re-sort results
    result = defaultdict(list)
    result['R'] = defaultdict(list)
    result['events'] = defaultdict(lambda: defaultdict(list))
    for (cfreq, freq_band), res in zip(freq_bands.items(), rlist):
        if res is None:
            msg = 'freq band (%.2f, %.2f): no result'
            log.debug(msg, *freq_band)
            g0opt, b, W, R, error, v0 = 6 * (None,)
        else:
            g0opt, b, W, R, error, v0 = res
        result['freq'].append(cfreq)
        result['g0'].append(g0opt)
        result['b'].append(b)
        # result['v0'].append(v0)
        if v0 is not None:
            result['v0'] = v0
        result['error'].append(error)
        for st in stations:
            if R is None:
                result['R'][st].append(None)
            else:
                result['R'][st].append(R.get(st))
        # result['W'].append(W)
        for event in events:
            evid = get_eventid(event)
            if W is None:
                result['events'][evid]['W'].append(None)
#                result['events'][evid]['omM'].append(None)
            else:
                result['events'][evid]['W'].append(W.get(evid))
#                omM = sds(W.get(evid), cfreq, kwargs.get('v0'), rho0)
#                result['events'][evid]['omM'].append(omM)
    # Calculate source properties omM, M0 and Mw
    for event in events:
        evid = get_eventid(event)
        args = (result['freq'], result['events'][evid], result['v0'], rho0,
                seismic_moment_method, seismic_moment_options,
                get_magnitude(event_dict[evid]))
        result['events'][evid] = insert_source_properties(*args)
    result['R'] = OrderedDict(result['R'])
    result['events'] = OrderedDict(result['events'])
    if 'freq' not in result or all([g0 is None for g0 in result['g0']]):
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
            plot_(result, eventid=get_eventid(event),
                  # v0=kwargs.get('v0'),
                  plot_eventresult=plot_eventresult,
                  plot_eventresult_options=plot_eventresult_options,
                  plot_eventsites=plot_eventsites,
                  plot_eventsites_options=plot_eventsites_options)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            log.exception('error while creating a plot (invert)')
    return result


def invert_wrapper(events, plot_results=False, plot_results_options={},
                   plot_sites=False, plot_sites_options={},
                   plot_sds=False, plot_sds_options={},
                   plot_mags=False, plot_mags_options={},
                   invert_events_simultaniously=False,
                   mean=None, **kwargs):
    """qopen function for a list or Catalog of events

    Depending on 'invert_events_simultaniously' flag the function
    calls :func: qopen for each event seperately or for all events once.
    In the first case mean results are calculated.

    :param events: is determined in :func: run.
    The rest of the options are described in the example configuration file.
    """
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
    if invert_events_simultaniously:
        result = invert(events, **kwargs)
    else:
        result = {'events': OrderedDict(), 'R': OrderedDict()}
        for i, event in enumerate(events):
            evid = get_eventid(event)
            msg = 'event %s (no %d of %d): %s processing'
            log.info(msg, evid, i + 1, len(events), 'start')
            res = invert([event], **kwargs)
            log.info(msg, evid, i + 1, len(events), 'end')
            if res:
                result['freq'] = res.pop('freq')
                res.update(res['events'].pop(evid))
                del res['events']
                result['events'][evid] = sort_dict(res)
        if len(result['events']) == 0:
            log.warning('invert_wrapper: no result')
            return
        g0, b, error, R, _, _, _ = collect_results(result)
        if np.all(np.isnan(g0)):
            log.warning('invert_wrapper: no result')
            return
        kw = {'axis': 0, 'robust': mean == 'robust',
              'weights': 1 / np.array(error) if mean == 'weighted' else None}
        result['g0'] = gmean(g0, **kw).tolist()
        result['b'] = gmean(b, **kw).tolist()
        result['error'] = gmean(error, **kw).tolist()
        for st, Rst in R.items():
            result['R'][st] = gmean(Rst, axis=0).tolist()
    result['config'] = {k: kwargs[k] for k in DUMP_CONFIG if k in kwargs}
    result['config'][
        'invert_events_simultaniously'] = invert_events_simultaniously
    result['config']['mean'] = mean
    result['config'] = sort_dict(result['config'], order=DUMP_CONFIG)
    result = sort_dict(result)
    # Optionally plot stuff
    try:
        plot_(result, plot_results=plot_results,
              plot_results_options=plot_results_options,
              plot_sites=plot_sites, plot_sites_options=plot_sites_options,
              plot_sds=plot_sds, plot_sds_options=plot_sds_options,
              plot_mags=plot_mags, plot_mags_options=plot_mags_options)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        log.exception('error while creating a plot (invert_wrapper)')
    return result


def plot_(result, eventid=None, v0=None,
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
    if eventid is None:
        if plot_results:
            pkwargs = copy(plot_results_options)
            fname = pkwargs.pop('fname', 'results.pdf')
            from qopen.imaging import plot_results
            plot_results(result, fname=fname, **pkwargs)
            log.debug('create results plot at %s', fname)
        if plot_sites:
            pkwargs = copy(plot_sites_options)
            fname = pkwargs.pop('fname', 'sites.pdf')
            from qopen.imaging import plot_sites
            plot_sites(result, fname=fname, **pkwargs)
            log.debug('create sites plot at %s', fname)
        if plot_sds:
            pkwargs = copy(plot_sds_options)
            fname = pkwargs.pop('fname', 'sds.pdf')
            from qopen.imaging import plot_all_sds
            plot_all_sds(result, fname=fname, **pkwargs)
            log.debug('create sds plot at %s', fname)
        if plot_mags:
            pkwargs = copy(plot_mags_options)
            fname = pkwargs.pop('fname', 'mags.pdf')
            from qopen.imaging import plot_mags
            plot_mags(result, fname=fname, **pkwargs)
            log.debug('create mags plot at %s', fname)
    else:
        if eventid not in result['events']:
            raise ParseError('No event with this id in results')
        if plot_eventresult:
            pkwargs = copy(plot_eventresult_options)
            fname = pkwargs.pop('fname', 'eventresult_%s.pdf')
            fname = fname % (eventid,)
            title = 'event %s' % (eventid,)
            from qopen.imaging import plot_eventresult
            plot_eventresult(result, title=title, fname=fname, **pkwargs)
            log.debug('create eventresult plot at %s', fname)
        if plot_eventsites:
            pkwargs = copy(plot_eventsites_options)
            fname = pkwargs.pop('fname', 'eventsites_%s.pdf')
            fname = fname % (eventid,)
            title = 'event %s' % (eventid,)
            from qopen.imaging import plot_eventsites
            plot_eventsites(result, title=title, fname=fname)
            log.debug('create eventsites plot at %s', fname)


def load_func(modulename, funcname):
    """Load and return function from Python module"""
    sys.path.append(os.path.curdir)
    module = import_module(modulename)
    sys.path.pop(-1)
    func = getattr(module, funcname)
    return func


def init_data(data, client_options=None, plugin=None, cache_waveforms=False):
    """Return appropriate get_waveforms function

    See example configuration file for a description of the options"""
    if client_options is None:
        client_options = {}
    try:
        client_module = import_module('obspy.clients.%s' % data)
    except ImportError:
        client_module = None
    if client_module:
        Client = getattr(client_module, 'Client')
        client = Client(**client_options)

        def get_waveforms(event=None, **args):
            return client.get_waveforms(**args)
    elif data == 'plugin':
        modulename, funcname = plugin.split(':')
        get_waveforms = load_func(modulename.strip(), funcname.strip())
    else:
        from obspy import read
        stream = read(data)

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
    if use_cache and joblib:
        log.info('use waveform cache in %s', cache_waveforms)
        memory = joblib.Memory(cachedir=cache_waveforms, verbose=0)
        return memory.cache(wrapper)
    elif use_cache:
        log.warning('install joblib to use cache_waveforms option')
    return wrapper


class ConfigJSONDecoder(json.JSONDecoder):
    """Strip lines from comments"""
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


def run(conf=None, create_config=None, tutorial=False, eventid=None,
        get_waveforms=None, prefix=None, plot=None, fix_params=None,
        align_sites=None, align_sites_station=None, align_sites_value=1.,
        calc_source_params=None,
        **args):
    """
    Main entry point for a direct call from Python

    Example call: main(conf='conf.json')

    :param args: All args correspond to the respective command line and
        configuration options.
        See the example configuration file for help and possible arguments.
        Options in args can overwrite the configuration from the file.
    Exceptions from description in configuration file:
    :param events: can be filename or ObsPy Catalog object
    :param inventory: can be filename or ObsPy Inventory object
    :param get_waveforms: function, if given the data option will be ignored.
        get_waveforms will be called as described in the example configuration
        file
    """
    time_start = time.time()
    # Copy example files if create_config or tutorial
    if create_config or tutorial:
        if create_config is None:
            create_config = 'conf.json'
        srcs = ['conf.json']
        dest_dir = os.path.dirname(create_config)
        dests = [create_config]
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
    if conf in ('None', 'none', 'null', ''):
        conf = None
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
    # Optionally plot
    if plot:
        with open(plot) as f:
            result = json.load(f)
        if conf:
            result['config'] = args
        else:
            result['config'].update(args)
            args = result['config']
        plot_(result, eventid=eventid, **args)
        return
    # Configure logging
    kw = {'loggingc': args.pop('logging', None),
          'verbose': args.pop('verbose', 0),
          'loglevel': args.pop('loglevel', 3),
          'logfile': args.pop('logfile', None)}
    configure_logging(**kw)
    if not calc_source_params:
        try:
            # Read inventory
            inventory = args.pop('inventory')
            fi = args.pop('filter_inventory', None)
            if not isinstance(inventory, obspy.Inventory):
                if isinstance(inventory, str):
                    format_ = None
                else:
                    inventory, format_ = inventory
                inventory = obspy.read_inventory(inventory, format_)
                if fi:
                    inventory = inventory.select(**fi)
                channels = inventory.get_contents()['channels']
                stations = list(set(get_station(ch) for ch in channels))
                log.info('read inventory with %d stations', len(stations))
            elif fi:
                inventory = inventory.select(**fi)
            if not align_sites:
                # Read events
                events = args.pop('events')
                if (not isinstance(events, obspy.Catalog) or
                        not isinstance(events, list) or
                        (len(events) == 2 and isinstance(events[0], str))):
                    if isinstance(events, str):
                        format_ = None
                    else:
                        events, format_ = events
                    events = obspy.read_events(events, format_)
                    log.info('read %d events', len(events))
            # Initialize get_waveforms
                keys = ['client_options', 'plugin', 'cache_waveforms']
                tkwargs = {k: args.pop(k, None) for k in keys}
                if get_waveforms is None:
                    data = args.pop('data')
                    get_waveforms = init_data(data, **tkwargs)
                    log.info('init data from %s', data)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
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
    log.debug('start qopen routine with parameters %s', json.dumps(args))
    if not calc_source_params:
        args['inventory'] = inventory
    if not (align_sites or calc_source_params):
        args['get_waveforms'] = get_waveforms
        args['events'] = events
    output = args.pop('output', None)
    indent = args.pop('indent', None)
    if prefix:
        if output is not None:
            output = prefix + output
        targets = ['energies', 'optimization', 'fits', 'eventresult',
                   'eventsites', 'results', 'sites', 'sds', 'mags']
        for t in targets:
            key = 'plot_%s_options' % t
            default = '%s.png' % t
            if key in args:
                args[key]['fname'] = prefix + args[key].get('fname', default)
    if fix_params and not (align_sites or calc_source_params):
        # Optionally fix g0 and b
        log.info('use fixed g0 and b')
        with open(fix_params) as f:
            args['fix_params'] = json.load(f)
    if align_sites or calc_source_params:
        kw = {'seismic_moment_method': args.get('seismic_moment_method'),
              'seismic_moment_options': args.get('seismic_moment_options')}
    if align_sites:
        msg = 'align station site responses and re-calculate source parameters'
        log.info(msg)
        with open(align_sites) as f:
            result = json.load(f)
        align_site_responses(result, station=align_sites_station,
                             response=align_sites_value, **kw)
        result.setdefault('config', {}).update(kw)
        plot_(result, eventid=eventid, **args)
    elif calc_source_params:
        log.info('calculate source parameters')
        with open(calc_source_params) as f:
            result = json.load(f)
        calculate_source_properties(result, **kw)
        result.setdefault('config', {}).update(kw)
        plot_(result, eventid=eventid, **args)
    else:
        result = invert_wrapper(**args)
        # Output and return result
        log.debug('final results: %s', json.dumps(result))
    if output == 'stdout':
        print(json.dumps(result))
    elif output is not None:
        path = os.path.dirname(output)
        if path != '' and not os.path.isdir(path):
            os.makedirs(path)
        fmode = 'w' + 'b' * (not IS_PY3)  # dirty hack for now
        with open(output, fmode) as f:
            json.dump(result, f, indent=indent)
    time_end = time.time()
    log.debug('used time: %.1fs', time_end - time_start)
    return result


def run_cmdline(args=None):
    """Main entry point from the command line"""
    # Define command line arguments
    msg = ('Qopen: Seperation of intrinsic and scattering Q by envelope '
           'inversion')
    p = argparse.ArgumentParser(description=msg)
    msg = 'Configuration file to load (default: conf.json)'
    p.add_argument('-c', '--conf', default='conf.json', help=msg)
    msg = 'Process only event with this id'
    p.add_argument('-e', '--eventid', help=msg)
    msg = 'Set chattiness on command line. Up to 3 -v flags are possible'
    p.add_argument('-v', '--verbose', help=msg, action='count',
                   default=SUPPRESS)
    msg = ('Add prefix for all output files defined in config '
           '(useful for options operating on JSON files)')
    p.add_argument('--prefix', help=msg)

    g2 = p.add_argument_group('create example configuration or tutorial')
    msg = ('Create example configuration in specified file '
           '(default: conf.json if option is invoked without parameter)')
    g2.add_argument('--create-config', help=msg, nargs='?',
                    const='conf.json', default=SUPPRESS)
    msg = "Tutorial: create example configureation with data files"
    g2.add_argument('--tutorial', help=msg, action='store_true',
                    default=SUPPRESS)

    msg = ('parameters operating on JSON result file '
           '(no or different processing)')
    g1 = p.add_argument_group(msg)
    msg = ('Plot results. Can be used '
           'together with -e to plot event results from the given event')
    g1.add_argument('-p', '--plot', help=msg)
    msg = ('Fix g0 and b from results in given json file to determine better '
           'estimation of site responses and source energies (experimental)')
    g1.add_argument('--fix-params', help=msg)
    msg = ('Align site responses and correct source parameters from results '
           'in given json file (experimental)')
    g1.add_argument('--align-sites', help=msg)
    msg = ('Site response of this station is fixed '
           '(default: None -> product of station site responses is fixed)')
    g1.add_argument('--align-sites-station', help=msg)
    msg = ('Value of site response for specified station or product of '
           'station site responses (default: 1)')
    g1.add_argument('--align-sites-value', help=msg, default=1., type=float)
    msg = ('Calculate seismic moment and moment magnitude from results in '
           'given json file')
    g1.add_argument('--calc-source-params', help=msg)

    msg = ('Use these flags to overwrite values in the config file. '
           'See the example configuration file for a description of '
           'these options. Options representing dictionaries or lists are '
           'expected to be valid JSON.')
    g3 = p.add_argument_group('optional qopen arguments', description=msg)
    features_str = ('events', 'inventory', 'data', 'output',
                    'seismic-moment-method')
    features_json = ('seismic-moment-options',)
    features_bool = ('parallel', 'invert_events_simultaniously',
                     'plot_energies', 'plot_optimization', 'plot_fits',
                     'plot_eventresult', 'plot_eventsites')
    for f in features_str:
        g3.add_argument('--' + f, default=SUPPRESS)
    for f in features_json:
        g3.add_argument('--' + f, default=SUPPRESS, type=json.loads)
    g3.add_argument('--njobs', default=SUPPRESS, type=int)
    for f in features_bool:
        g3.add_argument('--' + f.replace('_', '-'), dest=f,
                        action='store_true', default=SUPPRESS)
        g3.add_argument('--no-' + f.replace('_', '-'), dest=f,
                        action='store_false', default=SUPPRESS)

    # Get command line arguments and start :func: run
    args = vars(p.parse_args(args))
    try:
        run(**args)
    except ParseError as ex:
        p.error(ex)
