# Copyright 2015-2019 Tom Eulenfeld, MIT license
"""
Plotting functions

Arguments supported by all plotting functions via its \\*\\*kwargs are:

:fname: file name for the plot output
        (if not provided the figure will be left open)
:title: title of the plot
:figsize: figure size (tuple of inches)
:dpi: resolution of image file

|
"""

from collections import OrderedDict
from copy import copy
import os

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from qopen.core import get_pair, collect_results
from qopen.source import moment_magnitude, source_model
from qopen.util import gerr, smooth_func, linear_fit

MS = mpl.rcParams['lines.markersize'] // 2

QUANTITIES = ('g0', 'lsc', 'Qsc', 'b', 'li', 'Qi', 'error', 'nobs')
QUANTITIES_EVENT = ('g0', 'lsc', 'Qsc', 'b', 'li', 'Qi', 'error', 'W', 'sds')
QLABELS = {'g0': r'g0 (m$^{-1}$)',
           'lsc': r'l$_{\mathrm{sc}}$ (km)',
           'Qsc': r'Q${_{\mathrm{sc}}}^{-1}$',
           'b': 'b (s$^{-1}$)',
           'li': r'l$_{\mathrm{i}}$ (km)',
           'Qi': r'Q${_{\mathrm{i}}}^{-1}$',
           'W': 'W (J/Hz)',
           'omM': r'$\omega$M (Nm)',
           'sds': r'$\omega$M (Nm)',
           'error': 'error',
           'nobs': 'nobs'}

DEPMAP = {'g0': 'g0', 'lsc': 'g0', 'Qsc': 'g0',
          'b': 'b', 'li': 'b', 'Qi': 'b',
          'W': 'W', 'omM': 'sds', 'sds': 'sds', 'error': 'error'}


def calc_dependent(quantity, value, freq=None, v0=None):
    """Calculate dependent value (Qsc, Qi, lsc, li) from g0 and b

    :param str quantity: one of Qsc, Qi, lsc, li
    :param value: value of g0 or b depending on requested quantity
    :param freq: frequency in Hz (needed for some calculations)
    :param v0: velocity in m/s (needed for some calculations)
    :return: value of quantity"""
    q = quantity
    val = np.array(value, dtype=float)
    if not np.isscalar(v0):
        v0 = v0[:, np.newaxis]
    if q in ('g0', 'b', 'W', '', 'error'):
        return val
    elif q == 'lsc':
        return 1 / val / 1000
    elif q == 'Qsc':  # actually inverse of Qsc
        return val * v0 / (2 * np.pi * freq)
    elif q == 'li':
        return v0 / val / 1000
    elif q == 'Qi':  # actually inverse of Qi
        return val / (2 * np.pi * freq)


def freqlim(freq):
    try:
        x1 = freq[0] ** 1.5 / freq[1] ** 0.5
        x2 = freq[-1] ** 1.5 / freq[-2] ** 0.5
    except IndexError:
        return
    return x1, x2


def _savefig(fig, title=None, fname=None, dpi=None, figsize=None):
    if figsize is not None:
        fig.set_size_inches(*figsize)
    extra = [fig.suptitle(title)] if title else None
    if fname:
        path = os.path.dirname(fname)
        if path != '':
            os.makedirs(path, exist_ok=True)
        fig.savefig(fname, bbox_inches='tight', bbox_extra_artists=extra,
                    dpi=dpi)
        plt.close(fig)


def _set_gridlabels(ax, i, nx, ny, N, xtlabel=True, ytlabel=True,
                    xlabel=None, ylabel=None):
    if i % nx != 0 and not ytlabel:
        plt.setp(ax.get_yticklabels(), visible=False)
    if i % nx == 0 and i // nx == (ny - 1) // 2 and ylabel:
        ax.set_ylabel(ylabel)
    if i < N - nx and not xtlabel:
        plt.setp(ax.get_xticklabels(), visible=False)
    if i % nx == (nx - 1) // 2 and i >= N - nx and xlabel:
        ax.set_xlabel(xlabel)


def plot_energies(energies,
                  bulk_window=None, coda_window=None, downsample_to=None,
                  xlim_lin=None, xlim_log=None, **kwargs):
    """
    Plot observed spectral energy densities on different scales (linear, log)
    """
    gs = gridspec.GridSpec(2 * len(energies), 2)
    gs.update(wspace=0.05)
    fig = plt.figure()
    sax1 = sax3 = None
    for i, tr in enumerate(energies):
        pair = get_pair(tr)
        otime = tr.stats.origintime
        if downsample_to is None:
            d = 1
        else:
            d = tr.stats.sampling_rate // downsample_to
        ts = np.arange(len(tr)) * tr.stats.delta
        ts = ts - (otime - tr.stats.starttime)
        c = 'k'
        ax2 = plt.subplot(gs[2 * i + 1, 0], sharex=sax1, sharey=sax1)
        ax1 = plt.subplot(gs[2 * i, 0], sharex=ax2)
        ax3 = plt.subplot(gs[2 * i:2 * i + 2, 1], sharex=sax3, sharey=sax3)
        ax1.annotate('%s' % pair[1], (1, 0.5), (-10, 0), 'axes fraction',
                     'offset points', size='small', ha='right', va='center')
        ax3.annotate('%s' % pair[0], (0, 1), (10, -5), 'axes fraction',
                     'offset points', size='small', ha='left', va='top')
        ax1.plot(ts[::d], tr.data[::d], color=c)
        ax2.semilogy(ts[::d], tr.data[::d], color=c)
        ax3.loglog(ts[::d], tr.data[::d], color=c)
        for ax in (ax1, ax2, ax3):
            ax.tick_params(labelbottom=False)
            ax.set_yticklabels([])
            if 'ponset' in tr.stats:
                tponset = tr.stats.ponset - otime
                ax.axvline(tponset, color='green', alpha=0.5)
            if 'sonset' in tr.stats:
                tsonset = tr.stats.sonset - otime
                ax.axvline(tsonset, color='b', alpha=0.5)
        for ax in (ax2, ax3):
            if bulk_window and coda_window:
                c = ('b', 'k')
                wins = (bulk_window[pair], coda_window[pair])
                for i, win in enumerate(wins):
                    ax.axvspan(win[0] - otime, win[1] - otime,
                               0.05, 0.08, color=c[i], alpha=0.5)
        if sax1 is None:
            sax1 = ax2
            sax3 = ax3
    if xlim_lin:
        ax1.set_xlim(xlim_lin)
    if xlim_log:
        ax3.set_xlim(xlim_log)
    loglocator = mpl.ticker.LogLocator(base=100)
    ax2.yaxis.set_major_locator(loglocator)
    ax3.yaxis.set_major_locator(loglocator)
    ax2.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax3.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax2.tick_params(labelbottom=True)
    ax3.tick_params(labelbottom=True)
    _savefig(fig, **kwargs)


def plot_lstsq(rec,  event_station_pairs, ax=None, base=np.e, **kwargs):
    """Plot solution of weighted least squares inversion"""
    eventids, stations = zip(*event_station_pairs)
    eventids = list(OrderedDict.fromkeys(eventids))
    stations = list(OrderedDict.fromkeys(stations))
    err, g0, b, W, R, info = rec
    tcoda, tbulk, Ecoda, Ebulk, Gcoda, Gbulk = info
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    tmin = min(tcoda[i][0] for i in range(len(tcoda)))
    tmax = max(tcoda[i][-1] for i in range(len(tcoda)))
    for i in range(len(tcoda)):
        ev, sta = event_station_pairs[i]
        offset = R[stations.index(sta)] * W[eventids.index(ev)] / W[0]
#        offset = R[i] if len(W) == 1 else C[i]
#        Bci = np.log(Ecoda[i]) - np.log(FS * Gcoda[i]) - np.log(offset)
        Bci = np.log(Ecoda[i]) - np.log(Gcoda[i]) - np.log(offset)
        ax.plot(tcoda[i], Bci / np.log(base), color='0.7')
    for i in range(len(tbulk)):
        ev, sta = event_station_pairs[i]
        offset = R[stations.index(sta)] * W[eventids.index(ev)] / W[0]
#        offset = R[i] if len(W) == 1 else C[i]
#        Bbi = np.log(Ebulk[i]) - np.log(FS * Gbulk[i]) - np.log(offset)
        Bbi = np.log(Ebulk[i]) - np.log(Gbulk[i]) - np.log(offset)
        ax.plot(tbulk[i], Bbi / np.log(base), 'o', color='0.4', mec='0.4',
                ms=MS)
        tmin = min(tmin, tbulk[i])
    t = np.linspace(tmin, tmax, 100)
    ax.plot(t, (np.log(W[0]) - b * t) / np.log(base), color='k')
    ax.set_xlim(right=tmax)
    if fig:
        _savefig(fig, **kwargs)


def plot_optimization(record, record_g0, event_station_pairs, num=7,
                      xlabel=r'g$_0$ ($\mathrm{m}^{-1}$)',
                      ylabel=r'misfit $\epsilon$',
                      **kwargs):
    """Plot some steps of optimization"""
    fig = plt.figure()
    if num > 1:
        n = (num + 1) // 2
        gs = gridspec.GridSpec(n, n)
        ax = plt.subplot(gs[1:, 0:-1])
        share = None
    else:
        ax = fig.add_subplot(111)
    title = kwargs.pop('title')
    if title:
        ax.annotate(title, (0, 1), (5, -5), 'axes fraction', 'offset points',
                    ha='left', va='top')
    err, g0 = zip(*record_g0)
    if not np.all(np.isinf(err)):
        ax.loglog(g0, err, 'xk')
        # best value is plotted blue
        ax.loglog(g0[-1], err[-1], 'xb', mew=2)
    # infinite values are plotted with red crosses
    if np.inf in err:
        g0_inf = [g0_ for (err_, g0_) in record_g0 if err_ == np.inf]
        err_inf = np.mean(ax.get_ylim())
        ax.loglog(g0_inf, err_inf * np.ones(len(g0_inf)), 'xr')
        for i in range(len(record)):
            if record[i][0] == np.inf:
                record[i] = (err_inf,) + record[i][1:]
    if num > 1:
        for i, rec in enumerate(record):
            err, g0, b, W, _, _ = rec
            if i < n:
                gsp = gs[0, i]
                l = str(i + 1)
            elif i < num - 1:
                gsp = gs[i - n + 1, -1]
                l = str(i + 1)
            else:
                gsp = gs[n - 1, -1]
                l = 'best'
            ax2 = plt.subplot(gsp, sharex=share, sharey=share)
            plot_lstsq(rec, event_station_pairs, ax=ax2)
            ax2.annotate(l, (0, 1), (5, -5), 'axes fraction',
                         'offset points', ha='left', va='top')
            l2 = 'g$_0$=%.1e\nb=%.1e' % (g0, b)
            l2 = l2 + '\nW%s=%.1e' % ('$_1$' * (len(W) > 1), W[0])
            ax2.annotate(l2, (1, 1), (-5, -5), 'axes fraction',
                         'offset points', ha='right', va='top',
                         size='xx-small')
            if l != 'best':
                ax.annotate(l, (g0, err), (5, 5), 'data', 'offset points',
                            ha='left', va='bottom')
            if i == 0:
                share = ax2
                yl = (r'$\ln \frac{E_{\mathrm{obs}\,ij}}{G_{ij}B_jR_i}$')
                if len(W) == 1:
                    yl = (r'$\ln \frac{E_{\mathrm{obs}\,i}}{G_iR_i}$')
                ax2.set_ylabel(yl)
                plt.setp(ax2.get_xticklabels(), visible=False)
            elif l == 'best':
                ax2.set_xlabel(r'time ($\mathrm{s}$)')
                plt.setp(ax2.get_yticklabels(), visible=False)
            else:
                plt.setp(ax2.get_xticklabels(), visible=False)
                plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.locator_params(axis='y', nbins=4)
        ax2.locator_params(axis='x', nbins=3)
    ax.set_xlabel(xlabel)
    # yl = (r'error $\mathrm{rms}\left(\ln\frac{E_{\mathrm{obs}, ij}}'
    #      r'{E_{\mathrm{mod}, ij}}\right)$')
    ax.set_ylabel(ylabel)
    _savefig(fig, **kwargs)


def _get_times(tr):
    t0 = tr.stats.starttime - tr.stats.origintime
    return np.arange(len(tr)) * tr.stats.delta + t0


def plot_fits(energies, g0, b, W, R, v0, info, G_func,
              smooth=None, smooth_window='bartlett',
              xlim=None, ylim=None,
              xlabel='time (s)', ylabel=r'E (Jm$^{-3}$Hz$^{-1}$)',
              **kwargs):
    """Plot fits of spectral energy densities"""
    tcoda, tbulk, Ecoda, Ebulk, Gcoda, Gbulk = info
    N = len(energies)
    n = int(np.ceil(np.sqrt(N)))
    fig = plt.figure()
    gs = gridspec.GridSpec(n, n)
    share = None
    if b is None:
        b = 0
    tmaxs = []
    ymaxs = []
    ymins = []
    c1 = 'mediumblue'
    c2 = 'darkred'
    c1l = '#8181CD'  # 37.25% #'#8686CD'
    c2l = '#8B6969'  # 25%  #'#8B6161'
    for i, energy in enumerate(energies):
        evid, station = get_pair(energy)
        ax = plt.subplot(gs[i // n, i % n], sharex=share, sharey=share)
        plot = ax.semilogy
        if np.isnan(R[station]):
            # can happen with fixed site amplifications
            # and reference site not avaiable
            continue

        def get_Emod(G, t):
            return R[station] * W[evid] * G * np.exp(-b * t)
#            return FS * R[station] * W[evid] * G * np.exp(-b * t)
        st = energy.stats
        r = st.distance
#        ax.axvline(st.starttime - st.origintime + r / v0, ls = '--', c='gray')
#        ax.axvline(r / v0, ls='--', c='gray')
        t = _get_times(energy) + r / v0 - (st.sonset - st.origintime)

        if smooth:
            plot(t, energy.data_unsmoothed, color='0.7')
        plot(t, energy.data, color=c1l)
        G_ = smooth_func(lambda t_: G_func(r, t_, v0, g0),
                         t, smooth, window=smooth_window)
        Emod = get_Emod(G_, t)
        Emod[Emod < 1e-30] = 1e-30
        plot(t, Emod, color=c2l)

        plot(tcoda[i], Ecoda[i], color=c1)
        Emodcoda = get_Emod(Gcoda[i], tcoda[i])
        plot(tcoda[i], Emodcoda, color=c2)

        if tbulk and len(tbulk) > 0:
            plot(tbulk[i], Ebulk[i], 'o', color=c1, mec=c1, ms=MS)
            Emodbulk = get_Emod(Gbulk[i], tbulk[i])
            plot(tbulk[i], Emodbulk, 'o', ms=MS - 1,
                 color=c2, mec=c2)

        l = '%s\n%s' % (evid, station)
        l = l + '\nr=%dkm' % (r / 1000)
        ax.annotate(l, (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top', size='x-small')
        _set_gridlabels(ax, i, n, n, N, xtlabel=False, ytlabel=False,
                        xlabel=xlabel, ylabel=ylabel)
        tmaxs.append(t[-1])
        ymaxs.append(max(np.max(Emod), np.max(energy.data)))
        ymins.append(min(np.min(Emodcoda), np.max(Ecoda[i])))
        if share is None:
            share = ax
#        if True:
#            save = {'t': t, 'data': energy.stats.orig_data,
#                    'Eobs': energy.data, 'Emod': Emod,
#                    'tcoda': tcoda[i], 'Eobscoda': Ecoda[i],
#                    'Emodcoda': Emodcoda}
#            if tbulk:
#                save.update({'tbulk': tbulk[i], 'Eobsbulk': Ebulk[i],
#                             'Emodbulk': Emodbulk})
#            np.savez(fname.replace('png', '') + station + '.npz', **save)
    ax.locator_params(axis='x', nbins=5, prune='upper')
    loglocator = mpl.ticker.LogLocator(base=100)
    ax.yaxis.set_major_locator(loglocator)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(xlim or (0, max(tmaxs)))
    ax.set_ylim(ylim or (0.1 * min(ymins), 1.5 * max(ymaxs)))
    _savefig(fig, **kwargs)


def plot_sds(freq, result, ax=None,
             annotate=False, va='bottom', annotate_label=None,
             seismic_moment_method=None, seismic_moment_options={},
             cmap='viridis_r', vmin=None, vmax=None, max_nobs=None,
             color=None,
             **kwargs):
    """Plot source displacement spectrum and fitted source model"""

    kw = {'s': 4 * MS ** 2, 'marker': 'o', 'zorder': 10, #'linewidth': 0.5,
          'c': 'k'}
    if color is None and 'nstations' in result and cmap is not None:
#        Rvals = np.array(list(result['R'].values()), dtype='float')
#        nobs = np.sum(~np.isnan(Rvals), axis=0)
        nobs = result['nstations']
        if max_nobs is None:
            max_nobs = np.max(nobs)
        cmap = plt.get_cmap(cmap, max_nobs)
        if vmax is None:
            vmax = max_nobs + 0.5
        if vmin is None:
            vmin = 0.5
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        kw.update({'c': nobs, 'norm': norm, 'cmap': cmap})
    elif color is not None:
        kw['c'] = None
        kw['color'] = color
    freq = np.array(freq)
    omM = np.array(result['sds'], dtype=float)
    if all(np.isnan(omM)):
        return
    fig = None
    obs = ('M0', 'fc', 'n', 'gamma')
    smo = seismic_moment_options

    def _get(k): return smo.get(k) or result.get(k)
    smo = {k: _get(k) for k in obs if _get(k) is not None}
    M0 = smo.get('M0')
    fc = smo.get('fc')
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    if seismic_moment_method == 'mean':
        ax.plot(freq, omM, 'o-', color='gray', mec='gray')
        ax.plot(freq[freq < fc], omM[freq < fc], 'o-k')
        kw['c'] = nobs[freq < fc]
        kw['linewidth'] = 0.5
        sc = ax.scatter(freq[freq < fc], freq[freq < fc], **kw)
    elif seismic_moment_method in ('fit', 'robust_fit'):
        sc = ax.scatter(freq, omM, **kw)
        if M0 and fc:
            f = np.linspace(freq[0] / 1.5, freq[-1] * 1.5, 100)
            omM2 = source_model(f, **smo)
            ax.plot(f, omM2, '-k')
    else:
        kw['linewidth'] = 0.5
        sc = ax.scatter(freq, omM, **kw)

    if M0:
        ax.axhline(M0, ls='--', color='k')
    if annotate_label is None:
        labels = OrderedDict((('M0', r'M$_0$={M0:.1e} Nm'),
                              ('fc', r'f$_{{\rm{{c}}}}$={fc:.1f} Hz'),
                              ('n', 'n={n:.1f}'),
                              ('gamma', r'$\gamma$={gamma:.2f}'),
                              ('fit_error', 'err={fit_error:.2f}')))
        labels = [labels[key].format(**result)# % np.float32(result[key])
                  for key in labels if key in result]
        label = '\n'.join(labels)
    else:
        try:
            label = annotate_label.format(**result)
        except KeyError:
            annotate=False
    if annotate and (annotate_label or len(labels) > 0):
        ypos = 1 if va == 'top' else 0
        ax.annotate(label, (1, ypos), (-5, 5 - 10 * ypos),
                    'axes fraction', 'offset points',
                    ha='right', va=va, size='x-small')

    if fig:
        _savefig(fig, **kwargs)
    return sc


def plot_eventresult(result, v0=None, quantities=QUANTITIES_EVENT,
                     seismic_moment_method=None, seismic_moment_options={},
                     xlabel='frequency (Hz)',
                     **kwargs):
    """Plot all results of one `~qopen.core.invert()` call"""
    v0 = v0 or result.get('v0') or result.get('config', {}).get('v0')
    freq = np.array(result['freq'])
    if 'W' in quantities or 'sds' in quantities:
        res = copy(result)
        _values_view = res.pop('events').values()
        res.update((list(_values_view))[0])
    else:
        res = result
    N = len(quantities)
    n = int(np.ceil(np.sqrt(N)))
    fig = plt.figure()
    gs = gridspec.GridSpec(n, n)
    share = None
    for i, q in enumerate(quantities):
        ax = plt.subplot(gs[i // n, i % n], sharex=share)
        if q == 'sds':
            plot_sds(freq, res, ax=ax,
                     seismic_moment_method=seismic_moment_method,
                     seismic_moment_options=seismic_moment_options)
        else:
            vals = calc_dependent(q, res[DEPMAP[q]], freq, v0)
            ax.loglog(freq, vals, 'o-k')
            if q == 'error':
                ax.set_yscale('linear')
        ax.annotate(QLABELS[q], (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top')
        _set_gridlabels(ax, i, n, n, N, xtlabel=False, xlabel=xlabel)
        if share is None:
            share = ax
    ax.set_xlim(freq[0], freq[-1])
    _savefig(fig, **kwargs)


def plot_eventsites(result,
                    xlabel='frequency (Hz)',
                    ylabel='energy site amplification',
                    **kwargs):
    """Plot site amplification factors of one `~qopen.core.invert()` call"""
    freq = np.array(result['freq'])
    R = result['R']
    N = len(R)
    n = int(np.ceil(np.sqrt(N)))
    fig = plt.figure()
    gs = gridspec.GridSpec(n, n)
    share = None
    allR = []
    for i, station in enumerate(sorted(R)):
        allR.extend(R[station])
        ax = plt.subplot(gs[i // n, i % n], sharex=share, sharey=share)
        Rs = np.array(R[station], dtype=float)
        if not np.all(np.isnan(Rs)):
            ax.loglog(freq, Rs, 'o-k')
        l = station
        ax.annotate(l, (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top', size='small')
        _set_gridlabels(ax, i, n, n, N, xtlabel=False, ytlabel=False,
                        xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        if share is None:
            share = ax
    allR = np.array(allR, dtype=float)
    allR = allR[~np.isnan(allR)]
    if np.min(allR) != np.max(allR):
        ax.set_ylim(np.min(allR), np.max(allR))
    ax.set_xlim(freq[0], freq[-1])
    _savefig(fig, **kwargs)


def plot_results(result, v0=None, quantities=QUANTITIES, mean=None,
                 llim=None, Qlim=None, xlabel='frequency (Hz)',
                 **kwargs):
    """Plot results"""
    freq = np.array(result['freq'])
    N = len(quantities)
    n = int(np.ceil(np.sqrt(N)))
    fig = plt.figure()
    gs = gridspec.GridSpec(n, n)
    share = None
    # True for invert_events_simultaneously
    single_inversion = 'g0' not in list(result['events'].values())[0]
    if single_inversion:
        colres = result
    else:
        colres = collect_results(result, only=('g0', 'b', 'error', 'v0'))
    v0 = v0 or result['config'].get('v0') or colres['v0']
    colres.pop('v0', None)
    weights = 1 / np.array(colres['error']) if mean == 'weighted' else None
    robust = mean == 'robust'
    for i, q in enumerate(quantities):
        ax = plt.subplot(gs[i // n, i % n], sharex=share)
        if q == 'nobs':
            if single_inversion:
                nobs = 1
            else:
                nobs = np.sum(~np.isnan(colres['g0']), axis=0)
            ax.bar(freq, nobs, width=0.1 * freq, color='gray')
        else:
            value = colres[DEPMAP[q]]
            value = calc_dependent(q, value, freq, v0)
            if not single_inversion:
                freqs = np.repeat(freq[np.newaxis, :], value.shape[0], axis=0)
                ax.plot(freqs, value, 'o', ms=MS, color='gray', mec='gray')
            means, err1, err2 = gerr(
                value, axis=0, weights=weights, robust=robust)
            errs = (err1, err2)
            ax.errorbar(freq, means, yerr=errs, marker='o',
                        mfc='k', mec='k', color='m', ecolor='m')
            if q != 'error':
                ax.set_yscale('log')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.annotate(QLABELS[q], (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top')
        _set_gridlabels(ax, i, n, n, N, xtlabel=False, xlabel=xlabel)
        if share is None:
            share = ax
        if q in ('Qsc', 'Qi') and Qlim:
            ax.set_ylim(Qlim)
        if q in ('lsc', 'li') and llim:
            ax.set_ylim(llim)
    ax.set_xlim(freqlim(freq))
    _savefig(fig, **kwargs)


def plot_sites(result, mean=None,
               xlim=None, ylim=(1e-2, 1e2), nx=None,
               cmap='viridis_r', vmin=None, vmax=None,
               xlabel='frequency (Hz)', ylabel='site amplification',
               show_excluded=True, sortkey=None,
               **kwargs):
    """Plot site amplification factors"""
    freq = np.array(result['freq'])
    # True for invert_events_simultaneously
    single_inversion = 'R' not in list(result['events'].values())[0]
    if single_inversion:
        colres = result
        R = copy(colres['R'])
        for sta in R:
            R[sta] = np.array(R[sta], dtype=float)
        max_nobs = 1
    else:
        colres = collect_results(result, only=['R', 'error'])
        R = colres['R']
        max_nobs = np.max([np.sum(~np.isnan(r), axis=0) for r in R.values()])
    weights = 1 / np.array(colres['error']) if mean == 'weighted' else None
    robust = mean == 'robust'
    N = max_nobs > 1
    for station in sorted(R, key=sortkey):
        if not np.all(np.isnan(R[station])):
            N = N + 1
#    N = len(R) + (max_nobs > 1)
    fig = plt.figure()
    nx, ny, gs = _get_grid(N, nx=nx)
    if cmap is None:
        cmap = 'black'
    cmap = plt.get_cmap(cmap, max_nobs)
    if vmax is None:
        vmax = max_nobs + 0.5
    if vmin is None:
        vmin = 0.5
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    share = None
    i = 0
    do_not_plot_stations = []
    for station in sorted(R, key=sortkey):
        if np.all(np.isnan(R[station])):
            do_not_plot_stations.append(station)
            continue
        ax = plt.subplot(gs[i // nx, i % nx], sharex=share, sharey=share)
        means, err1, err2 = gerr(R[station], axis=0, weights=weights,
                                 robust=robust)
        errs = (err1, err2)
#        if not np.all(np.isnan(R[station])):
        if max_nobs == 1:
            kw = {'c': 'k'}
        else:
            nobs = 1. * np.sum(~np.isnan(R[station]), axis=0)
            kw = {'c': nobs, 'norm': norm, 'cmap': cmap}
        if not single_inversion:
            freqs = np.repeat(freq[np.newaxis, :], R[station].shape[0], axis=0)
            ax.plot(freqs, R[station], 'o', ms=MS, color='gray', mec='gray')
        ax.errorbar(freq, means, yerr=errs, marker=None,
                    color='m', ecolor='m')
        sc = ax.scatter(freq, means, s=4 * MS ** 2,
                        marker='o', zorder=10,
                        linewidth=0.5,
                        **kw)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.annotate(station, (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top', size='x-small')
        _set_gridlabels(ax, i, nx, ny, N-(max_nobs > 1),
                        xtlabel=False, ytlabel=False,
                        xlabel=xlabel, ylabel=ylabel)
        if share is None:
            share = ax
        i += 1
    ax.set_xlim(xlim or freqlim(freq))
    if ylim:
        ax.set_ylim(ylim)
    if max_nobs != 1:
        ax = plt.subplot(gs[(N - 1) // nx, (N - 1) % nx])
        ax.set_axis_off()
        fig.colorbar(sc, ax=ax, shrink=0.9, format='%d', label='nobs',
                     ticks=np.arange(0, max_nobs + 1, max(1, max_nobs // 5)))
    if len(do_not_plot_stations) > 0 and show_excluded:
        ax.annotate('excluded: ' + ' '.join(do_not_plot_stations),
                     (1, 0), (-5, 5), 'figure fraction', 'offset points',
                     ha='right', size='x-small')
    _savefig(fig, **kwargs)


def _get_grid(N, nx=None):
    if nx is None:
        nx = ny = int(np.ceil(np.sqrt(N)))
    else:
        ny = 1 + (N-1) // nx
    gs = gridspec.GridSpec(ny, nx)
    return nx, ny, gs


def plot_all_sds(result, seismic_moment_method=None,
                 seismic_moment_options=None,
                 xlim=None, ylim=None, nx=None,
                 annotate=None, va='top',
                 annotate_label=None,
                 annotate_evid=True,
                 plot_only_ids=None,
                 cmap='viridis_r', vmin=None, vmax=None,
                 colors=None,
                 xlabel='frequency (Hz)',
                 ylabel=r'source displacement spectrum $\omega$M (Nm)',
                 **kwargs):
    """Plot all source displacement spectra with fitted source models"""
    freq = np.array(result['freq'])
    conf = result.get('config', {})
    smm = seismic_moment_method or conf.get('seismic_moment_method')
    smo = seismic_moment_options or conf.get('seismic_moment_options', {})
#    fc = seismic_moment_options.pop('fc', None)
    result = result['events']
    if plot_only_ids:
        result = {id_: r for id_, r in result.items() if id_ in plot_only_ids}
    if ('nstations' not in list(result.values())[0] or cmap is None or
            colors is not None):
        max_nobs = 1  # single inversion or uniform color
    else:
        nobs = [evres['nstations'] for evres in result.values()]
        max_nobs = np.max(nobs)
    N = len(result) + (max_nobs != 1)
#    n = int(np.ceil(np.sqrt(N)))
    fig = plt.figure()
#    gs = gridspec.GridSpec(n, n)
    nx, ny, gs = _get_grid(N, nx=nx)
    share = None
    if annotate is None:
        annotate = nx < 7
    color = None
    for i, evid in enumerate(result):
        if colors is not None:
            try:
                color = colors[evid]
            except:
                color = colors
        ax = plt.subplot(gs[i // nx, i % nx], sharex=share, sharey=share)
        sc = plot_sds(freq, result[evid], seismic_moment_method=smm, va=va,
                      seismic_moment_options=smo, ax=ax, annotate=annotate,
                      annotate_label=annotate_label,
                      cmap=cmap, vmin=vmin, vmax=vmax, max_nobs=max_nobs,
                      color=color)
        if annotate_evid:
            ax.annotate(evid, (0, 0), (5, 5), 'axes fraction',
                        'offset points', ha='left', va='bottom',
                        size='x-small')
        _set_gridlabels(ax, i, nx, ny, N - (max_nobs != 1),
                        xtlabel=False, ytlabel=False,
                        xlabel=xlabel, ylabel=ylabel)
        if share is None:
            share = ax
    ax.autoscale()
    ax.set_xlim(xlim or freqlim(freq))
    if ylim:
        ax.set_ylim(ylim)
    if max_nobs != 1:
        if (N-1) % nx == (nx - 1) // 2 and xlabel:
            N += 1
        ax = plt.subplot(gs[(N-1) // nx, (N-1) % nx])
        ax.set_axis_off()
        fig.colorbar(sc, ax=ax, shrink=0.9, format='%d', label='nstations',
                     ticks=np.arange(0, max_nobs + 1, max(1, max_nobs // 5)))
    _savefig(fig, **kwargs)


def _secondary_yaxis_seismic_moment(ax, inverse=False):
    Mw2M0 = lambda Mw:  moment_magnitude(Mw, inverse=True)
    funcs = (moment_magnitude, Mw2M0) if inverse else (Mw2M0, moment_magnitude)
    ax2 = ax.secondary_yaxis('right', functions=funcs)
    ax2.set_yscale('log')
    ylabel = 'moment magnitude' if inverse else 'seismic moment $M_0$ (Nm)'
    ax2.set_ylabel(ylabel)
    return ax2


def plot_mags(result, xlim=None, ylim=None, plot_only_ids=None,
              xlabel='M from catalog', ylabel='Mw from inversion',
              secondary_yaxis=True,
              **kwargs):
    """Plot Qopen moment magnitudes versus catalogue magnitudes"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    temp = [(r['Mcat'], r['Mw']) for id_, r in result['events'].items()
            if r.get('Mcat') is not None and r.get('Mw') is not None and
            (plot_only_ids is None or id_ in plot_only_ids)]
    if len(temp) == 0:
        return
    Mcat, Mw = zip(*temp)
    ax.plot(Mcat, Mw, 'ok', ms=MS)
    if xlim is not None:
        mmin, mmax = xlim
    else:
        mmin, mmax = np.min(Mcat), np.max(Mcat)
    m = np.linspace(mmin, mmax, 100)

    if len(Mw) > 3:
        a, b = linear_fit(Mw, Mcat)
        ax.plot(m, a * m + b, '-', color='0.5', label='%.2fM %+.2f' % (a, b))
#    if len(Mw) > 3:
#        _, b2 = linear_fit(Mw, Mcat, m=1)
#        ax.plot(m, m + b2, '--m', label='M %+.2f' % (b2,))
    if len(Mw) > 3:
        ax.legend(loc='lower right')
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if secondary_yaxis:
        _secondary_yaxis_seismic_moment(ax)
    _savefig(fig, **kwargs)
