# Copyright 2015-2016 Tom Eulenfeld, MIT license
"""Plotting functions"""

# The following lines are for Py2/Py3 support with the future module.
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (  # analysis:ignore
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)

from copy import copy
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os

from qopen.core import get_pair, collect_results
from qopen.source import source_model
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
          'W': 'W', 'omM': 'omM', 'sds': 'omM', 'error': 'error'}


def calc_dependent(quantity, value, freq=None, v0=None):
    """Calculate dependent value (Qsc, Qi, lsc, li) from g0 and b

    :param str quantity: one of Qsc, Qi, lsc, li
    :param value: value of g0 or b depending on reuqested quantity
    :param freq: frequency in Hz (needed for some calculations)
    :param v0: velocity in m/s (needed for some calculations)
    :return: value of quantity"""
    q = quantity
    val = np.array(value, dtype=float)
    if q in ('g0', 'b', 'W', 'omM', 'error'):
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


def _savefig(fig, title=None, fname=None, dpi=None):
    if title:
        extra = (fig.suptitle(title),)
    else:
        extra = ()
    if fname:
        path = os.path.dirname(fname)
        if path != '' and not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(fname, bbox_inches='tight', bbox_extra_artists=extra,
                    dpi=dpi)
        plt.close(fig)


def _set_gridlabels(ax, i, nx, ny, N, xlabel='frequency (Hz)', ylabel=None):
    if i % nx != 0 and ylabel:
        plt.setp(ax.get_yticklabels(), visible=False)
    elif i // nx == (ny - 1) // 2 and ylabel:
        ax.set_ylabel(ylabel)
    if i < N - nx and xlabel:
        plt.setp(ax.get_xticklabels(), visible=False)
    elif i % nx == (nx - 1) // 2 and N - i <= nx and xlabel:
        ax.set_xlabel(xlabel)


def plot_energies(energies,
                  bulk_window=None, coda_window=None, downsample_to=None,
                  xlim_lin=None, xlim_log=None,
                  figsize=None, **kwargs):
    """
    Plot observed spectral energy densities on different scales (linear, log)
    """
    gs = gridspec.GridSpec(2 * len(energies), 2)
    gs.update(wspace=0.05)
    fig = plt.figure(figsize=figsize)
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
            plt.setp(ax.get_xticklabels(), visible=False)
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
    plt.setp(ax2.get_xticklabels(), visible=True)
    plt.setp(ax3.get_xticklabels(), visible=True)
    _savefig(fig, **kwargs)


def plot_lstsq(rec, ax=None, fname=None, base=np.e):
    """Plot solution of weighted least squares inversion"""
    err, g0, b, W, R, info = rec
    tcoda, tbulk, Ecoda, Ebulk, Gcoda, Gbulk = info
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    tmin = min(tcoda[i][0] for i in range(len(tcoda)))
    tmax = max(tcoda[i][-1] for i in range(len(tcoda)))
    for i in range(len(tcoda)):
        offset = R[i % len(R)] * W[i // len(R)] / W[0]
#        offset = R[i] if len(W) == 1 else C[i]
#        Bci = np.log(Ecoda[i]) - np.log(FS * Gcoda[i]) - np.log(offset)
        Bci = np.log(Ecoda[i]) - np.log(Gcoda[i]) - np.log(offset)
        ax.plot(tcoda[i], Bci / np.log(base), color='0.7')
    for i in range(len(tbulk)):
        offset = R[i % len(R)] * W[i // len(R)] / W[0]
#        offset = R[i] if len(W) == 1 else C[i]
#        Bbi = np.log(Ebulk[i]) - np.log(FS * Gbulk[i]) - np.log(offset)
        Bbi = np.log(Ebulk[i]) - np.log(Gbulk[i]) - np.log(offset)
        ax.plot(tbulk[i], Bbi / np.log(base), 'o', color='0.4', mec='0.4',
                ms=MS)
        tmin = min(tmin, tbulk[i])
    t = np.linspace(tmin, tmax, 100)
    ax.plot(t, (np.log(W[0]) - b * t) / np.log(base), color='k')
    ax.set_xlim(right=tmax)
    if fig and fname:
        _savefig(fig, fname=fname)


def plot_optimization(record, record_g0, num=7, fname=None, title=None,
                      figsize=None, **kwargs):
    """Plot some steps of optimization"""
    fig = plt.figure(figsize=figsize)
    if num > 1:
        n = (num + 1) // 2
        gs = gridspec.GridSpec(n, n)
        ax = plt.subplot(gs[1:, 0:-1])
        share = None
    else:
        ax = fig.add_subplot(111)
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
            plot_lstsq(rec, ax=ax2)
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
    ax.set_xlabel(r'g$_0$ ($\mathrm{m}^{-1}$)')
    # yl = (r'error $\mathrm{rms}\left(\ln\frac{E_{\mathrm{obs}, ij}}'
    #      r'{E_{\mathrm{mod}, ij}}\right)$')
    ax.set_ylabel(r'misfit $\epsilon$')
    _savefig(fig, fname=fname, **kwargs)


def _get_times(tr):
    t0 = tr.stats.starttime - tr.stats.origintime
    return np.arange(len(tr)) * tr.stats.delta + t0


def plot_fits(energies, g0, b, W, R, v0, info, G_func,
              smooth=None, smooth_window='bartlett',
              xlim=None, ylim=None, fname=None, title=None, figsize=None,
              **kwargs):
    """Plot fits of spectral energy densities"""
    tcoda, tbulk, Ecoda, Ebulk, Gcoda, Gbulk = info
    N = len(energies)
    n = int(np.ceil(np.sqrt(N)))
    fig = plt.figure(figsize=figsize)
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
        index = np.argwhere(Emod < 1e-30)[-1]
        Emod[index] = 1e-30
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
        _set_gridlabels(ax, i, n, n, N, xlabel='time (s)',
                        ylabel=r'E (Jm$^{-3}$Hz$^{-1}$)')
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
    _savefig(fig, fname=fname, title=title, **kwargs)


def plot_sds(freq, result, ax=None, fname=None,
             annotate=False,
             seismic_moment_method=None, seismic_moment_options={}):
    """Plot source displacement spectrum and fitted source model"""
    freq = np.array(freq)
    omM = np.array(result['omM'], dtype=np.float)
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
    if seismic_moment_method == 'mean':
        ax.loglog(freq, omM, 'o-', color='gray', mec='gray')
        ax.loglog(freq[freq < fc], omM[freq < fc], 'o-k')
    elif seismic_moment_method in ('fit', 'robust_fit'):
        ax.loglog(freq, omM, 'ok')
        if M0 and fc:
            f = np.linspace(freq[0] / 1.5, freq[-1] * 1.5, 100)
            omM2 = source_model(f, **smo)
            ax.loglog(f, omM2, '-k')
    else:
        ax.loglog(freq, omM, 'o-k')
    labels = []
    if M0:
        ax.axhline(M0, ls='--', color='k')
        if annotate:
            labels.append(r'M$_0$=%.1e Nm' % M0)
    labels = {'M0': r'M$_0$=%.1e Nm',
              'fc': r'f$_{\rm{c}}$=%.1f Hz',
              'n': 'n=%.1f', 'gamma': r'$\gamma$=%.2f'}
    labels = [labels[key] % result[key] for key in obs if key in result]
    if len(labels) > 0 and annotate:
        ax.annotate('\n'.join(labels), (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top', size='x-small')

    if fig and fname:
        _savefig(fig, fname=fname)


def plot_eventresult(result, v0=None, fname=None, title=None,
                     quantities=QUANTITIES_EVENT,
                     seismic_moment_method=None, seismic_moment_options={},
                     figsize=None):
    """Plot all results of one `~qopen.core.invert()` call"""
    v0 = v0 or result.get('v0') or result.get('config', {}).get('v0')
    freq = np.array(result['freq'])
    res = copy(result)
    _values_view = res.pop('events').values()
    res.update((list(_values_view))[0])
    N = len(quantities)
    n = int(np.ceil(np.sqrt(N)))
    fig = plt.figure(figsize=figsize)
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
        ax.annotate(QLABELS[q], (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top')
        _set_gridlabels(ax, i, n, n, N)
        if share is None:
            share = ax
    ax.set_xlim(freq[0], freq[-1])
    _savefig(fig, fname=fname, title=title)


def plot_eventsites(result, fname=None, title=None, figsize=None):
    """Plot site amplification factors of one `~qopen.core.invert()` call"""
    freq = np.array(result['freq'])
    R = result['R']
    N = len(R)
    n = int(np.ceil(np.sqrt(N)))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n, n)
    share = None
    allR = []
    for i, station in enumerate(sorted(R)):
        allR.extend(R[station])
        ax = plt.subplot(gs[i // n, i % n], sharex=share, sharey=share)
        Rs = np.array(R[station], dtype=np.float)
        if not np.all(np.isnan(Rs)):
            ax.loglog(freq, Rs, 'o-k')
        l = station
        ax.annotate(l, (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top', size='small')
        _set_gridlabels(ax, i, n, n, N, ylabel='site correction')
        if share is None:
            share = ax
    allR = np.array(allR, dtype=np.float)
    allR = allR[~np.isnan(allR)]
    if np.min(allR) != np.max(allR):
        ax.set_ylim(np.min(allR), np.max(allR))
    ax.set_xlim(freq[0], freq[-1])
    _savefig(fig, fname=fname, title=title)


def plot_results(result, v0=None, fname=None, title=None,
                 quantities=QUANTITIES, mean=None,
                 llim=None, Qlim=None, figsize=None):
    """Plot results"""
    freq = np.array(result['freq'])
    N = len(quantities)
    n = int(np.ceil(np.sqrt(N)))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n, n)
    share = None
    g0, b, error, R, _, _, v02 = collect_results(result)
    v0 = v0 or result['config'].get('v0') or v02
    result = {'g0': g0, 'b': b, 'error': error, 'R': R}
    weights = 1 / np.array(error) if mean == 'weighted' else None
    robust = mean == 'robust'
    for i, q in enumerate(quantities):
        ax = plt.subplot(gs[i // n, i % n], sharex=share)
        if q == 'nobs':
            nobs = np.sum(~np.isnan(g0), axis=0)
            ax.bar(freq, nobs, width=0.1 * freq, color='gray')
        else:
            value = result[DEPMAP[q]]
            value = calc_dependent(q, value, freq, v0)
            freqs = np.repeat(freq[np.newaxis, :], value.shape[0], axis=0)
            ax.loglog(freqs, value, 'o', ms=MS, color='gray', mec='gray')
            means, err1, err2 = gerr(
                value, axis=0, weights=weights, robust=robust)
            errs = (err1, err2)
            ax.errorbar(freq, means, yerr=errs, marker='o',
                        mfc='k', mec='k', color='m', ecolor='m')
        ax.annotate(QLABELS[q], (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top')
        _set_gridlabels(ax, i, n, n, N, ylabel=None)
        if share is None:
            share = ax
        if q in ('Qsc', 'Qi') and Qlim:
            ax.set_ylim(Qlim)
        if q in ('lsc', 'li') and llim:
            ax.set_ylim(llim)
    ax.set_xlim(freqlim(freq))
    _savefig(fig, fname=fname, title=title)


def plot_sites(result, fname=None, title=None, mean=None,
               xlim=None, ylim=(1e-2, 1e2), nx=None, figsize=None):
    """Plot site amplification factors"""
    freq = np.array(result['freq'])
    g0, b, error, R, _, _, _ = collect_results(result)
    weights = 1 / np.array(error) if mean == 'weighted' else None
    robust = mean == 'robust'
    max_nobs = np.max([np.sum(~np.isnan(r), axis=0) for r in R.values()])
    N = max_nobs > 1
    for station in sorted(R):
        if not np.all(np.isnan(R[station])):
            N = N + 1
#    N = len(R) + (max_nobs > 1)
#    for i
    fig = plt.figure(figsize=figsize)
    nx, ny, gs = _get_grid(N, nx=nx)
    cmap = plt.get_cmap('hot_r', max_nobs)
    norm = mpl.colors.Normalize(vmin=0.5, vmax=max_nobs + 0.5)
    share = None
    i = 0
    for station in sorted(R):
        if np.all(np.isnan(R[station])):
            continue
        ax = plt.subplot(gs[i // nx, i % nx], sharex=share, sharey=share)
        means, err1, err2 = gerr(R[station], axis=0, weights=weights,
                                 robust=robust)
        nobs = 1. * np.sum(~np.isnan(R[station]), axis=0)
        errs = (err1, err2)
        freqs = np.repeat(freq[np.newaxis, :], R[station].shape[0], axis=0)
#        if not np.all(np.isnan(R[station])):
        if max_nobs == 1:
            kwargs = {'c': 'k'}
        else:
            kwargs = {'c': nobs, 'norm': norm, 'cmap': cmap}
        ax.loglog(freqs, R[station], 'o', ms=MS, color='gray', mec='gray')
        ax.errorbar(freq, means, yerr=errs, marker=None,
                    color='m', ecolor='m')
        sc = ax.scatter(freq, means, s=4 * MS ** 2,
                        marker='o', zorder=10,
                        linewidth=0.5,
                        **kwargs)
        ax.annotate(station, (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top', size='x-small')
        _set_gridlabels(ax, i, nx, ny, N, ylabel='amplification factor')
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
    _savefig(fig, fname=fname, title=title)


def _get_grid(N, nx=None):
    if nx is None:
        nx = ny = int(np.ceil(np.sqrt(N)))
    else:
        ny = 1 + (N-1) // nx
    gs = gridspec.GridSpec(ny, nx)
    return nx, ny, gs


def plot_all_sds(result, seismic_moment_method=None,
                 seismic_moment_options=None,
                 fname=None, title=None, xlim=None, ylim=None, nx=None,
                 figsize=None):
    """Plot all source displacement spectra with fitted source models"""
    freq = np.array(result['freq'])
    conf = result.get('config', {})
    smm = seismic_moment_method or conf.get('seismic_moment_method')
    smo = seismic_moment_options or conf.get('seismic_moment_options', {})
#    fc = seismic_moment_options.pop('fc', None)
    result = result['events']
    N = len(result)
#    n = int(np.ceil(np.sqrt(N)))
    fig = plt.figure(figsize=figsize)
#    gs = gridspec.GridSpec(n, n)
    nx, ny, gs = _get_grid(N, nx=nx)
    share = None
    for i, evid in enumerate(sorted(result)):
        ax = plt.subplot(gs[i // nx, i % nx], sharex=share, sharey=share)
        plot_sds(freq, result[evid], seismic_moment_method=smm,
                 seismic_moment_options=smo, ax=ax, annotate=nx < 7)
        ax.annotate(evid, (0, 0), (5, 5), 'axes fraction',
                    'offset points', ha='left', va='bottom', size='x-small')
        _set_gridlabels(ax, i, nx, ny, N, ylabel=r'$\omega$M (Nm)')
        if share is None:
            share = ax
    ax.autoscale()
    ax.set_xlim(xlim or freqlim(freq))
    if ylim:
        ax.set_ylim(ylim)
    _savefig(fig, fname=fname, title=title)


def plot_mags(result, fname=None, title=None, xlim=None, ylim=None,
              figsize=None):
    """Plot Qopen moment magnitudes versus catalogue magnitudes"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    temp = [(r['Mcat'], r['Mw']) for r in result['events'].values()
            if r.get('Mcat') is not None and r.get('Mw') is not None]
    if len(temp) == 0:
        return
    Mcat, Mw = zip(*temp)
    ax.plot(Mcat, Mw, 'ok', ms=MS)
    if xlim is not None:
        mmin, mmax = xlim
    else:
        mmin, mmax = np.min(Mcat), np.max(Mcat)
    m = np.linspace(mmin, mmax, 100)

    if len(Mw) > 2:
        a, b = linear_fit(Mw, Mcat)
        ax.plot(m, a * m + b, '-m', label='%.2fM %+.2f' % (a, b))
    if len(Mw) > 3:
        _, b2 = linear_fit(Mw, Mcat, m=1)
        ax.plot(m, m + b2, '--m', label='M %+.2f' % (b2,))
    if len(Mw) > 2:
        ax.legend(loc='lower right')
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel('M from catalog')
    ax.set_ylabel('Mw from inversion')
    _savefig(fig, fname=fname, title=title)
