# Copyright 2015-2016 Tom Eulenfeld, MIT license
"""
Radiative Transfer: Approximative interpolation solution of Paasschens (1997)

Use the ``qopen-rt`` command line script to calculate or plot the
spectral energy densitiy Green's function.

Used variables::

    r ... Distance to source in m
    t ... Time after source in s
    c ... Velocity in m/s
    l ... (transport) mean free path in m
    la ... absorption length in m
    g0 = 1/l ... (transport) scattering coefficient



.. note::
    The formula for the Green's function is valid for the
    scattering coefficient (g0) under assumption of isotropic scattering.
    However, g0 is used by `qopen.core` module as transport scattering
    coefficient (g*) under the assumption of non-isotropic scattering.
    ``g*=g0`` is a reasonable assumption under these conditions (see paper).

"""

# The following lines are for Py2/Py3 support with the future module.
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (  # analysis:ignore
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)

import argparse
import numpy as np


def Gdirect(t, c, g0, var='t'):
    """Direct wave term of radiative transfer solution"""
    if var == 'r':
        fac = 1  # * delta(r - c * t)
    else:
        fac = 1 / c  # * delta(t - r / c)
    return fac * np.exp(-c * t * g0) / (4 * np.pi * c ** 2 * t ** 2)


def F(x):
    return np.sqrt(1 + 2.026 / x)


def Gcoda_red(r, t):
    """Coda term for ``t>r`` in reduced coordinates ``r'=rg0, t'=tg0c``"""
    a = 1 - r ** 2 / t ** 2
    o = (a ** 0.125 / (4 * np.pi * t / 3) ** 1.5 *
         np.exp(t * (a ** 0.75 - 1)) * F(t * a ** 0.75))
    return o


def Gr_red(r, t, eps=0):
    """
    Coda term as a function of r in reduced coordinates  ``r'=rg0, t'=tcg0``

    All values for ``r>t-eps`` will be 0"""
    o1 = np.zeros(np.count_nonzero(t - r <= eps))
    r = r[t - r > eps]
    o2 = Gcoda_red(r, t)
    return np.hstack((o2, o1))


def G_red(r, t, eps='dt'):
    """Coda term as a function of t in reduced coordinates  ``r'=rg0, t'=tcg0``

    All values for ``t<r+eps`` will be 0"""
    if eps == 'dt':
        eps = t[1] - t[0]
    if isinstance(t, (int, float)):
        if t - r <= eps:
            return 0
        else:
            return Gcoda_red(r, t)
    o1 = np.zeros(np.count_nonzero(t - r <= eps))
    t = t[t - r > eps]
    o2 = Gcoda_red(r, t)
    return np.hstack((o1, o2))


def G(r, t, c, g0, eps=None, include_direct=True):
    """Full Green's function with direct wave term (optional)"""
    t_isarray = isinstance(t, np.ndarray)
    r_isarray = isinstance(r, np.ndarray)
    if t_isarray and r_isarray:
        raise ValueError('Only one of t or r are allowed to be numpy arrays')
    elif eps is not None and (t_isarray or r_isarray):
        raise ValueError('eps must be None if t or r is an array')
    elif t_isarray:
        eps = float(t[1] - t[0])
        eps_nodim = eps * c * g0
    elif r_isarray:
        eps = float(r[1] - r[0])
        eps_nodim = eps * g0
    else:
        if eps is None:
            eps = 0
        eps_nodim = eps * c * g0
    G_redfunc = Gr_red if r_isarray else G_red
    G_ = G_redfunc(r * g0, t * c * g0, eps=eps_nodim) * g0 ** 3
    if r_isarray and include_direct:
        msg = 'include_direct=True is not implemented for G as a function of r'
        raise NotImplementedError(msg)
    elif t_isarray:
        i = np.count_nonzero(t - r / c <= eps)
        G_[:i] = 0
        if include_direct and 0 < i < len(G_) and eps > 0:
            b = Gdirect(r / c, c, g0, var='t') if include_direct else 0
            G_[i - 1] = b / float(t[1] - t[0])
    return G_


def plot_t(c, g0, r, t=None, N=100, log=False):
    """Plot Green's function as a function of time"""
    import matplotlib.pyplot as plt
    if t is None:
        t = 10 * r / c
    ts = r / c + np.logspace(-3, np.log10(t - r / c), N)
    G_ = G(r, ts, c, g0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if log:
        ax.semilogy(ts, G_)
    else:
        ax.plot(ts, G_)
    ax.set_xlim((0, t))
    ax.set_ylabel('G')
    ax.set_xlabel('t (s)')
    plt.show()


def plot_r(c, g0, t, r=None, N=100, log=False):
    """Plot Green's function as a function of distance"""
    import matplotlib.pyplot as plt
    if r is None:
        r = 10 * c * t
    rs = np.linspace(0, c * t - 0.1, N)
    G_ = G(rs, t, c, g0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if log:
        ax.semilogy(rs, G_)
    else:
        ax.plot(rs, G_)
    ax.set_xlim((0, r))
    ax.set_ylabel('G')
    ax.set_xlabel('r (m)')
    plt.show()


def main(args=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    choices = ('calc', 'plot-t', 'plot-r')
    p.add_argument('command', help='command', choices=choices)
    p.add_argument('c', help='velocity', type=float)
    p.add_argument('l', help='transport mean free path', type=float)
    p.add_argument('-r', help='distance from source', type=float)
    p.add_argument('-t', help='time after source', type=float)
    p.add_argument('--log', help='log plot', action='store_true')
    msg = 'absorption length'
    p.add_argument('-a', '--absorption', help=msg, type=float)
    msg = 'calculate direct wave term, ignore argument -r and -a'
    p.add_argument('-d', '--direct-wave', help=msg, action='store_true')
    args = p.parse_args(args)
    r, t, c, l, la = args.r, args.t, args.c, args.l, args.absorption
    com = args.command
    if com == 'calc':
        if args.direct_wave:
            print(Gdirect(t, c, 1/l))
        else:
            res = G(r, t, c, 1/l)
            if la:
                res = res * np.exp(-c * t / la)
            print(res)
    elif com == 'plot-t':
        plot_t(c, 1/l, r, t=t, log=args.log)
    elif com == 'plot-r':
        plot_r(c, 1/l, t, r=r, log=args.log)


if __name__ == '__main__':
    main()
