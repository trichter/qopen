# Author: Tom Richter
"""
Radiative Transfer: Approximative interpolation solution of Paasschens (1997)

r ... Distance to source in m
t ... Time after source in s
c ... Velocity in m/s
l ... transport mean free path in m
la ... absorption length in m
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
import scipy

from qopen.util import smooth as smooth_func

# Correction factor for free surface
# According to Emoto (2010) or Sato page 394 figure 9.39 this is around 4
# for S-waves.
FS = 4

def Gb(t, c, g0, var='t'):
    """Bulk term of RT solution"""
    if var == 'r':
        fac = 1  # * delta(r - c * t)
    else:
        fac = 1 / c  # * delta(t - r / c)
    return fac * np.exp(-c * t * g0) / (4 * np.pi * c ** 2 * t ** 2)


def F(x):
    return np.sqrt(1 + 2.026 / x)


def Gcoda_red(r, t):
    """Coda term for t>r in reduced coordinates r'=rg, t'=tgc"""
    a = 1 - r ** 2 / t ** 2
    o = (a ** 0.125 / (4 * np.pi * t / 3) ** 1.5 *
         np.exp(t * (a ** 0.75 - 1)) * F(t * a ** 0.75))
    return o

def Gr_red(r, t, eps=0):
    """Coda term as a function of r in reduced coordinates  r'=rg0, t'=tcg0

    All values for r>t-eps will be 0"""
    o1 = np.zeros(np.count_nonzero(t - r <= eps))
    r = r[t - r > eps]
    o2 = Gcoda_red(r, t)
    return np.hstack((o2, o1))


def G_red(r, t, eps='dt'):
    """Coda term as a function of t in reduced coordinates  r'=rg0, t'=tcg0

    All values for t<r+eps will be 0"""
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


def G(r, t, c, g0, eps='dt', include_bulk=True):
    """Full Green's function with bulk term (optional)"""
    t_isarray = isinstance(t, np.ndarray)
    r_isarray = isinstance(r, np.ndarray)
    if t_isarray and r_isarray:
        raise ValueError('Only one of t or r are allowed to be numpy arrays')
    elif (not t_isarray and eps == 'dt') or (not r_isarray and eps == 'dr'):
        eps=0
    elif t_isarray:
        eps = float(t[1] - t[0])
    elif r_isarray:
        eps = float(r[1] - r[0])
    if r_isarray:
        G_ = Gr_red(r * g0, t * c * g0, eps=eps * g0) * g0 ** 3
        msg = 'include_bulk=True is not implemented for G as a function of r'
        if include_bulk:
            raise NotImplementedError(msg)
    elif t_isarray:
        G_ = G_red(r * g0, t * c * g0, eps=eps * c * g0) * g0 ** 3
        if not t_isarray:
            return FS * G_
        i = np.count_nonzero(t - r / c <= eps)
        G_[:i] = 0
        #j = len(G_) - np.count_nonzero(G_)
        #if i != j:
        #    print(i, j, len(G_))
        #assert i == j

        if include_bulk and 0 < i < len(G_) and eps > 0:
            tend = t[i - 1] + 0.5 * eps
            intG_, err = intG(r, tend, c, g0)
            G_[i - 1] = G_[i] + intG_ / eps
    else:
        G_ = G_red(r * g0, t * c * g0, eps=eps * c * g0) * g0 ** 3
    # include surface correction
    return FS * G_


def intGcoda_red(r, t, eps=1e-5, N=1):
    """Time-Integral of reduced coda term from t0=r to t"""
    if t < r or eps >= t-r:
        return 0, 0
    f = lambda t1: G_red(r, t1, eps=0)
    t2 = r + np.logspace(np.log10(eps), np.log10(t - r), N + 1)
    o = [scipy.integrate.quad(f, t2[i], t2[i + 1], limit=300) for i in range(N)]
    o = np.sum(np.array(o), axis=0)
    return o[0], o[1]



def intG(r, t, c, g0, eps=1e-4, N=1, include_bulk=True):
    """Time-Integral of full Green's function from t0=r/c to t"""
    if t < r / c:
        return 0, 0
    intGr, err = intGcoda_red(r * g0, t * c * g0, eps=eps, N=N)
    b = Gb(r / c, c, g0, var='t') if include_bulk else 0
    # include surface correction
    return FS * (intGr * g0 ** 2 / c + b), err * g0 ** 2 / c

def Gsmooth(r, t, c, g0, smooth=None, smooth_window='bartlett'):
    """Smoothed full Green's function"""
    if smooth is None:
        G_ = G(r, t, c, g0)
    else:
        dt = t[1] - t[0]
        samples = int(round(smooth / dt))
        N1 = (samples - 1) // 2
        N2 = samples // 2
        t_ = np.hstack((t[0] - N1 * dt + np.arange(N1) * dt, t,
                        t[-1] + dt + np.arange(N2) * dt))
        G_ = G(r, t_, c, g0)
        G_ = smooth_func(G_, samples, method=None, window=smooth_window)
    return G_

#def rt_3D_b(t, c, l, var='t'):
#    """Bulk term of RT solution"""
#    if var == 'r':
#        fac = 1  # * delta(r - c * t)
#    else:
#        fac = 1 / c  # * delta(t - r / c)
#    return fac * np.exp(-c * t / l) / (4 * np.pi * c ** 2 * t ** 2)
#
#
#def G2(x):
#    return np.sqrt(1 + 2.026 / x)
#
#
#def rt_3D_approx(r, t, c, l, dt=None, bulk=True):
#    """Approximative solution of RT with bulk term"""
#    if isinstance(t, (int, float)):
#        t = np.array([t])
#        if dt is None:
#            dt = 0.005
#    elif dt is None:
#        dt = max(0.5 * (t[1] - t[0]), 0.001)
#    out1 = np.zeros(np.count_nonzero(c * t - r <= dt))
#    t = t[c * t - r > dt]
#    arg0 = r ** 2 / c ** 2 / t ** 2
#    arg1 = c * t / l
#    # Heaviside(c * t - r) *
#    out2 = ((1 - arg0) ** (1 / 8) / (4 * np.pi * l * c * t / 3) ** (3 / 2) *
#            np.exp(arg1 * ((1 - arg0) ** (3 / 4) - 1)) *
#            G2(arg1 * (1 - arg0) ** (3 / 4)))
#    if bulk and len(out1) > 0 and len(out2) > 0:
#        cor, err = integrate_t_rt_3D_approx(r, r / c + dt, c, l)
#        out2[0] = out2[0] + cor / (t[1] - t[0])
#
#    return np.hstack((out1, out2))
#
#
#def rt_3D_approx_smooth(r, t, c, l, smooth=None, smooth_window='bartlett'):
#    if smooth is None:
#        G = rt_3D_approx(r, t, c, l)
#    else:
#        dt = t[1] - t[0]
#        samples = int(round(smooth / dt))
#        N1 = (samples - 1) // 2
#        N2 = samples // 2
#        t_ = np.hstack((t[0] - N1 * dt + np.arange(N1) * dt, t,
#                        t[-1] + dt + np.arange(N2) * dt))
#        G = rt_3D_approx(r, t_, c, l)
#        G = smooth_func(G, samples, method=None, window=smooth_window)
#    return G
#
#
#def integrate_t_rt_3D_approx(r, t, c, l, N=1):
#    """Time intergal of approximative solution from r/c+0.001 to t"""
#    if t <= r / c:
#        return 0
#    res = [np.array((rt_3D_b(r / c, c, l, var='t'), 0))]
#    f = lambda t1: rt_3D_approx(r, t1, c, l, dt=0, bulk=False)
#    t = r / c + np.logspace(-3, np.log10(t - r / c), N + 1)
#    for i in range(N):
#        res0 = scipy.integrate.quad(f, t[i], t[i + 1], limit=300)
#        res.append(np.array(res0))
#    res = np.sum(res, axis=0)
#    return res[0], res[1]


def plot_t(c, g0, r, t=None, N=100, log=False):
    """Plot solution as a function of time"""
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
    """Plot solution as a function of distance"""
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
    choices = ('calc', 'plot-t', 'plot-r', 'integrate-t')
    p.add_argument('command', help='command', choices=choices)
    p.add_argument('c', help='velocity', type=float)
    p.add_argument('l', help='transport mean free path', type=float)
    p.add_argument('-r', help='distance from source', type=float)
    p.add_argument('-t', help='time after source', type=float)
    p.add_argument('--log', help='log plot', action='store_true')
    msg = 'absorption length'
    p.add_argument('-a', '--absorption', help=msg, type=float)
    msg = 'calculate ballistic term, ignore argument -r and -a'
    p.add_argument('-b', '--ballistic', help=msg, action='store_true')
    args = p.parse_args(args)
    r, t, c, l, la = args.r, args.t, args.c, args.l, args.absorption
    com = args.command
    if com == 'calc':
        if args.ballistic:
            print(Gb(t, c, 1/l))
        else:
            res = G(r, t, c, 1/l)
            if la:
                res = res * np.exp(-c * t / la)
            print(res)
    elif com == 'plot-t':
        plot_t(c, 1/l, r, t=t, log=args.log)
    elif com == 'plot-r':
        plot_r(c, 1/l, t, r=r, log=args.log)
    elif com == 'integrate-t':
        print(intG(r, t, c, 1/l))


if __name__ == '__main__':
    main()
