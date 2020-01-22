# Copyright 2015-2019 Tom Eulenfeld, MIT license
"""
Radiative Transfer: 3D approximative interpolation solution of Paasschens (1997)

Use the ``qopen-rt`` command line script to calculate or plot the
spectral energy densitiy Green's function.

This module also defines analytical Green's functions for 1D and 2D isotropic
radiative transfer. The Green's function used by the ``qopen`` command
line program can be switched with the "G_plugin" configuration option.

Used variables:

.. code-block:: none

    r ... Distance to source in m
    t ... Time after source in s
    c ... Velocity in m/s
    l ... (transport) mean free path in m
    la ... absorption length in m
    g0 = 1/l ... (transport) scattering coefficient


.. note::
    The formula for the 3D Green's function is valid for the
    scattering coefficient (g0) under assumption of isotropic scattering.
    However, g0 is used by `qopen.core` module as transport scattering
    coefficient (g*) under the assumption of non-isotropic scattering.
    ``g*=g0`` is a reasonable assumption under these conditions.

"""

import argparse
import numpy as np


def rt1d_direct(t, c, g0):
    l = 1 / g0
    return 1 / 2 * np.exp(-c * t / 2 / l)  # * delta(c * t - r)


def rt1d_coda(r, t, c, g0):
    from scipy.special import iv
    l = 1 / g0
    arg0 = np.sqrt(c ** 2 * t ** 2 - r ** 2)
    arg1 = arg0 / 2 / l
    t1 = 1 / 4 / l * np.exp(-c * t / 2 / l)
    t2 = iv(0, arg1) + c * t * iv(1, arg1) / arg0
    return t1 * t2  # * Heaviside(c * t - r)


def rt2d_direct(t, c, g0):
    l = 1 / g0
    return np.exp(-c * t / l) / (2 * np.pi * c * t)  # * delta(c * t - r)


def rt2d_coda(r, t, c, g0):
    l = 1 / g0
    t1 = 1 / (2 * np.pi * l * c * t)
    t2 = (1 - r ** 2 / c ** 2 / t ** 2) ** (-1 / 2)
    t3 = np.exp((np.sqrt(c ** 2 * t ** 2 - r ** 2) - c * t) / l)
    return t1 * t2 * t3  # * Heaviside(c * t - r)


def rt3d_direct(t, c, g0, var='t'):
    t1 = np.exp(-c * t * g0)
    t2 = (4 * np.pi * c ** 2 * t ** 2)
    return t1 / t2  # * delta(c * t - r)


def _F(x):
    return np.sqrt(1 + 2.026 / x)


def rt3d_coda_reduced(r, t):
    # Coda term for r<t in reduced variables r'=rg0, t'=tg0c  (3d)
    a = 1 - r ** 2 / t ** 2
    t1 = a ** 0.125 / (4 * np.pi * t / 3) ** 1.5
    t2 = np.exp(t * (a ** 0.75 - 1)) * _F(t * a ** 0.75)
    return t1 * t2  # * Heaviside(t - r)


def rt3d_coda(r, t, c, g0):
    # Heaviside(c * t - r) *
    return rt3d_coda_reduced(r * g0, t * c * g0) * g0 ** 3


def diff(r, t, c, g0, dim=3):
    D = c / (dim * g0)
    t1 = (4 * np.pi * D * t) ** (-dim / 2)
    t2 = np.exp(-r ** 2 / (4 * D * t))
    return t1 * t2  # * Heaviside(c * t - r)


def diffapprox(r, t, c, g0, dim=3):
    D = c / (dim * g0)
    t1 = (4 * np.pi * D * t) ** (-dim / 2)
    return t1  # * Heaviside(c * t - r)


def G(r, t, c, g0, type='rt3d', include_direct=True):
    """Full Green's function with direct wave term (optional)"""
    if type == 'rt3d':
        Gcoda = rt3d_coda
        Gdirect = rt3d_direct
    elif type == 'rt2d':
        Gcoda = rt2d_coda
        Gdirect = rt2d_direct
    elif type == 'rt1d':
        Gcoda = rt1d_coda
        Gdirect = rt1d_direct
    elif type in ('diff1d', 'diff2d', 'diff3d'):
        dim = int(type[-2])
        Gcoda = lambda r, t, c, g0: diff(r, t, c, g0, dim=dim)
        Gdirect = lambda t, c, g0: np.zeros_like(t)
    elif type in ('diffapprox1d', 'diffapprox2d', 'diffapprox3d'):
        dim = int(type[-2])
        Gcoda = lambda r, t, c, g0: diffapprox(r, t, c, g0, dim=dim)
        Gdirect = lambda t, c, g0: np.zeros_like(t)
    else:
        NotImplementedError
    t_isarray = isinstance(t, np.ndarray)
    r_isarray = isinstance(r, np.ndarray)
    if not t_isarray and not r_isarray:
        if t - r / c < 0:
            G_ = 0
        else:
            G_ = Gcoda(r, t, c, g0)
    elif t_isarray and r_isarray:
        if len(t) != len(r):
            msg = ('If t and r are numpy arrays,'
                   'they need to have the same length')
            raise ValueError(msg)
        if include_direct:
            msg = 'If t and r are numpy arrays, include_direct not supported'
            raise NotImplementedError(msg)
        G_ = np.zeros(np.shape(t))
        ind = c * t - r >= 0
        G_[ind] = Gcoda(r[ind], t[ind], c, g0)
    elif t_isarray:
        G_ = np.zeros(len(t))
        eps = float(t[1] - t[0])
        i = np.count_nonzero(c * t - r < 0)
        G_[i+1:] = Gcoda(r, t[i+1:], c, g0)
        if include_direct and 0 < i < len(G_):
            # factor 1 / c due to conversion of Dirac delta from
            # delta(r - c * t) to delta(t - r / c)
            G_[i] = Gdirect(r / c, c, g0) / eps / c
    elif r_isarray:
        G_ = np.zeros(len(r))
        eps = float(r[1] - r[0])
        i = -np.count_nonzero(c * t - r < 0)
        if i == 0:
            i = len(r)
        G_[:i] = Gcoda(r[:i], t, c, g0)
        if include_direct and i != len(G_):
            G_[i] = Gdirect(t, c, g0) / eps
    return G_


def G_rt3d(r, t, c, g0):
    """
    Full Green's function for 3d radiative transfer (approximation).

    See Paaschens (1997), equation 36.
    """
    return G(r, t, c, g0, type='rt3d')


def G_rt2d(r, t, c, g0):
    """
    Full Green's function for 2d radiative transfer.

    See Paaschens (1997), equation 26.
    """
    return G(r, t, c, g0, type='rt2d')


def G_rt1d(r, t, c, g0):
    """
    Full Green's function for 1d radiative transfer

    See Paaschens (1997), equation 2, originally from Hemmer (1961).
    """
    return G(r, t, c, g0, type='rt1d')


def G_diff3d(r, t, c, g0):
    """
    Green's function for 3d diffusion

    See Paaschens (1997), equation 1.
    """
    return G(r, t, c, g0, type='diff3d')


def G_diffapprox3d(r, t, c, g0):
    """
    Green's function for 3d diffusion, discarding second term.

    Often used to determine Qcoda

    See Paaschens (1997), equation 1.
    """
    return G(r, t, c, g0, type='diffapprox3d')


def plot_t(c, g0, r, t=None, N=1000, log=False, include_direct=False, la=None,
           type='rt3d', scale=False):
    """Plot Green's function as a function of time"""
    import matplotlib.pyplot as plt
    if type == 'all':
        kw = dict(t=t, N=N, log=log, include_direct=include_direct, la=la,
                  scale=True)
        plot_t(c, g0, r, type='rt3d', **kw)
        plot_t(c, g0, r, type='rt2d', **kw)
        plot_t(c, g0, r, type='rt1d', **kw)
        plot_t(c, g0, r, type='diff3d', **kw)
        plot_t(c, g0, r, type='diffapprox3d', **kw)
        plt.legend()
        return
    if t is None:
        t = 10 * r / c
    ts = r / c + np.logspace(-3, np.log10(t - r / c), N)
    G_ = G(r, ts, c, g0, include_direct=include_direct, type=type)
    if la is not None:
        G_ = G_ * np.exp(-c * t / la)
    if scale:
        G_ = G_ / np.max(G_)
    ax = plt.gca()
    if log:
        ax.semilogy(ts, G_, label=type)
    else:
        ax.plot(ts, G_, label=type)
    ax.set_xlim((0, t))
    ax.set_ylabel('G')
    ax.set_xlabel('t (s)')


def plot_r(c, g0, t, r=None, N=1000, log=False, include_direct=False, la=None,
           type='rt3d', scale=False):
    """Plot Green's function as a function of distance"""
    import matplotlib.pyplot as plt
    if type == 'all':
        kw = dict(r=r, N=N, log=log, include_direct=include_direct, la=la,
                  scale=True)
        plot_r(c, g0, t, type='rt3d', **kw)
        plot_r(c, g0, t, type='rt2d', **kw)
        plot_r(c, g0, t, type='rt1d', **kw)
        plot_r(c, g0, t, type='diff3d', **kw)
        plot_r(c, g0, t, type='diffapprox3d', **kw)
        plt.legend()
        return
    if r is None:
        r = 1.1 * c * t
    rs = np.linspace(0, r, N)
    G_ = G(rs, t, c, g0, include_direct=include_direct, type=type)
    if la is not None:
        G_ = G_ * np.exp(-c * t / la)
    if scale:
        G_ = G_ / np.max(G_)
    ax = plt.gca()
    if log:
        ax.semilogy(rs, G_, label=type)
    else:
        ax.plot(rs, G_, label=type)
    ax.set_xlim((0, r))
    ax.set_ylabel('G')
    ax.set_xlabel('r (m)')


def plot_rt(c, g0, t=None, r=None, N=1000, log=False, include_direct=False, la=None,
           type='rt3d'):
    """Plot Green's function as a function of distance"""
    import matplotlib.pyplot as plt
    if t is None and r is None:
        r = 10 / g0
    if t is None:
        t = 10 * r / c
    if r is None:
        r = 1.1 * c * t
    ts = np.linspace(0, t, N)
    rs = np.linspace(0, r, N)
    mesh = np.meshgrid(rs, ts)
    # ingnore include direct
    if include_direct:
        from warnings import warn
        warn('ignore direct wave for rt plot')
    G_ = G(*mesh, c, g0, include_direct=False, type=type)
    if la is not None:
        G_ = G_ * np.exp(-c * mesh[1] / la)
    G_[np.isnan(G_)] = 0
    ax = plt.gca()
    if log:
        G_ = np.log(G_)
    im = ax.pcolormesh(*mesh, G_, label=type, cmap='jet')
    ax.contour(*mesh, G_, colors='k', levels=50, linewidths=1)
    ax.set_ylabel('t (s)')
    ax.set_xlabel('r (m)')
    plt.colorbar(im)


def main(args=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    choices = ('calc', 'calc-direct', 'plot-t', 'plot-r', 'plot-rt')
    p.add_argument('command', help='command', choices=choices)
    p.add_argument('c', help='velocity (m/s)', type=float)
    p.add_argument('l', help='transport mean free path (m)', type=float)
    p.add_argument('-r', help='distance from source (m)', type=float)
    p.add_argument('-t', help='time after source (s)', type=float)
    msg = 'absorption length (m)'
    p.add_argument('-a', '--absorption', help=msg, type=float)
    p.add_argument('--log', help='log plot', action='store_true')
    msg = 'do not include direct wave term in plots'
    p.add_argument('--no-direct', help=msg, action='store_true')
    msg = "type of Green's function to use"
    p.add_argument('--type', help=msg, default='rt3d',
                   choices=('rt3d', 'rt2d', 'rt1d',
                            'diff3d', 'diff2d', 'diff1d',
                            'diffapprox3d', 'diffapprox2d', 'diffapprox1d',
                            'all'))
    args = p.parse_args(args)
    r, t, c, l, la, type = (args.r, args.t, args.c, args.l, args.absorption,
                            args.type)
    com = args.command
    if 'calc' in com:
        if com == 'calc':
            G_ = G(r, t, c, 1/l, type=type)
        else:
            Gdirect = (rt1d_direct if type == 'rt1d' else
                       rt2d_direct if type == 'rt2d' else
                       rt3d_direct if type == 'rt3d' else None)
            if Gdirect is None:
                raise ValueError('No direct term for this Greens function.')
            G_ = Gdirect(t, c, 1/l)
        if la is not None:
            G_ = G_ * np.exp(-c * t / la)
        print(G_)
    else:
        import matplotlib.pyplot as plt
        kw = dict(log=args.log, include_direct=not args.no_direct,
                  la=la, type=type)
        if com == 'plot-t':
            plot_t(c, 1/l, r, t=t, **kw)
        elif com == 'plot-r':
            plot_r(c, 1/l, t, r=r, **kw)
        elif com == 'plot-rt':
            plot_rt(c, 1/l, t=t, r=r, **kw)
        plt.show()


if __name__ == '__main__':
    main()
