# -*- coding: utf-8 -*-
# Copyright 2015-2017 Tom Eulenfeld, MIT license
"""
Fit source displacement spectrum with source model

(and some other functions dealing with the source)

If you want to fit source displacement spectra on the command line again
use the ``qopen --calc-source-params`` option.
"""

import logging
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
import scipy.optimize

from qopen.util import gstat


def sds(W, f, v, rho):
    """
    Calculate source displacement spectrum ωM from spectral source energy W

    according to Sato & Fehler (2012, p.188)

    :param W: spectral source energy (J/Hz)
    :param f,v,rho: frequency, mean velocity, mean density
    :return: source displacement spectrum (in Nm)"""
    return np.sqrt(W * 2.5 / np.pi * rho * v ** 5 / f ** 2)


def source_model(freq, M0, fc, n=2, gamma=1):
    """Model for source displacement spectrum (Abercrombie 1995)

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
    """Fit source displacement spectrum and calculate seismic moment

    :param freq,omM: frequencies, source displacement spectrum (same length)
    :param method: 'mean' - take mean of sds of frequencies below fc,
        'fit', 'robust_fit' - fit source model to obtain M0.
        If one or more of fc, n, gamma are None, M0 and these values are
        simultaneously determined.
        Robust version uses a robust linear model (which downweights outliers).
    :param fc,n,gamma: corner frequency and coefficients for source model
    :param fc_lim,gamma_lim: bounds for corner frequency and gamma
        (used for optimization if respective variable is set to None)
    :param fc0,gamma0: starting values of fc and gamma for optimization
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
            mean, err = gstat(M0, unbiased=False)
            return {'M0': np.exp(mean), 'fit_error': err}
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
            err = np.mean(res.resid ** 2)
            if opt:
                return err
            return {'M0': np.exp(res.params[0]), 'fit_error': err ** 0.5}

        def lstsqab(fc, a, opt=False):
            # Inversion for M0 and b
            model = _source_model_ab(freq, 1, fc, a, 1)
            y = np.log(omM)
            X = np.empty((len(y), 2))
            X[:, 0] = 1
            X[:, 1] = np.log(model)
            res = Model(y, X).fit()
            err = np.mean(res.resid ** 2)
            if opt:
                return err
            return {'M0': np.exp(res.params[0]), 'b': res.params[1],
                    'fit_error': err ** 0.5}

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
            log = logging.getLogger('qopen.source')
            log.debug(msg, unknowns, opt.message.lower())
        if 'a' in result:
            a = result.pop('a')
            b = result.pop('b')
            result['gamma'] = 1 / b
            result['n'] = a * b
        return result


def moment_magnitude(M0, inverse=False):
    """
    Moment magnitude Mw from seismic moment M0

    Based on Kanamori (1977), an alternative definition is based on
    Hanks and Kanamori (1979) with an offset of -6.03.
    Difference due to rounding errors.

    :param M0: seismic moment in Nm
    :param inverse: return the inverse relation ship M0(Mw)
    """
    if inverse:
        Mw = M0
        return 10 ** (1.5 * (Mw + 6.07))
    return 2 / 3 * np.log10(M0) - 6.07


def calculate_source_properties(results, rh0=None, v0=None,
                                seismic_moment_method=None,
                                seismic_moment_options=None):
    """Calculate source porperties for results"""
    conf = results.get('config', {})
    rho0 = rh0 or conf.get('rho0')
    v02 = v0 or conf.get('v0')
    smm = seismic_moment_method or conf.get('seismic_moment_method')
    smo = seismic_moment_options or conf.get('seismic_moment_options')
    freq = results.get('freq')
    if rho0:
        for r in dict(results['events']).values():  # dict from future.builtins
            v0 = r.get('v0') or v02
            r.pop('sds', None)
            r.pop('M0', None)
            r.pop('fc', None)
            r.pop('n', None)
            r.pop('gamma', None)
            r.pop('fit_error', None)
            if v0:
                insert_source_properties(freq, r, v0, rho0, smm, smo)
    return results


def insert_source_properties(freq, evresult, v0, rho0, seismic_moment_method,
                             seismic_moment_options, catmag=None):
    """Insert sds, Mw and possibly Mcat in evresult dictionary"""
    from qopen.core import sort_dict
    if evresult['W'] and rho0 and v0:
        evresult['sds'] = []
    for i, f in enumerate(freq):
        if evresult['W'][i] and rho0 and v0:
            evresult['sds'].append(sds(evresult['W'][i], f, v0, rho0))
        else:
            evresult['sds'].append(None)
    if seismic_moment_method:
        omM = evresult['sds']
        fitresult = fit_sds(freq, omM, method=seismic_moment_method,
                            **seismic_moment_options)
        if fitresult is not None:
            if np.isnan(fitresult.get('fit_error', 1)):
                fitresult['fit_error'] = None
            evresult.update(fitresult)
            if 'M0' in fitresult:
                evresult['Mw'] = moment_magnitude(fitresult['M0'])
        if catmag is not None:
            evresult['Mcat'] = catmag
    return sort_dict(evresult)
