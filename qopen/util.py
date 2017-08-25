# Copyright 2015-2017 Tom Eulenfeld, MIT license
"""Some utility functions"""

import functools

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM


def cache(f):
    cache = f.cache = {}

    @functools.wraps(f)
    def _f(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = f(*args, **kwargs)
        return cache[key]
    return _f


LOGGING_DEFAULT_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'capture_warnings': True,
    'formatters': {
        'file': {
            'format': ('%(asctime)s %(module)-6s%(process)-6d%(levelname)-8s'
                       '%(message)s')
        },
        'console': {
            'format': '%(levelname)-8s%(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'console',
            'level': None,
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'file',
            'level': None,
            'filename': None,
        },
    },
    'loggers': {
        'qopen': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'py.warnings': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        }

    }
}


def weighted_stat(data, axis=None, weights=None, unbiased=True):
    """Weighted mean and biased/unbiased standard deviation"""
    mean = np.ma.average(data, axis=axis, weights=weights)
    if np.ma.count(data) < 2:
        std = np.nan
    elif weights is None:
        std = np.ma.std(data, axis=axis, ddof=unbiased)
    else:
        var = np.ma.average((data - mean) ** 2, axis=axis, weights=weights)
        if unbiased:
            V1 = np.ma.sum(weights * ~data.mask, axis=axis)
            V2 = np.ma.sum(weights ** 2 * ~data.mask, axis=axis)
            std = (var / (1 - V2 / V1 ** 2)) ** 0.5
        else:
            std = var ** 0.5
    return mean, std


def robust_stat(data, axis=None, fall_back=5):
    """Robust mean and mean absolute deviation

    See also:
    statsmodels.sf.net/stable/rlm.html
    """
    data = np.ma.masked_invalid(data)
    assert len(data.shape) < 3
    if axis is not None:
        if axis == 0:
            data = data.T
        mean = np.empty(data.shape[0])
        err = np.empty(data.shape[0])
        for i, d in enumerate(data):
            mean[i], err[i] = robust_stat(d)
        return mean, err
    assert len(data.shape) < 2
    data = data[~data.mask]
    if len(data) < fall_back:
        return weighted_stat(data)
    res = RLM(data, np.ones(len(data))).fit()
    return res.params[0], res.scale


def gstat(data, axis=None, weights=None, unbiased=True, robust=False):
    """Weighted or robust geometric mean and error (log scale)"""
    # warnings may pop up due to a bug in numpy
    # https://github.com/numpy/numpy/issues/4959
    data = np.log(np.ma.masked_invalid(data))
    if robust and weights is not None:
        raise NotImplementedError
    elif robust:
        mean, err = robust_stat(data, axis=axis)
    else:
        mean, err = weighted_stat(data, axis=axis, weights=weights,
                                  unbiased=unbiased)
    return mean, err


def gmean(data, **kwargs):
    """Weighted or robust geometric mean"""
    return np.exp(gstat(data, **kwargs)[0])


def gerr(data, **kwargs):
    """Weighted or robust geometric mean and lower/upper error"""
    mean, err = gstat(data, **kwargs)
    err1 = np.exp(mean) - np.exp(mean - err)
    err2 = np.exp(mean + err) - np.exp(mean)
    return np.exp(mean), err1, err2


def smooth_func(f, t, window_len=None, window='flat'):
    """Smooth a function f at time samples t"""
    if window_len is None:
        f_ = f(t)
    else:
        dt = t[1] - t[0]
        if np.sum(np.abs(np.diff(t)-dt)) > 1e-5:
            msg = 'samples have to be evenly spaced'
            raise ValueError(msg)
        samples = int(round(window_len / dt))
        N1 = (samples - 1) // 2
        N2 = samples // 2
        t_ = np.hstack((t[0] - N1 * dt + np.arange(N1) * dt, t,
                        t[-1] + dt + np.arange(N2) * dt))
        f_ = f(t_)
        f_ = smooth(f_, samples, method=None, window=window)
    return f_


def smooth(x, window_len=None, window='flat', method='zeros'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.

    :param x: the input signal
    :param window_len: the dimension of the smoothing window; should be an
        odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'
        flat window will produce a moving average smoothing.
    :param method: handling of border effects\n
        'zeros': zero padding on both ends (len(smooth(x)) = len(x))\n
        'reflect': pad reflected signal on both ends (same)\n
        'clip': pad signal on both ends with the last valid value (same)\n
        None: no handling of border effects
        (len(smooth(x)) = len(x) - len(window_len) + 1)

    See also:
    www.scipy.org/Cookbook/SignalSmooth
    """
    if window_len is None:
        return x
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming',"
                         "'bartlett', 'blackman'")
    if method == 'zeros':
        s = np.r_[np.zeros((window_len - 1) // 2), x,
                  np.zeros(window_len // 2)]
    elif method == 'reflect':
        s = np.r_[x[(window_len - 1) // 2:0:-1], x,
                  x[-1:-(window_len + 1) // 2:-1]]
    elif method == 'clip':
        s = np.r_[x[0] * np.ones((window_len - 1) // 2), x,
                  x[-1] * np.ones(window_len // 2)]
    else:
        s = x
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    return np.convolve(w / w.sum(), s, mode='valid')


def linear_fit(y, x, m=None, method='robust'):
    """Linear fit between x and y

    :param y,x: data
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
