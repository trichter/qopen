# Copyright 2015-2017 Tom Eulenfeld, MIT license
"""
Align site responses of different runs of Qopen

Consider two runs of Qopen with stations A, B in run 1 and stations
B, C in run 2. Qopen assumes a mean site amplification for each run of 1
with the default configuration.
Use `align_site_responses` or the command line option ``qopen --align-sites``
to correct source power and site amplification factors afterwards.
In the above case site responses and source powers will be adjusted such that
the site response of station B is the same for both runs.
"""

from collections import defaultdict, OrderedDict
import logging
import warnings

import numpy as np
from obspy.geodetics import gps2dist_azimuth
import scipy

from qopen.source import calculate_source_properties
from qopen.util import gmean


log = logging.getLogger('qopen.site')
log.addHandler(logging.NullHandler())


def _collect_station_coordinates(inventory):
    coords = {}
    for net in inventory.networks:
        for sta in net.stations:
            cha = sta.channels[0]
            lat = cha.latitude or sta.latitude
            lon = cha.longitude or sta.longitude
            key = '%s.%s' % (net.code, sta.code)
            coords[key] = (lat, lon)
    return coords


def _collectR(results, freqi=0, only=None):
    from qopen.core import collect_results
    col = collect_results(results, freqi=freqi, only=['R'])
    R = col['R']
    if only is not None:
        R = {sta: R[sta] for sta in only}
#    for evid, evres in results['events'].items():
#        for sta, Rval in evres['R'].items():
#            if use_only and sta not in use_only:
#                continue
#            R[sta].append(Rval[freqi])
    return R


def _get_number_of_freqs(results):
    Nf = None
    for evid, eres in results['events'].items():
        Nf2 = len(eres['W'])
        if Nf is not None:
            assert Nf == Nf2
        Nf = Nf2
    return Nf


def _rescale_results(results, factors, only=None):
    log.debug('scale events and site responses')
    Nf = _get_number_of_freqs(results)
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
                if only and sta not in only[i]:
                    Rsta[i] = None
            if all(Rsta[i] is None or np.isnan(Rsta[i])
                   for Rsta in R.values()):
                W[i] = None


def _Rmean(R):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        Rm = [np.nanmean(np.log(np.array(x, dtype=np.float)))
              for x in R.values()]
    return np.nanmean(Rm)


def _Rstd(R):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        Rstd = [np.nanstd(np.log(np.array(x, dtype=np.float)))
                for x in R.values()]
    return np.nanmean(Rstd)


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


def _find_unconnected_areas(results, freqi, ignore_stations=None):
    areas = []
    for evid in results['events']:
        R = results['events'][evid]['R']
        area = {sta for sta, Rsta in R.items()
                if Rsta[freqi] is not None and not np.isnan(Rsta[freqi]) and
                (ignore_stations is None or sta not in ignore_stations)}
        if len(area) > 0:
            areas.append(area)
    areas = _merge_sets(areas)
    areas = {list(a)[0]: a for a in areas}
    log.info('found %d unconnected areas', len(areas))
    for name in areas:
        stations = areas[name]
        log.debug('area "%s" with %d stations', name, len(stations))
    return areas


def _join_unconnected_areas(areas, max_distance, inventory):
    # At the moment, areas are joined by setting the site response of
    # the station pair with smallest distance to 1
    # Often this works, but sometimes it produces undesired results
    coordinates = _collect_station_coordinates(inventory)
    station_by_coordinate = {c: sta for sta, c in coordinates.items()}
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
    # join unconnected regions if distance is smaller than max_distance
    near_stations = {}
    while len(distance) > 0:
        nearest_pair = min(distance, key=distance.get)
        dist = distance[nearest_pair][0]
        if dist > max_distance:
            break
        s1, s2 = distance[nearest_pair][1]
        near_stations[s1] = s2
        near_stations[s2] = s1
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
    return areas, near_stations


def align_site_responses(results, station=None, response=1., use_sparse=True,
                         seismic_moment_method=None,
                         seismic_moment_options=None,
                         ignore_stations=None):
    """
    Align station site responses and correct source parameters (experimental)

    Determine best factor for each event so that site response is the same
    for each station and different events.

    :param results: original result dictionary. For the other options see
        the help for the corresponding command line options or configuration
        parameters.
    :return: corrected result dictionary
    """
    # Ignore not existing event results, sort dict by event id
    results['events'] = OrderedDict(sorted([
        (evid, eres) for (evid, eres) in results['events'].items()
        if eres is not None], key=lambda x: x[0]))
    join_unconnected = None
    if join_unconnected:
        inventory = None
        msg = 'This feature needs more work and tests'
        raise NotImplementedError(msg)
    Ne = len(results['events'])
    if Ne == 1:
        use_sparse = False
    # Determine number of freqs
    Nf = _get_number_of_freqs(results)
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
    std_before = []
    # one for each freq
    largest_areas = []
    for i in range(Nf):
        log.debug('align sites for freq no. %d', i)
        # find unconnected areas
        areas = _find_unconnected_areas(results, i,
                                        ignore_stations=ignore_stations)
        if join_unconnected:
            areas, near_stations = _join_unconnected_areas(
                areas, join_unconnected, inventory)
        largest_area = max(areas, key=lambda k: len(areas[k]))
        msg = 'use largest area %s with %d stations'
        log.info(msg, largest_area, len(areas[largest_area]))
        largest_area = areas[largest_area]
        largest_areas.append(largest_area)
        R = _collectR(results, freqi=i, only=largest_area)
        std_before.append(_Rstd(R))
        row = [0]
        b = []
        if use_sparse:
            Arepr = [[], [[], []]]
        else:
            Arepr = []
        norm_row_A = defaultdict(float)
        norm_row_b = 0
        first = {}
        last = {}
        stations_used_norm = set()
        # add pairs of site responses for one station and different events
        for k, item in enumerate(results['events'].items()):
            evid, eres = item
            for sta, Rsta in eres['R'].items():
                Rsta = Rsta[i]
                if sta not in largest_area:
                    continue
                if Rsta is None or np.isnan(Rsta):
                    continue
                if station is None or sta == station or sta in station:
                    # collect information for normalization
                    stations_used_norm.add(sta)
                    fac = 1. / Nstations[i][sta]
                    norm_row_A[k] += fac
                    norm_row_b -= np.log(Rsta) * fac
                if sta in last:
                    # add pairs of site responses for one station
                    # and two different events
                    kl, Rstal = last[sta]
                    b_val = np.log(Rstal) - np.log(Rsta)
                    construct_ols(((k, 1), (kl, -1)), b_val)
                    last[sta] = k, Rsta
                elif (join_unconnected and sta in near_stations.keys() and
                        near_stations[sta] in last):
                    # add pairs of site responses for two nearby stations
                    # (in two previously unconnected areas)
                    # and two different events
                    kl, Rstal = last[near_stations[sta]]
                    b_val = np.log(Rstal) - np.log(Rsta)
                    construct_ols(((k, 1), (kl, -1)), b_val)
                    last[sta] = k, Rsta
                else:
                    last[sta] = first[sta] = (k, Rsta)
        # pin mean site response or site response of specific station(s)
        norm_row_b = norm_row_b / len(stations_used_norm) + np.log(response)
        for k in norm_row_A:
            norm_row_A[k] /= len(stations_used_norm)
        construct_ols(norm_row_A.items(), norm_row_b)
        msg = 'constructed %scoefficient matrix with shape (%d, %d)'
        log.debug(msg, 'sparse ' * use_sparse, row[0], Ne)
        # solve least squares system
        b = np.array(b)
        if use_sparse:
            A = scipy.sparse.csr_matrix(tuple(Arepr), shape=(row[0], Ne))
            res = scipy.sparse.linalg.lsmr(A, b, atol=1e-9)
        else:
            A = np.array(Arepr)
            res = scipy.linalg.lstsq(A, b, overwrite_a=True, overwrite_b=True)
        factors[:, i] = np.exp(res[0])

    # Scale W and R
    _rescale_results(results, factors, only=largest_areas)
    # Calculate sds, M0 and m again
    calculate_source_properties(
        results, seismic_moment_method=seismic_moment_method,
        seismic_moment_options=seismic_moment_options)
    # Calculate mean Rs again, use robust mean here
    results.setdefault('R', {})
    for st, Rst in _collectR(results, freqi=None).items():
        if not np.all(np.isnan(Rst)):
            results['R'][st] = gmean(Rst, axis=0, robust=True).tolist()
    std_after = []
    for i in range(Nf):
        R = _collectR(results, freqi=i)
        std_after.append(_Rstd(R))
    msg = ('aligned sites for all frequencies, reduction of mean stdev: ' +
           ', '.join(Nf * ['%.2g']) + ' -> ' + ', '.join(Nf * ['%.2g']))
    log.info(msg, *(std_before + std_after))
    return results
