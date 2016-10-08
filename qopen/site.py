# -*- coding: utf-8 -*-
# Author: Tom Eulenfeld
"""
Align site responses of different runs of Qopen.

Consider two runs of Qopen with stations A, B in run 1 and stations
B, C in run 2. Qopen assumes a mean site amplification for each run of 1
with the default configuration.
Use `align_site_responses` to correct source power and site amplification
factors afterwards.
In the above case site responses and source powers will be adjusted such that
the site response of station B is the same for both runs.
"""

from collections import defaultdict
import logging
import numpy as np
from obspy.geodetics import gps2dist_azimuth
import scipy

from qopen.source import calculate_source_properties


log = logging.getLogger('qopen.site')
log.addHandler(logging.NullHandler())


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

