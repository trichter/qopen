# Author: Tom Eulenfeld

from obspy import read_events, read_inventory, Stream, UTCDateTime as UTC
from obspy.clients.arclink import Client as ArcClient
from obspy.clients.fdsn import Client as FSDNClient


evname = './example_events.xml'
invname = './example_inventory.xml'
wavname = './example_data.mseed'
wavformat = 'mseed'

t1, t2 = UTC('1992-01-01'), UTC('2005-01-01')
stations = 'BFO BUG CLZ FUR TNS'.split()
seed_id = 'GR.???..HH?'
net, sta, loc, cha = seed_id.split('.')
event_kwargs = {'starttime': t1, 'endtime': t2, 'minmagnitude': 4.4,
                'latitude': 50, 'longitude': 7, 'maxradius': 2.5}
inventory_kwargs = {'starttime': t1, 'endtime': t2, 'network': net,
                    'station': sta, 'location': loc, 'channel': cha,
                    'level': 'response'}
client_kwargs = {'host': 'eida.bgr.de', 'port': 18001, 'user': 'example@qopen'}


def get_events():
    print('Read event file')
    try:
        return read_events(evname)
    except:
        pass
    client = FSDNClient('NERIES')
    events = client.get_events(**event_kwargs)
    events.events.sort(key=lambda e: e.origins[0].time)
    events.write(evname, 'QUAKEML')
    return events


def get_inventory():
    print('Read inventory file')
    try:
        return read_inventory(invname, 'STATIONXML')
    except:
        pass
    print('Create inventory file...')
    client = FSDNClient('ORFEUS')
    inv = client.get_stations(**inventory_kwargs)
    for net in inv:
        for sta in net[:]:
            if sta.code not in stations:
                net.stations.remove(sta)
    inv.write(invname, 'STATIONXML')
    return inv


def get_waveforms():
    events = get_events()
    client = ArcClient(**client_kwargs)
    wforms = Stream()
    for i, event in enumerate(events):
        print('Fetch data for event no. %d' % (i + 1))
        t = event.preferred_origin().time
        for sta in stations:
            args = (net, sta, loc, cha, t - 10, t + 220)
            try:
                stream = client.getWaveform(*args)
            except:
                print('no data for %s' % (args,))
                continue
            sr = stream[0].stats.sampling_rate
            stream.decimate(int(sr) // 20, no_filter=True)
            for tr in stream:
                del tr.stats.mseed
            stream.merge()
            wforms.extend(stream)
    wforms.write(wavname, wavformat)
    return wforms


print(get_events())
print()
print(get_inventory())
print()
print(get_waveforms())
print()
print('Succesfully created example files')
