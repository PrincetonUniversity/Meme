#!/usr/bin/env python3.7

from _pybgpstream import BGPStream, BGPRecord, BGPElem
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, Set
import pickle

timeString = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
filename = "prefix2as_%s.pkl" % timeString 

print("Beginning record scrape. Results will be saved to", filename)
# Create a new bgpstream instance and a reusable bgprecord instance
stream = BGPStream()
rec = BGPRecord()

# Consider RIPE RRC 11 only
#stream.add_filter('collector','rrc11')

# Consider RIB dumps only
stream.add_filter('record-type', 'ribs')

# Consider this time interval:
intervalSeconds = int(2 * 60 * 60)
endTime = int(time.time())
startTime = endTime - intervalSeconds
stream.add_interval_filter(startTime,endTime)

# Start the stream
stream.start()

prefixes = set([])

# any AS that is along a path to a prefix
prefixRouters : Dict['prefix', Set['ASes']] = defaultdict(set)
ixpParticipants = set()
ases = set()

printPeriod = 10
lastPrintTime = 0
numRecords = 0
# Get next record
while(stream.get_next_record(rec)):
    # Print the record information only if it is not a valid record
    numRecords += 1
    # for small testing
    if numRecords > 400000:
        break
    currTime = time.time()
    if currTime - lastPrintTime > printPeriod:
        lastPrintTime = currTime
        print("At record %d. At collector %s" % (numRecords, rec.collector))

    if rec.status != "valid":
        print(rec.project, rec.collector, rec.type, rec.time, rec.status)
    else:
        elem = rec.get_next_elem()
        while(elem):
            # Print record and elem information
            #print(rec.project, rec.collector, rec.type, rec.time, rec.status, end=' ')
            #print(elem.type, elem.peer_address, elem.peer_asn, elem.fields)
            if 'next-hop' in elem.fields:
                hop = elem.fields['next-hop']
                prefix = elem.fields['prefix']
                
                asPath = elem.fields['as-path'].split(" ")
                origin = asPath[-1]
                advertiser = asPath[0]

                ixpParticipants.add(advertiser)

                for asHop in asPath:
                    prefixRouters[prefix].add(asHop)
                    ases.add(asHop)

                prefixes.add(prefix)
            #print(rec.project, rec.collector, rec.type, rec.time, rec.status, end=' ')
            #print(elem.type, elem.peer_address, elem.peer_asn, elem.fields)

            elem = rec.get_next_elem()

print("Num distinct IXP participants:", len(ixpParticipants))
print("Num distinct prefixes:", len(prefixes))
print("Num distinct AS numbers in paths:", len(ases))

avgRouters = sum([len(routers) for routers in prefixRouters.values()]) / len(prefixRouters)
print("Average number of routers for prefixes are", avgRouters)

prefixAnnouncers = {prefix : [AS for AS in routers if AS in ixpParticipants] for prefix,routers in prefixRouters.items()}
avgAnnouncers = sum([len(announcers) for announcers in prefixAnnouncers.values()]) / len(prefixAnnouncers)
print("Average number of announcers for prefixes are", avgAnnouncers)

print("Writing to file", filename)
with open(filename, 'wb') as fp:
    pickle.dump((ixpParticipants, prefixRouters), fp)
print("Done.")

