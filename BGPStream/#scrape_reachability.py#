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

# Consider RIB dumps only
stream.add_filter('record-type', 'ribs')

# Only consider the past two hours
intervalSeconds = int(2 * 60 * 60)
endTime = int(time.time())
startTime = endTime - intervalSeconds
stream.add_interval_filter(startTime,endTime)

# Start the stream
stream.start()


# any AS that is along a path to a prefix should be able to route packets for that prefix,
# hence I call them "routers"
prefixRouters : Dict['prefix', Set['ASes']] = defaultdict(set)
# any AS that is the first hop of a path announced at an IXP is assumed to be a participant
ixpParticipants = set()
# anyone in any path anywhere ever
ases = set()

printPeriod = 10 # how frequently to print status messages, in seconds
lastPrintTime = 0
numRecords = 0
# Get next record
while(stream.get_next_record(rec)):
    # Print the record information only if it is not a valid record
    numRecords += 1
    ##### uncomment this for small testing
    if numRecords > 400000:
        break
    ########
    # Print periodically so we know it isnt stalled
    currTime = time.time()
    if currTime - lastPrintTime > printPeriod:
        lastPrintTime = currTime
        print("At record %d. At collector %s" % (numRecords, rec.collector))

    if rec.status != "valid":
        print(rec.project, rec.collector, rec.type, rec.time, rec.status)
    else:
        elem = rec.get_next_elem()
        while(elem):
            # if the record is a route, it will have 'next-hop' in the fields
            if 'next-hop' in elem.fields:
                hop = elem.fields['next-hop']
                prefix = elem.fields['prefix']
                
                asPath = elem.fields['as-path'].split(" ")
                origin = asPath[-1] # last hop in the path
                advertiser = asPath[0] # first hop in the path. presumed to be connected to the IXP

                ixpParticipants.add(advertiser)

                for asHop in asPath:
                    prefixRouters[prefix].add(asHop)
                    ases.add(asHop)

            # Print record and elem information
            #print(rec.project, rec.collector, rec.type, rec.time, rec.status, end=' ')
            #print(elem.type, elem.peer_address, elem.peer_asn, elem.fields)

            elem = rec.get_next_elem()

print("Num distinct IXP participants:", len(ixpParticipants))
print("Num distinct prefixes:", len(prefixRouters))
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

