#!/usr/bin/env python3.7

from _pybgpstream import BGPStream, BGPRecord, BGPElem
from collections import defaultdict


# Create a new bgpstream instance and a reusable bgprecord instance
stream = BGPStream()
rec = BGPRecord()

# Consider RRC06 only
stream.add_filter('collector','rrc06')

# Consider RIBs dumps only
stream.add_filter('record-type','ribs')

# Consider messages from the 25152 peer only
stream.add_filter('peer-asn','25152')

# Consider entries associated with 185.84.166.0/23 and more specifics
stream.add_filter('prefix','185.84.166.0/23')

# Consider entries having a community attribute with value 3400
stream.add_filter('community','*:3400')

# Consider this time interval:
# Sat, 01 Aug 2015 7:50:00 GMT -  08:10:00 GMT
stream.add_interval_filter(1438415400,1438416600)

# Start the stream
stream.start()

# <community, prefix > dictionary
community_prefix = defaultdict(set)

# Get next record
while(stream.get_next_record(rec)):
    elem = rec.get_next_elem()
    while(elem):        
        # Get the prefix
        pfx = elem.fields['prefix']
        # Get the associated communities
        communities = elem.fields['communities']
        # for each community save the set of prefixes
        # that are affected
        for c in communities:
            ct = str(c["asn"]) + ":" + str(c["value"])
            community_prefix[ct].add(pfx)
        elem = rec.get_next_elem()

# Print the list of MOAS prefix and their origin ASns
for ct in community_prefix:
    print("Community:", ct, "==>", ",".join(community_prefix[ct]))

