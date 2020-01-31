#!/usr/bin/env python3.7
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[2]:


import pybgpstream as pbs
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, Set
import pickle


# AMS-IX
collectors = ["route-views.amsix"]


# In[3]:


# Find a time where rib dumps occurred in this range
startSearchTime = "2020-01-07 00:00:00 UTC"
endSearchTime = "2020-01-07 23:59:59 UTC"

def getFirstRibDumpTime():
    fullDayRibStream = pbs.BGPStream(
        from_time=startSearchTime,
        until_time=endSearchTime,
        collectors=collectors,
        record_type="ribs"
    )
    for rec in fullDayRibStream.records():
        for elem in rec:
            if 'next-hop' in elem.fields:
                return rec.time
            
startTime = int(getFirstRibDumpTime())
endTimeShort = startTime + 60
endTimeLong = startTime + 60 * 15
print("Found RIB dumps at time %d" % startTime)


# In[4]:


ribStream = pbs.BGPStream(
        collectors=collectors,
        record_type="ribs"
    )
ribStream.add_interval_filter(startTime, endTimeShort)

updateStream = pbs.BGPStream(
        collectors=collectors,
        record_type="updates"
    )
updateStream.add_interval_filter(startTime, endTimeLong)


# In[5]:


def extractMatrix(stream):
    prefixSets = defaultdict(set)
    for rec in stream.records():
        for elem in rec:
            # if the record is a route, it will have 'next-hop' in the fields
            if 'next-hop' in elem.fields:

                prefix = elem.fields['prefix']

                announcer = elem.fields['as-path'].split(" ")[0]

                prefixSets[prefix].add(announcer)

    return prefixSets
        
matrix = extractMatrix(ribStream)


# In[6]:


print("%d rows in matrix" % len(matrix))
print("First 100 rows:")
for prefix, ases in list(matrix.items())[:10]:
    print(prefix, str(list(ases)))


# In[7]:


def extractUpdates(stream):
    updates = []
    for rec in stream.records():
        for elem in rec:
            if len(elem.fields) > 0:
                if elem.type in ['A', 'W']:
                    announcer = elem.fields.get('as-path', None)
                    if announcer is None:
                        announcer = elem.peer_asn
                    else:
                        announcer = announcer.split()[0]
                    updateType = '+' if elem.type == 'A' else '-'
                    updateTuple = (rec.time, announcer, updateType, elem.fields['prefix'])
                    updates.append(updateTuple)
                #print(elem.type)
                #print(elem.fields)
    return updates
updates = extractUpdates(updateStream)


# In[8]:


print("%d updates scraped." % len(updates))
print("First 100 updates:")
for update in updates[:10]:
    print(update)


# In[9]:


pickle_filename = "bgpstream_matrix_and_updates.pickle"
print("Saving matrix and updates to pickle", pickle_filename)

obj = {'matrix': matrix, 'updates':updates}
with open(pickle_filename, 'wb') as fp:
    pickle.dump(obj, fp)
print("Done.")


# In[ ]:




