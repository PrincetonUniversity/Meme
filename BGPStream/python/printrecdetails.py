#!/usr/bin/env python3.7

from _pybgpstream import BGPStream, BGPRecord, BGPElem
import time

# Create a new bgpstream instance and a reusable bgprecord instance
stream = BGPStream()
rec = BGPRecord()

# Consider RIPE RRC 10 only
#stream.add_filter('collector','rrc11')


# Consider RIBs dump only
stream.add_filter('record-type','ribs')

# Consider this time interval:
intervalSeconds = int(1 * 60 * 60)
endTime = int(time.time())
startTime = endTime - intervalSeconds
stream.add_interval_filter(startTime,endTime)


# Start the stream
stream.start()

collectors = set([])
projects = set([])
numRecords = 0
while(stream.get_next_record(rec)):
    # Print the record information only if it is not a valid record
    #if rec.status != "valid":
    #print(rec.project, rec.collector, rec.type, rec.time, rec.status)
    numRecords += 1
    projects.add(rec.project)
    if rec.collector not in collectors:
        print("New collector:", rec.collector)
        print("At %d records" % numRecords)
        collectors.add(rec.collector)

    if numRecords % 1000000 == 0:
        print("At %d records. Current collector: %s" % (numRecords, rec.collector))

print(projects, '\n', collectors)



