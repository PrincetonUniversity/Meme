#!/usr/bin/env python3.7

import pickle, sys, time
from datetime import datetime

timeString = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


ixpParticipants, prefixRouters = None, None
filename = sys.argv[1]
print("Reading %s..." % filename)
with open(filename, 'rb') as fp:
    ixpParticipants, prefixRouters = pickle.load(fp)
print("Done. %d participants, %d prefixes" % (len(ixpParticipants), len(prefixRouters)))

prefixes = list(prefixRouters.keys())
ASes = set.union(*list(prefixRouters.values()))
print("Num distinct AS numbers in paths:", len(ASes))



avgRouters = sum([len(routers) for routers in prefixRouters.values()]) / len(prefixRouters)
print("Average number of routers for prefixes are", avgRouters)

prefixAnnouncers = {prefix : [AS for AS in routers if AS in ixpParticipants] for prefix,routers in prefixRouters.items()}
avgAnnouncers = sum([len(announcers) for announcers in prefixAnnouncers.values()]) / len(prefixAnnouncers)
print("Average number of announcers for prefixes are", avgAnnouncers)

matrix = list(prefixAnnouncers.keys())
matrixFilename = "matrix_%s.pkl" % timeString
print("Writing matrix to file", matrixFilename)
with open(matrixFilename, 'wb') as fp:
    pickle.dump(matrix, fp)
print("Done.")
