import os, sys
import json
import pickle
try:
    from .MRSets import MRCode
except:
    from MRSets import MRCode

REPO_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(REPO_PATH, "data/")


def compress(filename):
    filepath = os.path.join(DATA_PATH, filename)

    with open(filepath, 'rb') as fp:
        supersets = pickle.load(fp)

    print("Total number of groups: ", len(set(supersets)))
    originalElements = set.union(*[set(superset) for superset in supersets])
    originalSets = [frozenset(superset) for superset in supersets]
    print("Total number of elements: ", len(originalElements))

    # orderedSets = [supersets[i*2 + 1] for i in range(len(supersets)/2)]
    # maxElement = max(max(superset) for superset in orderedSets)
    # ordering = {i:i for i in range(maxElement+1)}
    # rcode = RCode(orderedSets)
    # rcode.optimizeWidth()

    # hierarchy = True makes the MRCode uses biclutering hierarchy algorithm
    mrcode = MRCode(originalSets, hierarchy = True)
    # parameters is curretly a tuple of (Threshold of bicluster size, Goal of tag width)

    threshold = 5
    while threshold < 6:
        mrcode.optimize(parameters = (threshold, None))
        totalMemory = [len(rule) for rules in mrcode.matchStrings().values() for rule in rules]
        print("Chosen threshold: ", threshold)
        print("Total number of rules: ", len(totalMemory))
        print("Length of rule:", totalMemory[0])
        print("Total memory:", sum(totalMemory))
        threshold += 1

    # print("Tag width: ", len(mrcode.tagString(frozenset([]))))
    # print("Tag width: ", len(mrcode.matchStrings(frozenset(['AS8283']))['AS8283'][0]))
    # print("Tag width: ", mrcode.matchStrings(frozenset(['AS8283']))['AS8283'][0])


if __name__ == '__main__':
    compress("matrixfull.pickle")
