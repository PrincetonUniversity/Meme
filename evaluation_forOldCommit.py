import os, sys
import json
try:
    from .RSets import RCode
except:
    from RSets import RCode

REPO_PYTHON_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_PATH = os.path.realpath(os.path.join(REPO_PYTHON_PATH, '..'))
DATA_PATH = os.path.join(REPO_PATH, "data/")


def compress(filename):

    filepath = os.path.join(DATA_PATH, filename)

    with open(filepath, 'r') as fp:
        supersets = json.load(fp)

    originalElements = set.union(*[set(superset.keys()) for superset in supersets.values()])
    originalSets = [frozenset(superset.keys()) for superset in supersets.values()]
   

    # orderedSets = [supersets[i*2 + 1] for i in range(len(supersets)/2)]
    # maxElement = max(max(superset) for superset in orderedSets)
    # ordering = {i:i for i in range(maxElement+1)}
    # rcode = RCode(orderedSets)
    # rcode.optimizeWidth()

    width = 15
    while width <= 40:
        rcode = RCode(originalSets,maxWidth=width)
        rcode.optimizeMemory()

        totalMemory = [len(rule) for rules in rcode.allMatchStrings().values() for rule in rules]
        print("Chosen width: ", width)
        print("Total number of rules: ", len(totalMemory))
        print("Length of rule:", totalMemory[0])
        print("Total memory:", sum(totalMemory))
        width += 1



if __name__ == '__main__':
    compress("prefix_2_participants.json")
