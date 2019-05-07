import math
try:
    from .analyze import bitsRequiredFixedID, bitsRequiredVariableID
except:
    from analyze import bitsRequiredFixedID, bitsRequiredVariableID

from itertools import combinations


class UnionFind:
    nodeOwners = {}

    def __init__(self, nodes):
        # initially, each node is it's own owner
        self.nodeOwners = {node:node for node in nodes}

    def find(self, node):
        while(node != self.nodeOwners[node]):
            node = self.nodeOwners[node]
        return node

    def union(self, node1, node2):
        owner1 = self.find(node1)
        owner2 = self.find(node2)

        # make node1's root own node2
        self.nodeOwners[owner2] = owner1


    def connectedComponents(self):
        CCs = {}
        for node in self.nodeOwners:
            root = self.find(node)
            if root not in CCs:
                CCs[root] = [node]
            else:
                CCs[root].append(node)

        return list(CCs.values())


def mergeIntersectingSets(supersets):
    toSetID   = lambda x : ("Set", x)
    fromSetID = lambda x : x[1]
    isSetID   = lambda x : (type(x) is tuple) and (x[0] == "Set")

    allItems = set()
    for s in supersets:
        allItems.update(s)

    UF = UnionFind([toSetID(i) for i in range(len(supersets))] + list(allItems))

    for i, s in enumerate(supersets):
        sid = toSetID(i)
        for item in s:
            UF.union(sid, item)

    CCs = UF.connectedComponents()

    overlappingSets = [[fromSetID(t) for t in cc if isSetID(t)] for cc in CCs]

    mergedSets = []
    for overlapGroup in overlappingSets:
        mergedSet = set()
        for setIndex in overlapGroup:
            mergedSet.update(supersets[setIndex])
        mergedSets.append(mergedSet)
    return mergedSets



def removeSubsets(supersets):
    """ Removes all subsets from a list of (super)sets.
    """
    final_answer = []
    # defensive copy
    supersets = [set(superset) for superset in supersets]
    supersets.sort(key=len, reverse=True)
    i = 0
    while i < len(supersets):
        final_answer.append(supersets[i])

        for j in reversed(range(i+1, len(supersets))):
            if supersets[j].issubset(supersets[i]):
                del supersets[j]

        i += 1

    return final_answer


def greedyToEndAndRevert(supersets, initialCost, mergeValueFunc, costUpdateFunc):
    """ initialCost is the objective function evaluated over the given supersets.
        mergeValueFunc is a function(set1,set2) that assigns a value to merging a pair of supersets.
            Lower value is assumed to be better.
        costUpdateFunc is a function(set1,set2,previousCost,supersets) that returns an updated objective function value.
    """
    originalSupersets = [superset.copy() for superset in supersets]

    stepsTaken = []
    bestStoppingPoint = 0

    minCost = initialCost
    currCost = initialCost

    def _merge(_i, _j):
        # add elements of the jth superset to the ith superset
        supersets[_i].update(supersets[_j])
        # move the jth superset to the end of the list for efficient deletion
        supersets[_j], supersets[-1] = supersets[-1], supersets[_j]
        supersets.pop()

    while len(supersets) > 1:
        bestMergeValue = None
        besti = -1
        bestj = -1
        for (i,set1), (j,set2) in combinations(enumerate(supersets),2):
            mergeValue = mergeValueFunc(set1, set2)
            if bestMergeValue == None or mergeValue < bestMergeValue:
                bestMergeValue = mergeValue
                besti = i
                bestj = j

        currCost = costUpdateFunc(supersets[besti], supersets[bestj], currCost, supersets)
        _merge(besti, bestj)
        stepsTaken.append((besti, bestj))
        if currCost < minCost:
            minCost = currCost
            bestStoppingPoint = len(stepsTaken)


    # reset and repeat the mergings until the point that had the lowest cost
    supersets = originalSupersets
    for (i, j) in stepsTaken[:bestStoppingPoint]:
        _merge(i, j)

    return supersets



def minimizeVariableWidthGreedy(supersets):
    """ Given a list of supersets, greedily minimize the number of bits required
        when the superset IDs are a prefix code.
    """
    kraftSum = sum(2**len(superset) for superset in supersets)
    kraftImpact = lambda set1,set2 : -(2**len(set1) + 2**len(set2)) + 2**len(set1.union(set2))

    costUpdate = lambda set1,set2,prevCost,_ : prevCost + kraftImpact(set1,set2)

    return greedyToEndAndRevert(supersets, kraftSum, kraftImpact, costUpdate)



def minimizeFixedWidthGreedy(supersets):
    """ Given a list of supersets, greedily minimize the number of bits required
        to represent any set as a superset ID and corresponding bitmask.
    """
    initialTagWidth = bitsRequiredFixedID(supersets)

    unionSize = lambda set1,set2 : len(set1.union(set2))

    costUpdate = lambda _1,_2,prevCost,supersets : min(prevCost, bitsRequiredFixedID(supersets))

    return greedyToEndAndRevert(supersets, initialTagWidth, unionSize, costUpdate)



def minimizeRulesGreedy(supersets, ruleCounts, maxBits):
    """ Given a list of supersets, the number of rules needed regarding
        each participant in an outbound policy, and an upper bound
        on the number of bits we are willing to use, greedily minimize
        the number of rules that will result from the superset grouping.
    """
    # defensive copy
    supersets = [set(superset) for superset in supersets]

    # build the set of all participants
    participants = set()
    [participants.update(superset) for superset in supersets]

    # if the number of participants is less than the max number of bits,
    # just return the participants as a single superset!
    if len(participants) <= maxBits:
        return [participants]


    kraftSum = sum(2**len(superset) for superset in supersets)
    widthRequired = math.ceil(math.log(kraftSum, 2.0))

    while (len(supersets) > 1):

        # bestSet1 and bestSet2 are our top choices for merging
        bestImpact = 0
        bestSet1 = None
        bestSet2 = None

        # for every pair of sets
        for set1 in supersets:
            for set2 in supersets:
                if (set1 == set2):
                    continue

                # how would a merge alter kraft's inequality?
                kraftImpact = 2**len(set1.union(set2)) - (2**len(set1) + 2**len(set2))

                # if the merge would cause us to exceed the bit limit
                if math.ceil(math.log(kraftSum + kraftImpact, 2.0)) > maxBits:
                    continue


                # choose the pair with the biggest impact on rules
                impact = 0
                for part in set1.intersection(set2):
                    if part not in ruleCounts:
                        a = list(ruleCounts.keys())
                        b = list(set1.intersection(set2))
                        c = list(set1)
                        d = list(set2)
                        a.sort()
                        b.sort()
                        c.sort()
                        d.sort()
                        print(part, "not in", a)
                        print("Intersection:", b)
                        print("Set 1:", c)
                        print("Set 2:", d)
                    impact += ruleCounts[part]

                if (impact > bestImpact):
                    bestImpact = impact
                    bestSet1 = set1
                    bestSet2 = set2

        # if the best change is an increase, break
        if (bestImpact == 0):
            break
        # merge the two best sets
        bestSet1.update(bestSet2)
        supersets.remove(bestSet2)

    return supersets

