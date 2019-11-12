import math
try:
    from .analyze import bitsRequiredFixedID, bitsRequiredVariableID
    from .AbsNode import AbsNode
except:
    from analyze import bitsRequiredFixedID, bitsRequiredVariableID
    from AbsNode import AbsNode

from itertools import combinations
from collections import deque as queue



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
        mergedSet = supersets[overlapGroup[0]]
        for setIndex in overlapGroup[1:]:
            mergedSet.update(supersets[setIndex])
        mergedSets.append(mergedSet)
    return mergedSets


def generateCodeWords(supersets, absHierarchy=False, prefix = ''):
    """ Given a list of supersets of varying with, produce a prefix-free code for labeling those supersets.
        Longer supersets will be assigned shorter codewords.
    """
    if not absHierarchy:
        minWidth = bitsRequiredVariableID(supersets)
        # indices of supersets that have no codes yet
        uncodedIndices = [i for i in range(len(supersets))]
        # sort it in descending order of available code widths
        uncodedIndices.sort(key = lambda index: minWidth - len(supersets[index]), reverse = True)
        codeLens = [minWidth - len(supersets[i]) for i in uncodedIndices]

        freeCodes = queue(['']) # right is head, left is tail
        assignedCodes = ['']*len(supersets)

        while len(uncodedIndices) > 0:
            # If we have enough unused codes for all supersets,
            #  OR if the current shortest codeword length is the limit for the longest uncoded superset
            if len(freeCodes) >= len(uncodedIndices) or len(freeCodes[-1]) == codeLens[-1]:
                ssIndex = uncodedIndices.pop()
                codeLens.pop()
                assignedCodes[ssIndex] = freeCodes.pop()
            # else, we split the shortest codeword
            else:
                codeToSplit = freeCodes.pop()
                freeCodes.extendleft([codeToSplit + c for c in ['0','1']])

        freeCodes = list(freeCodes)

        return (assignedCodes, freeCodes)
    else:
        assignedCodes, freeCodes, absCodes = generateCodeWordsRec(prefix, supersets)
        return (assignedCodes, freeCodes, absCodes)

def generateCodeWordsRec(prefix, supersets):
    '''
        recursive function of assigning codes
    '''
    if isinstance(supersets, AbsNode):
        absNode = supersets
        minWidth = len(supersets)
        supersets = supersets.getChildren()
    else:
        absNode = None
        supersets = sorted(supersets, key = lambda x: len(x))
        minWidth = bitsRequiredVariableID(supersets)

    codeLens = [minWidth - len(superset) for superset in supersets]
    freeCodes = queue(['']) # right is head, left is tail
    assignedCodes = {}
    absCodes = {}

    absColCoded= False
    while len(supersets) > 0:
        # If we have enough unused codes for all supersets,
        # OR if the current shortest codeword length is the limit for the longest uncoded superset
        if len(freeCodes) >= len(supersets) or len(freeCodes[-1]) == codeLens[-1]:
            codeLens.pop()
            superset = supersets.pop()
            if isinstance(superset, AbsNode):
                newprefix = freeCodes.pop()
                # TODO: take use of freeCodes2
                assignedCodes2, freeCodes2, absCodes2 = generateCodeWordsRec(prefix + newprefix, superset)
                assignedCodes.update(assignedCodes2)
                absCodes.update(absCodes2)
            else:
                if absNode and frozenset([absNode.absCol]) == superset:
                    absCodes[absNode.absCol] = (prefix, prefix + freeCodes.pop())
                    absColCoded = True
                else:
                    assignedCodes[superset] = prefix + freeCodes.pop()
        # else, we split the shortest codeword
        else:
            codeToSplit = freeCodes.pop()
            freeCodes.extendleft([codeToSplit + c for c in ['0','1']])
    if absNode and not absColCoded:
        absCodes[absNode.absCol] = (prefix, prefix)
    freeCodes = list(freeCodes)
    return assignedCodes, freeCodes, absCodes


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



def minimizeMemoryGreedy(matrix):
    """ Given a list of supersets, the number of rules needed regarding
        each participant in an outbound policy, and an upper bound
        on the number of bits we are willing to use, greedily minimize
        the number of rules that will result from the superset grouping.
    """
    # defensive copy
    matrix = [set(row) for row in matrix]

    numMatchStrings = sum(len(row) for row in matrix)

    kraftSum = sum(2**len(row) for row in matrix)

    while (len(matrix) > 1):

        currMemoryCost = numMatchStrings * (kraftSum.bit_length())
        # bestRow1 and bestRow2 are our top choices for merging
        bestNewMemory = currMemoryCost
        bestRow1 = None
        bestRow2 = None
        bestKraftImpact = None

        # for every pair of sets
        for row1, row2 in combinations(matrix, 2):
            # how would a merge alter kraft's inequality?
            kraftImpact = 2**len(row1.union(row2)) - (2**len(row1) + 2**len(row2))

            # choose the pair with the biggest impact on rules
            newTagWidth = (kraftSum + kraftImpact).bit_length()
            newMemoryCost = (numMatchStrings - len(row1.intersection(row2))) * newTagWidth

            if newMemoryCost < bestNewMemory:
                bestNewMemory = newMemoryCost
                bestRow1 = row1
                bestRow2 = row2
                bestKraftImpact = kraftImpact

        # if the best change is no change, break
        if (bestNewMemory == currMemoryCost):
            break
        # update objective function variables
        numMatchStrings -= len(bestRow1.intersection(bestRow2))
        kraftSum += bestKraftImpact
        # merge the two best sets
        bestRow1.update(bestRow2)
        matrix.remove(bestRow2)

    return matrix





