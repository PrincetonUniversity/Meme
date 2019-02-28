import math
from analyze import bitsRequiredFixedID, bitsRequiredVariableID


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


def minimizeVariableWidthGreedy(supersets):
    """ Given a list of supersets, greedily minimize the number of bits required
        when the superset IDs are a prefix code.
    """
    # defensive copy
    supersets = [set(superset) for superset in supersets]

    kraftSum = sum(2**len(superset) for superset in supersets)

    minKraftSum = kraftSum

    originalSupersets = [set(superset) for superset in supersets]
    originalKraftSum = kraftSum


    while (len(supersets) > 1):
        minKraftImpact = float("inf")
        bestSet1 = None
        bestSet2 = None
        for set1 in supersets:
            for set2 in supersets:
                if set1 == set2:
                    continue

                # remove the two sets from the sum, add their union
                kraftImpact = -(2**len(set1) + 2**len(set2)) + 2**len(set1.union(set2))

                if kraftImpact < minKraftImpact:
                    minKraftImpact = kraftImpact
                    bestSet1 = set1
                    bestSet2 = set2

        bestSet1.update(bestSet2)
        supersets.remove(bestSet2)

        kraftSum += minKraftImpact

        minKraftSum = min(kraftSum, minKraftSum)

    ## We just merged sets until we couldn't merge anymore, and we kept track of
    ## the min kraft inequality we saw. Now lets do it again but stop at the min.
    supersets = originalSupersets
    kraftSum = originalKraftSum

    while (kraftSum != minKraftSum):
        minKraftImpact = float("inf")
        bestSet1 = None
        bestSet2 = None
        for set1 in supersets:
            for set2 in supersets:
                if set1 == set2:
                    continue

                # remove the two sets from the sum, add their union
                kraftImpact = -(2**len(set1) + 2**len(set2)) + 2**len(set1.union(set2))

                if kraftImpact < minKraftImpact:
                    minKraftImpact = kraftImpact
                    bestSet1 = set1
                    bestSet2 = set2

        bestSet1.update(bestSet2)
        supersets.remove(bestSet2)

        kraftSum += minKraftImpact

        minKraftSum = min(kraftSum, minKraftSum)


    return supersets



def minimizeFixedWidthGreedy(supersets):
    """ Given a list of supersets, greedily minimize the number of bits required
        to represent any set as a superset ID and corresponding bitmask.
    """
    # defensive copy
    supersets = [set(superset) for superset in supersets]
    originalSupersets = [set(superset) for superset in supersets]

    # the longest superset determines the current mask size
    maxLength = max([len(superset) for superset in supersets])

    tagWidth = bitsRequiredFixedID(supersets)
    minTagWidth = tagWidth

    while (len(supersets) > 1):
        minUnionSize = maxLength * 2
        # bestSet1 and bestSet2 are our top choices for merging
        bestSet1 = None
        bestSet2 = None

        # for every pair of sets
        for set1 in supersets:
            # If the set size is larger than our current best merge size, then skip it.
            if  len(set1) > minUnionSize:
                continue
            for set2 in supersets:
                # Ditto
                if len(set2) > minUnionSize:
                    continue
                if (set1 == set2):
                    continue

                unionSize = len(set1.union(set2))

                # choose the pair with the smallest union size
                if (unionSize < minUnionSize):
                    minUnionSize = unionSize
                    bestSet1 = set1
                    bestSet2 = set2

        # merge the two best sets
        bestSet1.update(bestSet2)
        supersets.remove(bestSet2)
        # update the mask size if necessary
        maxLength = max(len(bestSet1), maxLength)

        minTagWidth = min(minTagWidth, bitsRequiredFixedID(supersets))


    # DO IT AGAIN BRO I DARE YOU
    supersets = originalSupersets
    tagWidth = bitsRequiredFixedID(supersets)

    while tagWidth != minTagWidth:
        minUnionSize = maxLength * 2
        # bestSet1 and bestSet2 are our top choices for merging
        bestSet1 = None
        bestSet2 = None

        # for every pair of sets
        for set1 in supersets:
            # If the set size is larger than our current best merge size, then skip it.
            if  len(set1) > minUnionSize:
                continue
            for set2 in supersets:
                # Ditto
                if len(set2) > minUnionSize:
                    continue
                if (set1 == set2):
                    continue

                unionSize = len(set1.union(set2))

                # choose the pair with the smallest union size
                if (unionSize < minUnionSize):
                    minUnionSize = unionSize
                    bestSet1 = set1
                    bestSet2 = set2

        # merge the two best sets
        bestSet1.update(bestSet2)
        supersets.remove(bestSet2)
        # update the mask size if necessary
        maxLength = max(len(bestSet1), maxLength)
        tagWidth = bitsRequiredFixedID(supersets)

    return supersets



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

