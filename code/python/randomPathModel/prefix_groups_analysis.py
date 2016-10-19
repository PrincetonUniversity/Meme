import math

def removeSubsets(prefixSets):
    """ Removes all subsets from a list of sets.
    """
    final_answer = []

    # defensive copy
    sets = [set(prefix) for prefix in prefixSets]
    sets.sort(key=len, reverse=True)
    i = 0
    while i < len(sets):
        final_answer.append(sets[i])

        for j in reversed(range(i+1, len(sets))):
            if sets[j].issubset(sets[i]):
                del sets[j]

        i += 1

    return final_answer


def bitsRequired(supersets):
    """ How many bits are needed to represent a set in this superset group?
    """
    logM = 0
    if len(supersets) > 1:
        logM = math.ceil(math.log(len(supersets) - 1, 2))
    maxS = max(len(superset) for superset in supersets)

    return int(logM + maxS)


def rulesRequired(supersets, ruleCounts):
    """ How many rules will be needed given a superset grouping and a
        dictionary of rule counts associated with each participant in a policy?
    """
    total = 0
    for superset in supersets:
        for part in superset:
            total += ruleCounts[part]
    return total


def minimize_bits_greedy(peerSets):
    """ Given a list of sets, greedily minimize the number of bits requied
        to represent any set as a superset ID and corresponding bitmask.
    """
    # defensive copy
    peerSets = [set(peerSet) for peerSet in peerSets]
    # smallest sets first
    peerSets.sort(key=len)


    # the longest superset determines the current mask size
    maxLength = max([len(prefix) for prefix in peerSets])

    while (len(peerSets) > 1):
        m = len(peerSets)

        # if M is 1 + 2^X for some X, logM will decrease after a merge
        deltaLog = 0
        if ((m - 1) & (m - 2)) == 0:
            deltaLog = -1

        # bestSet1 and bestSet2 are our top choices for merging
        minUnionSize = maxLength + 1
        minDelta = 1
        bestSet1 = None
        bestSet2 = None

        # for every pair of sets
        for set1 in peerSets:
            # If the sets from here on out are worse than our current best answer,
            # then break the loop. Recall the list is in increasing order of set size.
            if  len(set1) > minUnionSize:
                break
            for set2 in peerSets:
                # Ditto
                if len(set2) > minUnionSize:
                    break
                if (set1 == set2):
                    continue

                unionSize = len(set1.union(set2))

                # the merge's impact on the largest superset size
                delta = max(0, unionSize - maxLength)
                # add the merge's impact on the log_2(m) term
                delta += deltaLog

                # choose the pair with the smallest union size
                if (unionSize < minUnionSize):
                    minUnionSize = unionSize
                    minDelta = min(1, delta)
                    bestSet1 = set1
                    bestSet2 = set2

        # if the best change is an increase, break
        if (minDelta > 0):
            break
        # merge the two best sets
        bestSet1.update(bestSet2)
        peerSets.remove(bestSet2)
        # update the mask size if necessary
        maxLength = max(len(bestSet1), maxLength)

    return peerSets


def minimize_rules_greedy(peerSets, ruleCounts, maxBits):
    """ Given a list of supersets, the number of rules needed regarding
        each participant in an outbound policy, and an upper bound
        on the number of bits we are willing to use, greedily minimize
        the number of rules that will result from the superset grouping.
    """
    # defensive copy
    peerSets = [set(peerSet) for peerSet in peerSets]
    # smallest sets first
    peerSets.sort(key=len, reverse=True)

    # build the set of all participants
    participants = set()
    for peerSet in peerSets:
        participants.update(peerSet)

    # if the number of participants is less than the max number of bits,
    # just return the participants as a single superset!
    if len(participants) <= maxBits:
        return [participants]


    # the longest superset determines the current mask size
    maxLength = max([len(prefix) for prefix in peerSets])

    while (len(peerSets) > 1):
        m = len(peerSets)

        bits = bitsRequired(peerSets)

        # if M is 1 + 2^X for some X, logM will decrease after a merge
        if ((m - 1) & (m - 2)) == 0:
            bits = bits - 1


        # bestSet1 and bestSet2 are our top choices for merging
        bestImpact = 0
        bestSet1 = None
        bestSet2 = None

        # for every pair of sets
        for set1 in peerSets:
            for set2 in peerSets:
                if (set1 == set2):
                    continue

                unionSize = len(set1.union(set2))

                # if the merge would cause us to exceed the bit limit
                if bits + max(0, unionSize - maxLength) > maxBits:
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
                        print part, "not in", a
                        print "Intersection:", b
                        print "Set 1:", c
                        print "Set 2:", d
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
        peerSets.remove(bestSet2)
        # update the mask size if necessary
        maxLength = max(len(bestSet1), maxLength)

    return peerSets
