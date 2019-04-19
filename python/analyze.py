import math

def isSubsetOfSuperset(subset, supersets):

    for superset in supersets:
        if (set(superset)).issuperset(subset):
            return True
    return False


def getSupersetIndex(subset, supersets):
    for (i, superset) in enumerate(supersets):
        if (set(superset)).issuperset(subset):
            return i
    return -1

def bitsRequiredFixedID(supersets):
    """ How many bits are needed to represent any set in this superset grouping?
        Assumes fixed-width superset identifiers.
    """
    logM = 0
    if len(supersets) > 1:
        logM = math.ceil(math.log(len(supersets) - 1, 2))
    maxS = max(len(superset) for superset in supersets)

    return int(logM + maxS)



def bitsRequiredVariableID(supersets):
    """ How many bits are needed to represent any set in this superset grouping?
        Assumes optimal variable-width superset identifiers.
    """
    kraftSum = sum(2**len(superset) for superset in supersets)
    # print("Before ceiling, width: ",math.log(kraftSum, 2.0) + 0.5)
    return math.ceil(math.log(kraftSum, 2.0) + 0.5)


def rulesRequired(supersets, ruleCounts):
    """ How many rules will be needed given a superset grouping and a
        dictionary of rule counts associated with each participant in a policy?
    """
    total = 0
    for superset in supersets:
        for part in superset:
            total += ruleCounts[part]
    return total