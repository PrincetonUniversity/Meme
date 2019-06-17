import math



def ternary_compare(str1, str2):
    if len(str1) != len(str2):
        raise Exception("strings of unequal length compared: %s and %s" %(str1, str2))
    for (c,d) in zip(str1,str2):
        # compare every pair of bits. failure short-circuits
        if not ((c=='*') or (d=='*') or (c==d)):
            return False
    return True


def groupIdenticalColumns(elementSets):
    colIDs = set.union(*[set(es) for es in elementSets])
    # transpose the element matrix (so its a collection of columns instead of rows)
    transposed = {colID:frozenset([rowID for (rowID,row) in enumerate(elementSets) if colID in row]) for colID in colIDs}
    # build lists of columnIDs for columns that are identical
    identicalColGroups = {}
    for colID, col in transposed.items():
        group = identicalColGroups.get(col, None)
        if group is not None:
            group.append(colID)
        else:
            identicalColGroups[col] = [colID]
    return list(identicalColGroups.values())



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
        logM = (len(supersets)-1).bit_length()
    maxS = max(len(superset) for superset in supersets)

    return int(logM + maxS)


def bitsRequiredVariableID(supersets):
    """ How many bits are needed to represent any set in this superset grouping?
        Assumes optimal variable-width superset identifiers.
    """
    kraftSum = sum(2**len(superset) for superset in supersets)
    return (kraftSum-1).bit_length()


def rulesRequired(supersets, ruleCounts):
    """ How many rules will be needed given a superset grouping and a
        dictionary of rule counts associated with each participant in a policy?
    """
    total = 0
    for superset in supersets:
        for part in superset:
            total += ruleCounts[part]
    return total
