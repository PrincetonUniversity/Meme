import math, heapq
import networkx as nx
from unionFind import UnionFind
from typing import List, Set
from cutsOverload import minimum_node_cut
import logging
import util


logger = logging.getLogger("Analyze")

_tiebreaker = 0
def findBridges(G, maxComponentSize=None):
    global _tiebreaker
    _tiebreaker = 0
    bridges = []

    def _subgraphs(_G):
        return [_G.subgraph(cc).copy() for cc in nx.connected_components(_G)]

    subgraphMaxHeap = []
    def _push(*_subgraphs):
        global _tiebreaker
        for _subgraph in _subgraphs:
            heapq.heappush(subgraphMaxHeap, (-len(_subgraph.nodes), _tiebreaker, _subgraph))
            _tiebreaker += 1
    def _pop():
        size, _, sg = heapq.heappop(subgraphMaxHeap)
        return -size, sg

    _push(*_subgraphs(G))

    lastSumSize = len(G.nodes)
    while len(subgraphMaxHeap)  > 0:
        largestClusterSize, subgraph = _pop()

        if largestClusterSize <= 2:
            break
        
        # if a threshold was given on component sizes, we dont end until we get below it
        if (maxComponentSize != None) and (largestClusterSize <= maxComponentSize):
            break

        """ if no threshold was given, we instead remove bridges until it appears that the sum
            (largest_component_size + num_bridges) starts increasing
        """ 
        newSumSize = largestClusterSize + len(bridges) 
        if (maxComponentSize == None) and (newSumSize > lastSumSize):
            break
        lastSumSize = newSumSize

        # purely a timing optimization
        cut = None
        if len(subgraph) > 150:
            cut = minimum_node_cut(subgraph, approximate=1)
        else:
            cut = minimum_node_cut(subgraph, approximate=2)

        for node in cut:
            bridges.append(node)
            subgraph.remove_node(node)
    
        _push(*_subgraphs(subgraph))

    return bridges

def dissectMatrix(matrix, matrixCostFunc=None, **bridgekwargs):
    # this matrix cost func will force cutting until no bridge nodes can be found
    if matrixCostFunc == None:
        matrixCostFunc = lambda x: -1


    matrices = [matrix]
    matrixCosts = [matrixCostFunc(matrix)]

    while True:
        oldCostSum = sum(matrixCosts)
        logger.debug("Before cutting, matrix costs are: " + str(matrixCosts))
        logger.debug("Matrix cost sum is " + str(oldCostSum))

        matrixToCut = util.copyMatrix(matrices[-1])
        bridges = findBridges(util.matrixToGraph(matrixToCut), **bridgekwargs)
        # if no bridges are found, exit
        if len(bridges) == 0:
            break
        bridgeMatrix = util.extractSubmatrix(matrixToCut, bridges)

        cutMatrixCost = matrixCostFunc(matrixToCut)
        bridgeMatrixCost = matrixCostFunc(bridgeMatrix)

        newCostSum = sum(matrixCosts[:-1]) + cutMatrixCost + bridgeMatrixCost
        if newCostSum >= oldCostSum: break

        matrices[-1] = matrixToCut
        matrices.append(bridgeMatrix)

        matrixCosts[-1] = cutMatrixCost
        matrixCosts.append(bridgeMatrixCost)

    return matrices



def ternary_compare(str1, str2):
    if len(str1) != len(str2):
        raise Exception("strings of unequal length compared: %s and %s" %(str1, str2))
    for (c,d) in zip(str1,str2):
        # compare every pair of bits. failure short-circuits
        if not ((c=='*') or (d=='*') or (c==d)):
            return False
    return True





def groupOverlappingRows(matrix:List[Set], asRows=True, withImplicits=False):
    """ Given a matrix, return the rows of the matrix partitioned into groups,
        where no two rows from different groups intersect.
        Unless asRows is False, then the groups returned are the union of the rows.
    """
    G = nx.Graph()
    colOccurrences = {}
    for rowID, row in enumerate(matrix):
        for colID in row:
            G.add_edge(("R", rowID), ("C", colID))
            colOccurrences[colID] = colOccurrences.get(colID, 0) + 1

    components = nx.connected_components(G)

    colGroups = []
    rowGroups = []
    for cc in components:
        colGroup = []
        rowGroup = []
        for nodeType, node in cc:
            if nodeType == "C":
                colGroup.append(node)
            else:
                rowGroup.append(node)
        colGroups.append(colGroup)
        rowGroups.append(rowGroup)

    if asRows:
        returnGroups = rowGroups
    else:
        returnGroups = colGroups

    if withImplicits:
        implicitGroups = [[col for col in colGroup if colOccurrences[col] == len(rowGroup)] \
                            for (colGroup, rowGroup) in zip(colGroups, rowGroups)]
        return (returnGroups, implicitGroups)
    else:
        return returnGroups


def transposeMatrix(matrix : List[Set], frozen=False) -> List[Set]:
    transposed = {}
    for rowID, row in enumerate(matrix):
        for colID in row:
            col = transposed.setdefault(colID, set())
            col.add(rowID)
    if frozen:
        transposed = {colID:frozenset(rowIDs) for colID,rowIDs in transposed.items()}
    return transposed




def groupIdenticalColumns(elementSets):
    colIDs = set.union(*[set(es) for es in elementSets])
    # transpose the element matrix (so its a collection of columns instead of rows)
    transposed = transposeMatrix(elementSets, frozen=True)
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


def kraftsBound(lengths):
    kraftSum = sum(2**length for length in lengths)
    return (kraftSum-1).bit_length()


def bitsRequiredVariableID(supersets):
    """ How many bits are needed to represent any set in this superset grouping?
        Assumes optimal variable-width superset identifiers.
    """
    return kraftsBound([len(superset) for superset in supersets])


def rulesRequired(supersets, ruleCounts):
    """ How many rules will be needed given a superset grouping and a
        dictionary of rule counts associated with each participant in a policy?
    """
    total = 0
    for superset in supersets:
        for part in superset:
            total += ruleCounts[part]
    return total
