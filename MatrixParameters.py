import random
import numpy as np
import networkx as nx
import itertools
from cutsOverload import minimum_node_cut
from analyze import kraftsBound, transposeMatrix, groupOverlappingRows
import math

from unionFind import UnionFind
import sys
import logging
from collections import Counter, defaultdict
import json
import heapq

from util import printShellDivider, getShellWidth, shellHistogram

import random



def randomMatrix(rows=100, columns=100, density=0.1):
    matrixSize = rows * columns

    matrix = [set() for _ in range(rows)]
    for _ in range(int(matrixSize * density)):
        row = random.randint(0, rows-1)
        col = random.randint(0, columns-1)

        matrix[row].add(col)
    return matrix

def generateMatrix():
    density = 0.10 # average percent of rows that are full
    densityDev = 0.05 # deviation of percent that is added to or subtracted from density
    bridging = 0.10 # percent of blocks that overlap with each other
    hierarchy = None # how the hell we gonna implement this
    statics = 0.10 # percent of block attributes that are static
    skew = None # zipf / log distribution
    # number of distinct ASes that announce a prefix is skewed: ie many prefixes announced by only a single AS, few prefixes announcd many ASes

    width = 100 # number of attributes
    height = 100 # number of rows

    rowDensity = np.random.normal(loc=density, scale=densityDev)
    rowPop = max(1, int(width * rowDensity))


def matrixToRowSizeCounts(matrixWithCounts):
    rowSizes = []
    for row, count in matrixWithCounts:
        rowSizes.extend([len(row)]*count)

    sizeCounts = dict(Counter(rowSizes))

    return sizeCounts


def plotDistribution(matrixWithCounts, log=False):
    rowSizes = []
    for row, count in matrixWithCounts:
        rowSizes.extend([len(row)]*count)

    shellHistogram(rowSizes, title="Distribution of row sizes", log=log)


def distributionParameters(matrixWithCounts):
    sizeCounts = matrixToRowSizeCounts(matrixWithCounts)
    pass





def matrixToGraph(matrix):
    """ Docstring
    """
    G = nx.Graph()
    for row in matrix:
        for col1, col2 in itertools.combinations(row, 2):
            G.add_edge(col1, col2)
    return G





def extract(matrix, extractions):
    extractMatrix = []
    for row in matrix:
        extractMatrix.append(row.intersection(extractions))
        row.difference_update(extractions)

    return extractMatrix





tiebreaker = 0
def findBridges(G):
    global tiebreaker
    tiebreaker = 0
    bridges = []

    def _subgraphs(_G):
        return [_G.subgraph(cc).copy() for cc in nx.connected_components(_G)]

    subgraphMaxHeap = []
    def _push(*_subgraphs):
        global tiebreaker
        for _subgraph in _subgraphs:
            heapq.heappush(subgraphMaxHeap, (-len(_subgraph.nodes), tiebreaker, _subgraph))
            tiebreaker += 1
    def _pop():
        size, _, sg = heapq.heappop(subgraphMaxHeap)
        return -size, sg

    _push(*_subgraphs(G))

    lastSumSize = len(G.nodes)
    while len(subgraphMaxHeap)  > 0:
        largestClusterSize, subgraph = _pop()

        if largestClusterSize <= 2:
            break


        newSumSize = largestClusterSize + len(bridges)
        if newSumSize > lastSumSize:
            break
        lastSumSize = newSumSize

        # purely a timing optimization
        cut = None
        if len(subgraph) > 150:
            cut = minimum_node_cut(subgraph, approximate=1)
        else:
            cut = minimum_node_cut(subgraph, approximate=2)

        if len(cut) >= 2:
            print("Minimum cut size is", len(cut))
        for node in cut:
            bridges.append(node)
            subgraph.remove_node(node)
    
        _push(*_subgraphs(subgraph))

    return bridges


def approxMaxCut(G):
    E = len(G.edges)
    cutVal = 0

    while cutVal < E // 2:
        randCut = {node : random.getrandbits(1) for node in G.nodes}

        cutVal = 0
        for u,v in G.edges:
            if randCut[u] != randCut[v]:
                cutVal += 1

    part1, part2 = [], []
    for node, partID in randCut.items():
        if partID == 0:
            part1.append(node)
        else:
            part2.append(node)
    return (part1, part2)


def biggest(things):
    return max([len(thing) for thing in things])


def breakUpMatrix(matrix):
    matrices = [matrix]
    maxClusterSizes = [biggest(getClusters(matrix))]

    while True:
        oldMaxSizesSum = sum(maxClusterSizes)
        print("Before breaking, sizes are:", maxClusterSizes)
        print("Sum is", sum(maxClusterSizes))

        matrixToBreak = copyMatrix(matrices[-1])
        bridges = findBridges(matrixToGraph(matrixToBreak))
        bridgeMatrix = extract(matrixToBreak, bridges)

        brokenMatrixClusters = getClusters(matrixToBreak)
        lenBiggestBrokenCluster = biggest(brokenMatrixClusters)

        bridgeMatrixClusters = getClusters(bridgeMatrix)
        lenBiggestBridgeCluster = biggest(bridgeMatrixClusters)

        newMaxSizesSum = sum(maxClusterSizes[:-1]) + lenBiggestBrokenCluster + lenBiggestBridgeCluster
        if newMaxSizesSum >= oldMaxSizesSum: break

        matrices[-1] = matrixToBreak
        matrices.append(bridgeMatrix)

        maxClusterSizes[-1] = lenBiggestBrokenCluster
        maxClusterSizes.append(lenBiggestBridgeCluster)

    return matrices


def copyMatrix(matrix):
    return [set(row) for row in matrix]

def getMatrixStatistics(matrix, **extraInfo):
    matrix = [set(row) for row in matrix]
    logger = logging.getLogger("eval.matrixStats")
    allCols = set()
    for row in matrix:
        allCols.update(row)

    width = len(allCols)
    height = len(matrix)

    rowSizes = [len(row) for row in matrix]
    avgRowSize = sum(rowSizes) / height

    density = avgRowSize / width


    info = dict(extraInfo)
    info["width"] = width
    info["height"] = height
    info["avg row size"] = avgRowSize
    info["density"] = density

    info["row size counts"] = dict(Counter([len(row) for row in matrix]))

    tmatrix = transposeMatrix(matrix)
    info["col size counts"] = dict(Counter([len(row) for row in tmatrix]))

    clusters = groupOverlappingRows(matrix, asRows=False)
    info["cluster size counts"] = dict(Counter([len(cluster) for cluster in clusters]))

    logger.info(json.dumps(info))


def main():
    for filename in sys.argv[1:]:
        matrixWithCounts = None
        if filename.endswith(".json"):
            import json
            with open(filename, 'r') as fp:
                matrixWithCounts =  json.load(fp)
        elif filename.endswith('.pickle') or filename.endswith('.pkl'):
            import pickle
            with open(filename, 'rb') as fp:
                matrixWithCounts = pickle.load(fp)
        else:
            print("File format not recognized:", filename)
            continue

        print(matrixToRowSizeCounts(matrixWithCounts))
        plotDistribution(matrixWithCounts, log=True)

if __name__ == "__main__":
    main()
