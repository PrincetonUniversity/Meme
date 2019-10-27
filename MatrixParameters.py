import numpy as np
import networkx as nx
import itertools
from cutsOverload import minimum_node_cut
from analyze import kraftsBound

from unionFind import UnionFind
import sys

import heapq
import random



def generateMatrix():
    density = 0.10 # average percent of rows that are full
    densityDev = 0.05 # deviation of percent that is added to or subtracted from density
    bridging = 0.10 # percent of blocks that overlap with each other
    hierarchy = None # how the hell we gonna implement this
    statics = 0.10 # percent of block attributes that are static

    width = 100 # number of attributes
    height = 100 # number of rows


def generateRow():
    rowDensity = np.random.normal(loc=density, scale=densityDev)
    rowPop = max(1, int(width * rowDensity))


def matrixToGraph(matrix):
    G = nx.Graph()
    for row in matrix:
        for col1, col2 in itertools.combinations(row, 2):
            G.add_edge(col1, col2)
    return G


def getClusters(matrix):
    uf = UnionFind()
    for row in matrix:
        if len(row) < 1: continue
        lrow = list(row)
        firstCol = lrow[0]
        firstColParent = uf.find(firstCol)
        for col in lrow[1:]:
            uf.union(firstColParent, col)

    return uf.components()


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
    G = matrixToGraph(matrix)


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

def analyzeMatrix(matrix):
    allCols = set()
    for row in matrix:
        allCols.update(row)

    width = len(allCols)
    height = len(matrix)

    rowSizes = [len(row) for row in matrix]
    avgRowSize = sum(rowSizes) / height

    density = avgRowSize / width



    print("Width:", width)
    print("Height:", height)
    print("Average row size:", avgRowSize)
    print("Density:", density)

    clusters = getClusters(matrix)
    maxClusterSize = max([len(cluster) for cluster in clusters])

    print("Max cluster size before breaking:", maxClusterSize)


    print("breaking..")

    matrices = breakUpMatrix(matrix)

    totalSize = 0
    totalTagSize = 0
    for i, submatrix in enumerate(matrices):
        subclustering = getClusters(submatrix)
        totalTagSize += kraftsBound([len(subcluster) for subcluster in subclustering])
        maxSubClusterSize = biggest(subclustering)
        print("Max cluster size of matrix %d: %d" % (i, maxSubClusterSize))
        totalSize += maxSubClusterSize
    print("total size of all largest clusters:", totalSize)
    print("total expected tag size:", totalTagSize)



if __name__ == "__main__":
    for filename in sys.argv[1:]:
        matrix = None
        if filename.endswith(".json"):
            import json
            with open(filename, 'r') as fp:
                matrix = json.load(fp)
        elif filename.endswith('.pickle') or filename.endswith('.pkl'):
            import pickle
            with open(filename, 'rb') as fp:
                matrix = pickle.load(fp)
        else:
            print("File format not recognized:", filename)
            continue

        matrix = copyMatrix(matrix)
        analyzeMatrix(matrix)
