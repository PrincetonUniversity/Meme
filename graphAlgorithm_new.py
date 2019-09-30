from collections import defaultdict
import networkx as nx
import itertools
import copy
import time
import sys

from typing import List,Tuple,Set,Collection

try:
    from .analyze import bitsRequiredVariableID
    from .AbsNode import AbsNode
except:
    from analyze import bitsRequiredVariableID
    from AbsNode import AbsNode


def graphHierarchy(matrix, threshold=10, absThreshold=None):
    '''
        Entry point to main algorithm. Iterate over the matrix till the selected columns (to the next submatrix) are empty.
    '''

    supersetsList = []
    absHierarchyList = []
    matrix2 = matrix
    # matrix.sort(key=len, reverse=True)
    # print(matrix[0])

    while True:
        # call the main algorithm to get supersets, extractedColumns and absRoots
        supersets, extractedColumns, absRoots = extractNodes(matrix2, threshold)
        # construct supersets and absHierarchy;
        # extract additional columns if need to keep the superset tag width under certain absThreshold;
        # find absolute columns that need its own unique encoding
        frozenMatrix = set([frozenset(row.difference(extractedColumns)) for row in matrix])
        supersets, absHierarchy, addSelCols, separatePrefix = outputTransform(supersets, absRoots, frozenMatrix, absThreshold = absThreshold)

        # save information; if additional columns are selected to the next submatrix, extend selcols 
        extractedColumns.update(addSelCols)
        supersetsList.append(supersets)
        absHierarchyList.append(absHierarchy)

        # break the loop when the selected columns are empty
        if len(extractedColumns) == 0:
            break
        else:
            matrix2 = [set(row).intersection(extractedColumns) for row in matrix2]

    return supersetsList, absHierarchyList



def extractNodes(matrix, threshold = 10) -> (List,List,List):
    '''
        Main algorithm: find connected components as grouping (supersets), 
                        select columns for the next submatrix (selcols), 
                        construct hierarchy of absolute columns with variable columns (absRoots)
    '''
    if len(matrix) == 0 :
        print("Empty matrix. Skipped!")
        return [], [], []

    colMap = defaultdict(list)
    allCols = set([])
    for i, row in enumerate(matrix):
        allCols.update(row)
        for col in row:
            colMap[col].append(i)

    graph = nx.Graph()
    for col in allCols:
        graph.add_node(col, rows = set(colMap[col]))

    for row in matrix:
        for i1, i2 in itertools.combinations(row, 2):
            graph.add_edge(i1, i2)

    absRoots = []
    absParent = None
    extractedColumns = set([])
    supersets = []

    extractRecursive(graph, absRoots, absParent, extractedColumns, supersets, threshold)

    return supersets, extractedColumns, absRoots



def outputTransform(supersets, absRoots, frozenMatrix, absThreshold = None) -> (List,List,List,List):
    '''
        Construct absolute column hierarchy, find absolute columns that needs unque coding for itself
        newsupersets is a list of tuples (variable columns, absolute columns)
        absHierarchy is a list of mixture of supersets and AbsNodes
    '''

    # add unique encoding for absolute columns and ***update the height***
    separatePrefix = []
    for absNode in absRoots:
        separatePrefix.extend(absNode.checkPrefix(frozenMatrix))

    # move absolute columns to the next submatrix if the hierarchy tag width is above absThreshold
    addSelCols = []
    if absThreshold:
        absNodesToExamine = absRoots
        absRoots = []
        while len(absNodesToExamine) > 0:
            absNode = absNodesToExamine.pop()
            if len(absNode) >= absThreshold:
                addSelCols.append(absNode.absCol)
                supersets.extend(absNode.ownSupersets)
                absNodesToExamine.extend(absNode.absChildren)
            else:
                absRoots.append(absNode)

    # convert absRoots and supersets to expected formats for MRCode
    absHierarchy = copy.deepcopy(supersets)
    absHierarchy.extend(absRoots)
    # if there is no absolute column, no need to encode an empty code
    if len(absRoots) != 0:
        absHierarchy.append(frozenset("E")) # empty code holder

    supersets = [(superset, []) for superset in supersets]
    for absNode in absRoots:
        supersets.extend(absNode.getSupersetPairs())

    return supersets, absHierarchy, addSelCols, separatePrefix


def extractRecursive(graph, absRoots, absParent, extractedColumns, supersets, threshold):
    '''
        Recursive function to disconnect graphs into small connected components, 
        by extracting absolute columns (absRoots) and columns for the next matrix (extractedColumns).
    '''

    # if graph contains more than 1 node, find absolute columns; otherwise, force into base case.
    if len(graph) != 1:
        possibleAbsColTup = max(graph.nodes(data=True), key= lambda x : len(x[1]['rows']))
        possibleAbsCol = possibleAbsColTup[0]
        maxRows = possibleAbsColTup[1]['rows']
        isAbsCol = True
        for col in graph:
            if not graph.node[col]['rows'].issubset(maxRows):
                isAbsCol = False
                break
    else:
        isAbsCol = False

    # if there is an absolute column => if there is no parent, add to the absRoots; otherwise, add to parent's children list.
    #                                   update absParent.
    #                                   delete the column, and continue to split.
    if isAbsCol:
        newAbsNode = AbsNode(possibleAbsCol)
        if absParent != None:
            absParent.addChild(newAbsNode)
        else:
            absRoots.append(newAbsNode)
        absParent = newAbsNode
        graph.remove_node(possibleAbsCol)

    # if there is no absolute column => if the size is below threshold, add to superests/parent's ownSupersets and stop (Base Case);
    #                                   otherwise, if the graph is connected => take out minimum vertex cuts and split;
    #                                                                           else, split.
    else:
        if len(graph) < threshold:
            if absParent != None:
                absParent.addSuperset(frozenset(graph.nodes()))
            else:
                supersets.append(frozenset(graph.nodes()))
            return
        else:
            if nx.is_connected(graph):
                cut = nx.minimum_node_cut(graph)
                #print(len(cut))
                extractedColumns.update(cut)
                graph.remove_nodes_from(cut)

    # split into connected components
    # for each connected component, call extractGraphRec().
    for cc in nx.connected_components(graph):
        subgraph = graph.subgraph(cc).copy()
        extractRecursive(subgraph, absRoots, absParent, extractedColumns, supersets, threshold)
    return




