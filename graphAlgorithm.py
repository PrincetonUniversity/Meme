from collections import defaultdict
import networkx as nx
import itertools
import copy
import time
import sys
from cutsOverload import minimum_node_cut
import logging
import json


try:
    from .analyze import bitsRequiredVariableID
    from .AbsNode import AbsNode
except:
    from analyze import bitsRequiredVariableID
    from AbsNode import AbsNode


def flatWidth(matrix):
    """
        Removes all subsets from a list of sets and print width/count of flat tags.
        Not used right now.
    """
    setList = copy.deepcopy(matrix)  # defensive copy
    finalAnswer = []                 # matrix after subset merging
    setList.sort(key=len, reverse=True)
    i = 0

    while i < len(setList):
        if len(setList[i]) != 0:
            finalAnswer.append(setList[i])
        for j in reversed(range(i+1, len(setList))):
            if setList[j].issubset(setList[i]):
                del setList[j]
        i += 1

    print(finalAnswer)
    print("tag width, ", bitsRequiredVariableID(finalAnswer))
    print("num_tags, ", sum([len(superset) for superset in finalAnswer]))
    print()
    return


def getCodingInformation(absHierarchy, selCols, separatePrefix):
    '''
        Get basic information of encoding
    '''
    supersetGroupings = sorted([len(superset) for superset in absHierarchy])
    absNodeGroupings = sorted([len(superset) for superset in absHierarchy if isinstance(superset, AbsNode)])
    absSupersetGroupings = []
    for node in absHierarchy:
        if isinstance(node, AbsNode):
            absSupersetGroupings.extend(node.getAllSupersets())
    absSupersetGroupings = [len(superset) for superset in absSupersetGroupings]

    tagwidth = bitsRequiredVariableID(absHierarchy)
    numabscol = sum([rootNode.getAbsCount() for rootNode in absHierarchy if isinstance(rootNode, AbsNode)])

    info = {}
    info["Superset groupings"]      = str(supersetGroupings)
    info["Absolute node groupings"] = str(absNodeGroupings)
    info["Superset groupings in absolute hierarchy"] = str(absSupersetGroupings)
    info["Tag width"]               = str(tagwidth)
    info["Selected columns"]        = str(selCols)
    info["Number of absolute columns"] =  str(numabscol)

    if numabscol > 0:
        info["Absolute columns"] = str(set.union(*[rootNode.getAbsCols() for rootNode in absHierarchy if isinstance(rootNode, AbsNode)]))
        info["Number of absolute columns with own encoding"] = str(len(separatePrefix))
        info["Absolute columns with own encoding"] = str(separatePrefix)
    return tagwidth, info


def outputTransform(supersets, absRoots, frozenMatrix, absThreshold = None):
    '''
        Construct absolute column hierarchy, find absolute columns that needs unique coding for itself
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


def extractRec(graph, absRoots, absParent, selCols, supersets, threshold):
    '''
        Recursive function to disconnect graphs into small connected components, 
        by extracting absolute columns (absRoots) and columns for the next matrix (selCols).
    '''

    # if graph contains more than 1 node, find absolute columns; otherwise, force into base case.
    if len(graph) != 1:
        colTups = graph.nodes.data('rows')
        colTups = sorted(colTups, reverse=True, key=lambda x : len(x[1]))
        possibleAbsCol = colTups[0][0]
        maxCols = colTups[0][1]
        allCols = set.union(*[colTup[1] for colTup in colTups])

        if len(allCols) == len(maxCols):
            isAbsCol = True
        else:
            isAbsCol = False

            for node, cols in colTups:
                # Approximation
                # if len(cols) < len(maxCols) - 3:
                #     break

                diffCols = allCols.difference(cols)
                possibleSelCols = []
                for node2, cols2 in colTups:
                    if len(diffCols.intersection(cols2)) != 0:
                        possibleSelCols.append(node2)
                if len(possibleSelCols) < 3 and len(possibleSelCols) != len(colTups) - 1:
                    selCols.update(possibleSelCols)
                    graph.remove_nodes_from(possibleSelCols)
                    possibleAbsCol = node
                    isAbsCol = True
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
    #                                                                   else => split.
    else:
        if len(graph) < threshold:
            if absParent != None:
                absParent.addSuperset(frozenset(graph.nodes()))
            else:
                supersets.append(frozenset(graph.nodes()))
            return
        else:
            if nx.is_connected(graph):
                cut = []
                # Approximation: if the graph size is above a threshold, use the approximate minimum_node_cut
                if len(graph.nodes) > 150:
                    cut = minimum_node_cut(graph, approximate = 2)
                else:
                    cut = minimum_node_cut(graph)
                selCols.update(cut)
                graph.remove_nodes_from(cut)

    # split into connected components
    # for each connected component, call extractGraphRec().

    # if isAbsCol:
    #     for cc in nx.connected_components(graph):
    #        print(len(cc))
    #     print()
    for cc in nx.connected_components(graph):
        subgraph = graph.subgraph(cc).copy()
        extractRec(subgraph, absRoots, absParent, selCols, supersets, threshold)
    return

def findOneCut(graph, findAll=True):
    """
        Another version of approximate minimum_node_cut.
        Find one-node cuts of the graph, if findAll is True => find all such cuts,
                                                  otherwise => find the first one-node cut.
        Not used right now.
    """
    cut = []
    for node in graph.nodes:
        newNodeList = list(graph.nodes)
        newNodeList.remove(node)
        if not nx.is_connected(graph.subgraph(newNodeList)):
            cut.append(node)
            if not findAll:
                break
    return cut

def extractNodes(matrix, threshold = 10):
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

    if graph.size() == len(graph) * (len(graph) - 1)/2:
        return [frozenset(graph.nodes())], set([]), []


    absRoots = []
    absParent = None
    selCols = set([])
    supersets = []

    extractRec(graph, absRoots, absParent, selCols, supersets, threshold)

    return supersets, selCols, absRoots


def graphHierarchy(matrix, parameters):
    '''
        Main algorithm. Iterate over the matrix till the selected columns (to the next submatrix) are empty.
        parameters = (threshold, absThreshold)
    '''

    if parameters != None:
        threshold = parameters[0]
        absThreshold = parameters[1]

    else:
        threshold = 10
        absThreshold = None

    widthsum = 0
    widths = []
    infoList = []
    supersetsList = []
    absHierarchyList = []
    matrix2 = matrix
    # matrix.sort(key=len, reverse=True)
    # print(matrix[0])

    while True:
        # call the main agorithm to get supersets, selCols and absRoots
        supersets, selCols, absRoots = extractNodes(matrix2, threshold)
        # construct supersets and absHierarchy;
        # extract additional columns if need to keep the superset tag wid under certain absThreshold;
        # find absolute columns that need its own unique encoding
        frozenMatrix = set([frozenset(row.difference(selCols)) for row in matrix])
        supersets, absHierarchy, addSelCols, separatePrefix = outputTransform(supersets, absRoots, frozenMatrix, absThreshold = None)

        # save information; if additional columns are selected to the next submatrix, extend selcols 
        selCols.update(addSelCols)
        supersetsList.append(supersets)
        absHierarchyList.append(absHierarchy)

        # get width and information of the grouping
        width, info = getCodingInformation(absHierarchy, selCols, separatePrefix)
        widthsum += width
        widths.append(width)
        infoList.append(info)

        # break the loop when the selected columns are empty
        if len(selCols) == 0:
            break
        else:
            matrix2 = [set(row).intersection(selCols) for row in matrix2]
            # flatWidth(matrix2)

    #print("Reaching width: ", widthsum, " (", str(widths), " )")

        logger = logging.getLogger("eval.graphAlg")
        logger.info(json.dumps(info))
    return supersetsList, absHierarchyList


def graphHierarchy_maxcut(matrix, parameters):
    '''
        Not any improvements
    '''
    import random 
    random.seed()

    elementSets = set.union(*[set(row) for row in matrix])
    partition1 = []
    partition2 = []
    for elem in elementSets:
        i = random.randint(1, 10)
        if i < 5.5:
            partition1.append(elem)
        else:
            partition2.append(elem)

    supersetsList = []
    absHierarchyList = []

    supersets1 = [set(row).intersection(partition1) for row in matrix]
    graph = nx.Graph()
    for col in partition1:
        graph.add_node(col)
    for row in supersets1:
        for i1, i2 in itertools.combinations(row, 2):
            graph.add_edge(i1, i2)
    supersets1 = []
    for cc in nx.connected_components(graph):
        #print(cc)
        supersets1.append(frozenset(cc))
    absHierarchyList.append(supersets1)
    supersets1 = [(superset, []) for superset in supersets1]
    supersetsList.append(supersets1)


    supersets2 = [set(row).intersection(partition2) for row in matrix]
    graph = nx.Graph()
    for col in partition2:
        graph.add_node(col)
    for row in supersets2:
        for i1, i2 in itertools.combinations(row, 2):
            graph.add_edge(i1, i2)
    supersets2 = []
    for cc in nx.connected_components(graph):
        #print(cc)
        supersets2.append(frozenset(cc))

    absHierarchyList.append(supersets2)
    supersets2 = [(superset, []) for superset in supersets2]
    supersetsList.append(supersets2)

    return supersetsList, absHierarchyList


