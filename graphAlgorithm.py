from collections import defaultdict
import networkx as nx
import itertools
import copy
import time
import sys

try:
    from .analyze import bitsRequiredVariableID
    from .AbsNode import AbsNode
except:
    from analyze import bitsRequiredVariableID
    from AbsNode import AbsNode


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

    info = "Superset groupings\n" + str(supersetGroupings) \
           + "\nAbsolute node groupings\n" + str(absNodeGroupings) \
           + "\nSuperset groupings in absolute hierarchy\n" + str(absSupersetGroupings)\
           + "\nTag width\n" + str(tagwidth) \
           + "\nSelected columns\n" + str(selCols) \
           + "\nNumber of absolute columns\n" +  str(numabscol)

    if numabscol > 0:
        info += "\nAbsolute columns:\n" +  str(set.union(*[rootNode.getAbsCols() for rootNode in absHierarchy if isinstance(rootNode, AbsNode)]))
        info += "\nNumber of absolute columns with own encoding:\n" + str(len(separatePrefix))
        info += "\nAbsolute columns with own encoding:\n" + str(separatePrefix)
    info += "\n\n"
    return tagwidth, info


def outputTransform(supersets, absRoots, frozenMatrix, absThreshold = None):
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
                selCols.update(cut)
                graph.remove_nodes_from(cut)

    # split into connected components
    # for each connected component, call extractGraphRec().
    for cc in nx.connected_components(graph):
        subgraph = graph.subgraph(cc).copy()
        extractRec(subgraph, absRoots, absParent, selCols, supersets, threshold)
    return


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

    while True:
        # call the main agorithm to get supersets, selCols and absRoots
        supersets, selCols, absRoots = extractNodes(matrix2, threshold)
        # construct supersets and absHierarchy;
        # extract additional columns if need to keep the superset tag wid under certain absThreshold;
        # find absolute columns that need its own unique encoding
        frozenMatrix = set([frozenset(row.difference(selCols)) for row in matrix])
        supersets, absHierarchy, addSelCols, separatePrefix = outputTransform(supersets, absRoots, frozenMatrix, absThreshold = absThreshold)

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

    print("Reaching width: ", widthsum, " (", str(widths), " )")
    #for info in infoList: print(info)
    return supersetsList, absHierarchyList

