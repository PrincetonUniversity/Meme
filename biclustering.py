from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category = ConvergenceWarning)
warnings.filterwarnings("error", message = ".*divide by zero.*")

from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import random
import pickle
import numpy as np
import copy
import time
import math
import sys

from numpy.random import RandomState


try:
    from .analyze import groupIdenticalColumns, bitsRequiredVariableID
except:
    from analyze import groupIdenticalColumns, bitsRequiredVariableID


RANSDOMSTATE = RandomState()

class AbsNode:
    ownSupersets = None     # List of frozenset
    fullSupersets = None    # Set of frozenset (optional, can be None)
    absChildren = None      # List of AbsNodes
    absCol : int = 0
    height : int = 0
    ownPrefix : bool = False

    def __init__(self, absCol, supersets=None, posssibleChildren=None):
        self.absCol = absCol
        self.absChildren = []
        self.ownPrefix = False
        if supersets and posssibleChildren:
            self.fullSupersets = set(supersets)
            
            
            childSupersets = set([])
            for node in posssibleChildren:
                if node.fullSupersets.issubset(supersets):
                    childSupersets.update(node.fullSupersets)
                    self.absChildren.append(node)

            self.ownSupersets = list(self.fullSupersets.difference(childSupersets))
            self.height = bitsRequiredVariableID(self.getChildren())
        else:
            self.fullSupersets = None
            self.ownSupersets = []
            self.height = 0


    def addChild(self, child):
        self.absChildren.append(child)


    # expected argument: frozenset!
    def addSuperset(self, superset):
        self.ownSupersets.append(superset)


    def calculateHeight(self):
        self.height = bitsRequiredVariableID(self.getChildren())


    def getChildren(self):
        childrenList = self.ownSupersets + self.absChildren
        if self.ownPrefix:
            childrenList.append(frozenset([self.absCol]))
        return sorted(childrenList, key = lambda x: len(x))


    # always get self.height with this function to guarantee it is up to date
    def __len__(self):
        self.calculateHeight()
        return self.height


    def checkPrefix(self, frozenMatrix, parentAbsCols = []):
        # If there is no child, the confusion is avoided!
        if len(self.absChildren) == 0: return []

        newparentAbsCols = copy.deepcopy(parentAbsCols)
        newparentAbsCols.append(self.absCol)

        if frozenset(newparentAbsCols) in frozenMatrix:
            self.ownPrefix = True
            result = [self.absCol] # for printing purpose
        else:
            result = [] # for printing purpose

        for node in self.absChildren:
            result.extend(node.checkPrefix(frozenMatrix, newparentAbsCols))

        self.height = bitsRequiredVariableID(self.getChildren())
        return result


    def getSupersetPairs(self, parentAbsCols = []):

        newparentAbsCols = copy.deepcopy(parentAbsCols)
        newparentAbsCols.append(self.absCol)
        result = [(superset, newparentAbsCols) for superset in self.ownSupersets]

        for node in self.absChildren:
            result.extend(node.getSupersetPairs(newparentAbsCols))

        return result


    def __str__(self):
        string = "absCol: " + str(self.absCol) + " "
        string +=  "absChildren: " + str([str(absNode) for absNode in self.absChildren]) + " "
        string +=  "ownSupersets: " + str(self.ownSupersets) + "    "
        string +=  "ownPrefix: " + str(self.ownPrefix) + "    "
        return string

    def getAbsCount(self):
        return 1 + sum([node.getAbsCount() for node in self.absChildren])

    def getAbsCols(self):
        if not self.absChildren: return set([self.absCol])

        result = set.union(*[node.getAbsCols() for node in self.absChildren])
        result.add(self.absCol)
        return result

    def getAllCols(self):
        if len(self.ownSupersets) == 0:
            result = set([])
        else:
            result = set.union(*[set(superset) for superset in self.ownSupersets])
        result.add(self.absCol)

        for node in self.absChildren:
            result.update(node.getAllCols())

        return result
        

def removeSubsetsGetAbsCandidate(matrix):
    """ 
        Removes all subsets from a list of sets and return absCol candidates.
    """
    setList = copy.deepcopy(matrix)  # defensive copy
    absColCandMap = {}               # map from column to True/False
    finalAnswer = []                 # matrix after subset merging
    setList.sort(key=len, reverse=True)
    i = 0

    while i < len(setList):
        finalAnswer.append(setList[i])
        absColCandidate = setList[i]    # keep track of original columns
        absColCandidate2 = setList[i]   # keep track of maybe absolute candidates in this superset

        for j in reversed(range(i+1, len(setList))):
            if setList[j].issubset(setList[i]):
                # Find columns that are absolute in this superset
                absColCandidate2 = absColCandidate2.intersection(setList[j])
                del setList[j]

        # Only columns that are always absolute in all supersets can be absolute candidates
        for col in absColCandidate:
            if absColCandMap.get(col, True) and col in absColCandidate2:
                absColCandMap[col] = True
            else:
                absColCandMap[col] = False

        i += 1

    absColCand = [k for k, v in absColCandMap.items() if v]
    return finalAnswer, absColCand


def get_cluster_cols(model):
    cluster_cols = {}
    for i in range(model.get_params()["n_clusters"]):
        cluster_cols[i] = []

    for i, cluster_id in enumerate(model.column_labels_):
        cluster_cols[cluster_id].append(i)
    
    #print(sorted([(k, len(v))for k, v in cluster_cols.items()], key = lambda x : x[1]))
    return cluster_cols


def good_form(model, data):
    '''
        test the biclustering result of convergence
    '''
    data_count = np.count_nonzero(data)
    bicluster_count = 0
    
    for j in range(model.get_params()["n_clusters"]):
        bicluster_count += np.count_nonzero(model.get_submatrix(j, data))
    
    if bicluster_count != data_count:
        return False
    return True


def fitLoop(model, data):
    '''
        Loop for fitting to find n_svd_vecs
    '''
    #num_clusters = model.get_params()["n_clusters"]
    try:
        model.fit(data)
    except Exception as e:
        print(e)
        # The number of Lanczos vectors generated ncv must be greater than k+1 and smaller than n;
        # it is recommended that ncv > 2*k Default: min(n, max(2*k + 1, 20))
        #model.set_params(n_svd_vecs= min(min(data.shape), max(2*num_clusters + 1, 20) + i))
        #    if j > 100:
        raise Exception("Fit fails with %d columns!" % (data.shape[1]))
    return model


def plot(model, data, layer):
    '''
        plot the matrix
    '''
    fit_data = data[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    plt.figure(figsize=(4,4))
    plt.spy(fit_data, markersize=0.4, aspect='auto')
    plt.title("Spectral CoClustering for Layer %d" % (layer))
    plt.show()


def extractRec(data, colmap, absmap, abscolcand, abscols, selcols, supersets, num_clusters, col_threshold, sel_factor, layer, final=False, allmin=False):
    '''
        recursive function to cluster biclusters into smaller chunks, by finding abscols and extracting selcols.
    '''
    # base case
    if data.shape[1] == 0:
        return

    # find absolute columns
    abscols2 = copy.deepcopy(abscols)
    if data.shape[1] != 1:
        allone = np.all(data == 1, axis=0)
        allone = np.asarray(allone).reshape(-1)
        allone = np.array([False if colmap[k[0]] not in abscolcand else v for k, v in np.ndenumerate(allone)])

        newabscol = [colmap[col] for col in np.where(allone)[0]]
        assert(len(newabscol) <= 1)

        abscols2.extend([colmap[col] for col in np.where(allone)[0]])
        notallone = ~allone
        data = data[:, notallone]
        notallone = np.asarray(notallone).reshape(-1)
        colmap = {k : colmap[v] for k, v in enumerate(np.where(notallone)[0])}

        nonzero = ~np.all(data == 0, axis=1)
        nonzero = np.asarray(nonzero).reshape(-1)
        data = data[nonzero]

    # base case "final": add absolute column mappings from abscol to variable column supersets; append supersets without absolute columns
    if final:
        if len(abscols2) != 0:
            for abscol in abscols2:
                absmap[abscol].append(frozenset(colmap.values()))
        else:
            supersets.append(frozenset(colmap.values()))
        return

    # iteratively calling biclustering algorithm to find the largest n_clusters as possible
    model = SpectralCoclustering(n_clusters=num_clusters, random_state=RANSDOMSTATE, svd_method='arpack')
    while num_clusters > 1:
        try:
            model.fit(data)
            #model = fitLoop(model, data)
            # check if biclustering succeeds
            if good_form(model, data):
                break

        except Exception as e: 
            print(e)
            #if len(abscols2) != 0:
            #    for abscol in abscols2:
            #        absmap[abscol].append(frozenset(colmap.values()))
            #else:
            #    supersets.append(frozenset(colmap.values()))
            #return

        # iteratively cut the n_clusters
        if layer == 0:
            num_clusters = num_clusters - 1
        else:
            num_clusters = math.ceil(num_clusters/2.0)
        model.set_params(n_clusters=num_clusters)

    # if the original bicluster is smaller then the col_threshold, call the base case "final" to add the results to absmap and supersets;
    # the reason of such the condition is to try splitting the bicluster one more time when it is below the threshold.
    # For instance, if the bicluster is of size 9 < threshold 10, try splitting one more time may yield 9 one-element bisluters and optimize
    # the grouping.
    if data.shape[1] < col_threshold or allmin:
        if num_clusters < 2:
            extractRec(data, colmap, absmap, abscolcand, abscols2, selcols, supersets, min(data.shape)-1, col_threshold, sel_factor, layer + 1, final = True)
        else:
            for j in range(num_clusters):
                data2 = model.get_submatrix(j, data)
                colmap2 = {k : colmap[v] for k, v in enumerate(model.get_indices(j)[1].tolist())}
                extractRec(data2, colmap2, absmap, abscolcand, abscols2, selcols, supersets, min(data2.shape)-1, col_threshold, sel_factor, layer + 1, final = True)
        return

    # if the biclustering fails (num_clusters = 1), extract heavy hitter columns to the next submatrix
    if num_clusters < 2:
        col_counts = np.asarray(np.sum(data, axis=0)).reshape(-1)
        sel_threshold = max(int(max(col_counts) / sel_factor), min(col_counts) + 1)
        addselcols = [colmap[col] for col in np.where(col_counts >= sel_threshold)[0]]

        selcols.extend(addselcols)
        data = data[:, np.where(col_counts < sel_threshold)[0]]
        nonzero = ~np.all(data == 0, axis=1)
        nonzero = np.asarray(nonzero).reshape(-1)
        data = data[nonzero]
        colmap = {k : colmap[v] for k, v in enumerate(np.where(col_counts < sel_threshold)[0].tolist())}

        if len(set(col_counts)) == 1:
            extractRec(data, colmap, absmap, abscolcand, abscols2, selcols, supersets, min(data.shape)-1, col_threshold, sel_factor, layer, allmin = True)
        else:
            extractRec(data, colmap, absmap, abscolcand, abscols2, selcols, supersets, min(data.shape)-1, col_threshold, sel_factor, layer)
        return 

    # if biclustering succeeds, recursively call extractRec on each biclusters.
    else:
#             if layer == 0:
#                 plot(model, data, DFA, layer)
        for j in range(num_clusters):
            data2 = model.get_submatrix(j, data)
            colmap2 = {k : colmap[v] for k, v in enumerate(model.get_indices(j)[1].tolist())}
            extractRec(data2, colmap2, absmap, abscolcand, abscols2, selcols, supersets, min(data2.shape)-1, col_threshold, sel_factor, layer + 1)
        return 

def extractCols(integerMatrix, initNumClusters, colmap, absColCand, colThreshold = 5, selFactor = 2):
    '''
        Main algorithm: cluster columns (supersets), extract columns for the next submatrix (selcols), and find absolute columns (absmap)
    '''
    if initNumClusters == None: 
        initNumClusters = min(integerMatrix.shape)-1

    if not integerMatrix.tolist():
        selcolList.append([])
        supersetsList.append([])
        print("Empty matrix. Skipped!")
        return [], [], defaultdict(list)

    selcols = []
    supersets = []
    abscols = []
    absmap = defaultdict(list)

    # recursive function of the algorithm
    extractRec(integerMatrix, colmap, absmap, absColCand, abscols, selcols, supersets, initNumClusters, colThreshold, selFactor, layer = 0)

    return supersets, selcols, absmap


def outputTransform(supersets, absmap, frozenMatrix, absThreshold = None):
    '''
        construct absolute column hierarchy, find absolute columns that needs unque coding for itself
        newsupersets is a list of tuples (variable columns, absolute columns)
        absHierarchy is a list of mixture of supersets and AbsNodes
    '''
    
    supersets1 = supersets
    absmap1 = absmap

    if len(absmap1) == 0:
        absHierarchy = copy.deepcopy(supersets1)
        newsupersets = [(superset, []) for superset in supersets1]
        return newsupersets, absHierarchy, []
    
    # construct absolute column hierarchy
    absmapTuples = sorted([(k, v) for k,v in absmap1.items()], key = lambda x: len(x[1]))
    absRoot = {}
    addselcols = []
    superset2absMap = defaultdict(list)
    for k, v in absmapTuples:
        newNode = AbsNode(k, v, list(absRoot.values()))
        if absThreshold == None or len(newNode) < absThreshold:
            absRoot[k] = newNode
            for node in newNode.absChildren:
                absRoot.pop(node.absCol)
            for superset in v:
                superset2absMap[superset].append(k)
        else:
            for superset in newNode.ownSupersets:
                supersets1.append(superset)
            addselcols.append(k)

    # find absolute columns that needs unque coding for itself
    if len(addselcols) > 0:
        frozenMatrix = set([frozenset(row.difference(addselcols)) for row in frozenMatrix])
    separatePrefix = []
    for node in absRoot.values():
        separatePrefix.extend(node.checkPrefix(frozenMatrix, []))
    #print("separatePrefix", separatePrefix)

    absHierarchy = copy.deepcopy(supersets1)
    absHierarchy.extend(absRoot.values())
    absHierarchy.append(frozenset("E"))

    newsupersets = [(superset, []) for superset in supersets1]
    newsupersets.extend([(k, v) for k,v in superset2absMap.items()])

    return newsupersets, absHierarchy, addselcols

def getCodingInformation(supersets, absHierarchy, selcols, oldColID2newColID, integerMatrix):

    if absHierarchy:
        supersetGroupings = sorted([len(superset) for superset in absHierarchy])
        tagwidth = bitsRequiredVariableID(absHierarchy)
    else:
        supersetGroupings = sorted([len(superset) for superset in supersets])
        tagwidth = bitsRequiredVariableID(supersets)

    numabscol = sum([rootNode.getAbsCount() for rootNode in absHierarchy if isinstance(rootNode, AbsNode)])


    info = "Superset groupings\n" + str(supersetGroupings) \
           + "\nTag width\n" + str(tagwidth) \
           + "\nSelected columns\n" + str(selcols) \
           + "\nCounts of Selected columns\n" + str([np.count_nonzero(integerMatrix[:, oldColID2newColID[col]]) for col in selcols]) \
           + "\nDoes absolute columns exit?\n" +  str(numabscol)

    if numabscol > 0:
        info += "\nAbsolute columns:\n" +  str(set.union(*[rootNode.getAbsCols() for rootNode in absHierarchy if isinstance(rootNode, AbsNode)]))
    info += "\n\n"
    return tagwidth, info


def biclusteringHierarchy(matrix, parameters):
    '''
        Main algorithm. Iterate the randomized algorithm to cluster submatrices to meet the goal of tag width.
    '''
    # warning message filtering
    warnings.simplefilter("ignore", category = ConvergenceWarning)
    warnings.filterwarnings("error", message = ".*divide by zero.*")

    # remove subsets and get the absolute column candidates
    dfaMatrix, absColCand = removeSubsetsGetAbsCandidate(matrix)

    # construct a bidict mapping
    allCols = set.union(*[set(row) for row in dfaMatrix])
    newColID2oldColID = {newCol : oldCol for newCol, oldCol in enumerate(allCols)}
    oldColID2newColID = {oldCol : newCol for newCol, oldCol in enumerate(allCols)}

    # construct integer matrix
    integerMatrix = np.matrix([[1 if col in row else 0 for col in allCols] for row in dfaMatrix], dtype=int)
    
    # parameters for the initial call of extractCols
    selFactor = 1.05                         # recursive factor to kick out heavy hitter columns preventing biclustering split;
                                            # must be > 1; smaller -> more conservative
    if parameters != None:
        colThreshold = parameters[0]        # threshold of bicluster size in #columns
        goal = int(parameters[1]) + 0.5     # prevent rounding errors
    else:
        colThreshold = 10
        goal = -1                           # if no goal is given, make the algorihm end in one iteration
    initNumClusters = 80

    widthsum = goal + 1
    newsupersetsList = []
    absHierarchyList = []
    counter = 1

    # iteration to reach the goal
    while widthsum >= goal:
        widthsum = 0
        widths = []
        infoList = ''
        colmap = copy.deepcopy(newColID2oldColID)

        # call the main agorithm to get supersets, selcols and absmap
        supersets, selcols, absmap = extractCols(integerMatrix, initNumClusters, colmap, absColCand, colThreshold, selFactor)
        # construct supersets and absHierarchy
        frozenMatrix = set([frozenset(row.difference(selcols)) for row in matrix])
        newsupersets, absHierarchy, addselcols = outputTransform(supersets, absmap, frozenMatrix, absThreshold = None)

        # save information; if additional columns are selected to the second submatrix, extend selcols 
        selcols.extend(addselcols)
        newsupersetsList = [newsupersets]
        absHierarchyList = [absHierarchy]

        # get width and message of the grouping
        width, info = getCodingInformation(newsupersetsList, absHierarchy, selcols, oldColID2newColID, integerMatrix)
        widthsum += width
        widths.append(width)
        infoList += info

        # parameters for the following rounds of submatrices
        if colThreshold == 13:
            subColThreshold = colThreshold
        else:
            subColThreshold = max(3, colThreshold // 2)

        # while there are still columns for the next submatrix
        while selcols:
            matrix2 = [set(row).intersection(selcols) for row in matrix]
            dfaMatrix2, absColCand2 = removeSubsetsGetAbsCandidate(matrix2)
            submatrix = np.matrix([[1 if col in row else 0 for col in selcols] for row in dfaMatrix2], dtype=int)

            subInitNumClusters = None
            colmap = {k : v for k, v in enumerate(selcols)}

            supersets, selcols, absmap = extractCols(submatrix, subInitNumClusters, colmap, absColCand2, subColThreshold, selFactor)
            frozenMatrix = set([frozenset(row.difference(selcols)) for row in matrix])
            newsupersets, absHierarchy, addselcols = outputTransform(supersets, absmap, frozenMatrix, absThreshold = subColThreshold)
            selcols.extend(addselcols)
            
            width, info = getCodingInformation(newsupersetsList, absHierarchy, selcols, oldColID2newColID, integerMatrix)
            widthsum += width
            widths.append(width)
            infoList += info
            
            newsupersetsList.append(newsupersets)
            absHierarchyList.append(absHierarchy)

        print("Trying ", counter, "th time, reaching width: ", widthsum, " (", str(widths), " ) goal: ", goal)
        # ends after the first iteraton if there is no goal
        if goal < 0:
            goal = widthsum + 1

        # ends after iterate for 20 times failing to reach the goal
        if counter > 20:
            raise Exception("Failing to meet the goal after ", counter, " trials!")
        counter += 1
    print(infoList)
    return newsupersetsList, absHierarchyList


def getGraphCodingInformation(absHierarchy, selCols, separatePrefix):

    supersetGroupings = sorted([len(superset) for superset in absHierarchy])
    tagwidth = bitsRequiredVariableID(absHierarchy)
    numabscol = sum([rootNode.getAbsCount() for rootNode in absHierarchy if isinstance(rootNode, AbsNode)])

    info = "Superset groupings\n" + str(supersetGroupings) \
           + "\nTag width\n" + str(tagwidth) \
           + "\nSelected columns\n" + str(selCols) \
           + "\nNumber of absolute columns\n" +  str(numabscol)

    if numabscol > 0:
        info += "\nAbsolute columns:\n" +  str(set.union(*[rootNode.getAbsCols() for rootNode in absHierarchy if isinstance(rootNode, AbsNode)]))
        info += "\nNumber of absolute columns with own encoding:\n" + str(len(separatePrefix))
        info += "\nAbsolute columns with own encoding:\n" + str(separatePrefix)
    info += "\n\n"
    return tagwidth, info


def graphOutputTransform(supersets, absRoots, frozenMatrix, absThreshold = None):
    '''
        construct absolute column hierarchy, find absolute columns that needs unque coding for itself
        newsupersets is a list of tuples (variable columns, absolute columns)
        absHierarchy is a list of mixture of supersets and AbsNodes
    '''
    # add unique encoding for absolute columns
    separatePrefix = []
    for absNode in absRoots:
        separatePrefix.extend(absNode.checkPrefix(frozenMatrix))

    # move absolute columns to the next submatrix if the size is above the absThreshold
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

    absHierarchy = copy.deepcopy(supersets)
    absHierarchy.extend(absRoots)
    absHierarchy.append(frozenset("E")) # empty code holder

    supersets = [(superset, []) for superset in supersets]
    for absNode in absRoots:
        supersets.extend(absNode.getSupersetPairs())

    return supersets, absHierarchy, addSelCols, separatePrefix


def extractGraphRec(graph, absRoots, absParent, selCols, supersets, threshold):

    if len(graph) == 0:
        print("WHY!!!!!!!")
        return 

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
        if absParent:
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
            if absParent:
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
        extractGraphRec(subgraph, absRoots, absParent, selCols, supersets, threshold)
    return


def extractGraphNodes(matrix, threshold = 10):
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

    extractGraphRec(graph, absRoots, absParent, selCols, supersets, threshold)

    return supersets, selCols, absRoots


def graphHierarchy(matrix, parameters):
    if parameters != None:
        threshold = parameters
    else:
        threshold = 10

    widthsum = 0
    widths = []
    infoList = []
    supersetsList = []
    absHierarchyList = []
    matrix2 = matrix

    while True:
        # call the main agorithm to get supersets, selCols and absRoots
        supersets, selCols, absRoots = extractGraphNodes(matrix2, threshold)
        # construct supersets and absHierarchy
        frozenMatrix = set([frozenset(row.difference(selCols)) for row in matrix])
        supersets, absHierarchy, addSelCols, separatePrefix = graphOutputTransform(supersets, absRoots, frozenMatrix, absThreshold = None)

        # save information; if additional columns are selected to the second submatrix, extend selcols 
        selCols.update(addSelCols)
        supersetsList.append(supersets)
        absHierarchyList.append(absHierarchy)

        # get width and message of the grouping
        width, info = getGraphCodingInformation(absHierarchy, selCols, separatePrefix)
        widthsum += width
        widths.append(width)
        infoList.append(info)

        if len(selCols) == 0:
            break
        else:
            matrix2 = [set(row).intersection(selCols) for row in matrix2]

    print("Reaching width: ", widthsum, " (", str(widths), " )")
    for info in infoList: print(info)
    return supersetsList, absHierarchyList


