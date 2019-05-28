from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category = ConvergenceWarning)
warnings.filterwarnings("error", message = ".*divide by zero.*")

from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import pickle
import numpy as np
import copy
import time
import sys

from numpy.random import RandomState


try:
    from .analyze import groupIdenticalColumns, bitsRequiredVariableID
except:
    from analyze import groupIdenticalColumns, bitsRequiredVariableID


RANSDOMSTATE = RandomState()

class AbsNode:
    ownSupersets = None # List
    fullSupersets = None # Set
    absNodes = None # List
    absCol : int = 0
    height : int = 0
    strict : bool = False

    def __init__(self, absCol, supersets, posssibleChildren, strict = True):
        self.absCol = absCol
        self.fullSupersets = set(supersets)
        self.absNodes = []
        self.strict = strict
        
        childSupersets = set([])
        for node in posssibleChildren:
            if node.fullSupersets.issubset(supersets):
                childSupersets.update(node.fullSupersets)
                self.absNodes.append(node)

        self.ownSupersets = list(self.fullSupersets.difference(childSupersets))

        self.height = bitsRequiredVariableID(self.getChildren())
    
    def getChildren(self):
        allItems = self.ownSupersets + self.absNodes
        if self.strict:
            allItems = allItems + [frozenset([self.absCol])]
        return sorted(allItems, key = lambda x: len(x))
    
    def __len__(self):
        return self.height

    def __str__(self):
        string = "absCol: " + str(self.absCol) + "\n"
        string +=  "absNodes: " + str([absNode.absCol for absNode in self.absNodes]) + "\n"
        string +=  "ownSupersets: " + str(self.ownSupersets) + "\n"
        return string
    def getAbsCount(self):
        return 1 + sum([node.getAbsCount() for node in self.absNodes])
        

def removeSubsetsGetAbsCandidate(setList):
    """ Removes all subsets from a list of sets and return absCol candidates.
    """
    absColCandMap = {}
    finalAnswer = []
    # defensive copy
    setList.sort(key=len, reverse=True)
    i = 0
    while i < len(setList):
        finalAnswer.append(setList[i])
        absColCandidate = setList[i]
        absColCandidate2 = setList[i]

        for j in reversed(range(i+1, len(setList))):
            if setList[j].issubset(setList[i]):
                absColCandidate2 = absColCandidate2.intersection(setList[j])
                del setList[j]

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


# Testing code of non-convergence
def good_form(model, data):
    data_count = np.count_nonzero(data)
    bicluster_count = 0
    
    for j in range(model.get_params()["n_clusters"]):
        bicluster_count += np.count_nonzero(model.get_submatrix(j, data))
    
    if bicluster_count != data_count:
        return False
    return True


# Loop for fitting to find n_svd_vecs
def fitLoop(model, data):
    i = 0
    j = 0
    while True:
        try:
            model.fit(data)
            break
        except:
            j += 1
            i = j // 3
            model.set_params(n_svd_vecs= min(min(data.shape), max(2*6 + 1, 20) + i))
            if j > 100:
                raise Exception("Fit fails with %d columns!" % (data.shape[1]))
                
    return model


# Plotting
def plot(model, data, layer):
    fit_data = data[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    plt.figure(figsize=(4,4))
    plt.spy(fit_data, markersize=0.4, aspect='auto')
    plt.title("Spectral CoClustering for Layer %d" % (layer))
    plt.show()


def extractRec(data, colmap, absmap, abscolcand, abscols, selcols, supersets, num_clusters, col_threshold, sel_factor, layer, final=False):    
    if data.shape[1] == 0:
        return

    abscols2 = copy.deepcopy(abscols)
    if data.shape[1] != 1:
        allone = np.all(data == 1, axis=0)
        allone = np.asarray(allone).reshape(-1)
        allone = np.array([False if colmap[k[0]] not in abscolcand else v for k, v in np.ndenumerate(allone)])
# Testing
#         for i in range(len(allone)):
#             if allone[i] != allone2[i]:
#                 print("allone", [colmap[col] for col in np.where(allone)[0]])
#                 print("allone2", [colmap[col] for col in np.where(allone2)[0]])
        temp = [colmap[col] for col in np.where(allone)[0]]

        abscols2.extend([colmap[col] for col in np.where(allone)[0]])
        notallone = ~allone
        data = data[:, notallone]
        notallone = np.asarray(notallone).reshape(-1)
        colmap = {k : colmap[v] for k, v in enumerate(np.where(notallone)[0])}
# Testing
#         if data.shape[1] == 0:
#             print("Alert!!!!!")
#             print(colmap)
    if final:
        if len(abscols2) != 0:
            for abscol in abscols2:
                absmap[abscol].append(frozenset(colmap.values()))
        else:
            supersets.append(frozenset(colmap.values()))
        return

    model = SpectralCoclustering(n_clusters=num_clusters, random_state=RANSDOMSTATE, svd_method='arpack')
    while num_clusters > 1:
        try:
            model = fitLoop(model, data)
        except Exception as e: 
            #print(e)
            if len(abscols2) != 0:
                for abscol in abscols2:
                    absmap[abscol].append(frozenset(colmap.values()))
            else:
                supersets.append(frozenset(colmap.values()))
            return

        if good_form(model, data):
            break

        if layer == 0:
            num_clusters = num_clusters - 1
        else:
            num_clusters = num_clusters // 2
        model.set_params(n_clusters=num_clusters)

    if data.shape[1] < col_threshold:
        if num_clusters < 2:
# Testing
#             if len(set(abscols2).intersection(set(colmap.values()))) != 0:
#                 print("1",abscols2, colmap.values())
            extractRec(data, colmap, absmap, abscolcand, abscols2, selcols, supersets, min(data.shape), col_threshold, sel_factor, layer + 1, final = True)
        else:
            for j in range(num_clusters):
                data2 = model.get_submatrix(j, data)
                colmap2 = {k : colmap[v] for k, v in enumerate(model.get_indices(j)[1].tolist())}
                extractRec(data2, colmap2, absmap, abscolcand, abscols2, selcols, supersets, min(data2.shape), col_threshold, sel_factor, layer + 1, final = True)
        return

    if num_clusters < 2:
        col_counts = np.asarray(np.sum(data, axis=0)).reshape(-1)
        sel_threshold = max(max(col_counts) // sel_factor, min(col_counts) + 1)

        selcols.extend([colmap[col] for col in np.where(col_counts >= sel_threshold)[0]])
        data = data[:, np.where(col_counts < sel_threshold)[0]]
        nonzero = ~np.all(data == 0, axis=1)
        nonzero = np.asarray(nonzero).reshape(-1)
        data = data[nonzero]
        colmap = {k : colmap[v] for k, v in enumerate(np.where(col_counts < sel_threshold)[0].tolist())}
        extractRec(data, colmap, absmap, abscolcand, abscols2, selcols, supersets, min(data.shape), col_threshold, sel_factor + 1, layer)
        return 
    else:
#             if layer == 0:
#                 plot(model, data, DFA, layer)
        for j in range(num_clusters):
            data2 = model.get_submatrix(j, data)
            colmap2 = {k : colmap[v] for k, v in enumerate(model.get_indices(j)[1].tolist())}
            extractRec(data2, colmap2, absmap, abscolcand, abscols2, selcols, supersets, min(data2.shape), col_threshold, sel_factor, layer + 1)
        return 

def extractCols(integerMatrix, initNumClusters, colmap, absColCand, colThreshold = 5, selFactor = 2):
    if initNumClusters == None: 
        initNumClusters = min(integerMatrix.shape)

    if not integerMatrix.tolist():
        selcolList.append([])
        supersetsList.append([])
        print("Empty matrix. Skipped!")
        return [], [], defaultdict(list)

    selcols = []
    supersets = []
    abscols = []
    absmap = defaultdict(list)

    extractRec(integerMatrix, colmap, absmap, absColCand, abscols, selcols, supersets, initNumClusters, colThreshold, selFactor, layer = 0)

    return supersets, selcols, absmap


def outputTransform(supersets, absmap, newColID2oldColID, strict, absThreshold = None):
    # Back to old ColID   
    supersets1 = supersets
    absmap1 = absmap
    # supersets1 = [frozenset([newColID2oldColID[col] for col in group]) for group in supersets]
    # absmap1 = {newColID2oldColID[k] : [frozenset([newColID2oldColID[col] for col in superset])
    #            for superset in v] for k, v in absmap.items()}

    if len(absmap1) == 0:
        absHierarchy = copy.deepcopy(supersets1)
        newsupersets = [(superset, []) for superset in supersets1]
        return newsupersets, absHierarchy, []
    
    absmapTuples = sorted([(k, v) for k,v in absmap1.items()], key = lambda x: len(x[1]))
    absRoot = {}
    addselcols = []
    superset2absMap = defaultdict(list)
    for k, v in absmapTuples:
        newNode = AbsNode(k, v, list(absRoot.values()), strict)
        if absThreshold == None or len(newNode) < absThreshold:
            absRoot[k] = newNode
            for node in newNode.absNodes:
                absRoot.pop(node.absCol)
            for superset in v:
                superset2absMap[superset].append(k)
        else:
            for superset in newNode.ownSupersets:
                supersets1.append(superset)
            addselcols.append(k)

    absHierarchy = copy.deepcopy(supersets1)
    absHierarchy.extend(absRoot.values())

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

    info = "Superset groupings\n" + str(supersetGroupings) \
           + "\nTag width\n" + str(tagwidth) \
           + "\nSelected columns\n" + str(selcols) \
           + "\nCounts of Selected columns\n" + str([np.count_nonzero(integerMatrix[:, oldColID2newColID[col]]) for col in selcols]) \
           + "\nDoes absolute columns exit?\n" +  str(sum([rootNode.getAbsCount() for rootNode in absHierarchy if isinstance(rootNode, AbsNode)]))
    return tagwidth, info


def biclusteringHierarchy(matrix, parameters, strict):

    warnings.simplefilter("ignore", category = ConvergenceWarning)
    warnings.filterwarnings("error", message = ".*divide by zero.*")

    matrix, absColCand = removeSubsetsGetAbsCandidate(matrix)

    allCols = set.union(*[set(row) for row in matrix])
    newColID2oldColID = {newCol : oldCol for newCol, oldCol in enumerate(allCols)}
    oldColID2newColID = {oldCol : newCol for newCol, oldCol in enumerate(allCols)}

    integerMatrix = np.matrix([[1 if col in row else 0 for col in allCols] for row in matrix], dtype=int)

    selFactor = 2
    if parameters != None:
        colThreshold = parameters[0]
        goal = int(parameters[1]) + 0.5
    else:
        colThreshold = 10
        goal = -1
    initNumClusters = 80

    widthsum = goal + 1
    newsupersetsList = []
    absHierarchyList = []
    counter = 1

    while widthsum >= goal:
        widthsum = 0
        infoList = ''
        colmap = copy.deepcopy(newColID2oldColID)

        supersets, selcols, absmap = extractCols(integerMatrix, initNumClusters, colmap, absColCand, colThreshold, selFactor)
        newsupersets, absHierarchy, addselcols = outputTransform(supersets, absmap, newColID2oldColID, strict, absThreshold = None)

        selcols.extend(addselcols)
        newsupersetsList = [newsupersets]
        absHierarchyList = [absHierarchy]

        width, info = getCodingInformation(newsupersetsList, absHierarchy, selcols, oldColID2newColID, integerMatrix)
        widthsum += width
        infoList += info

        subColThreshold = max(3, colThreshold // 2)
        while selcols:
            
            subInitNumClusters = None
            colmap = {k : v for k, v in enumerate(selcols)}
            selcolsnewindices = [oldColID2newColID[col] for col in selcols]

            submatrix = integerMatrix[:, selcolsnewindices]
            nonzero = ~np.all(submatrix == 0, axis=1)
            nonzero = np.asarray(nonzero).reshape(-1)
            submatrix = submatrix[nonzero]
            
            supersets, selcols, absmap = extractCols(submatrix, subInitNumClusters, colmap, absColCand, subColThreshold, selFactor)
            newsupersets, absHierarchy, _ = outputTransform(supersets, absmap, newColID2oldColID, strict, absThreshold = None)

            width, info = getCodingInformation(newsupersetsList, absHierarchy, selcols, oldColID2newColID, integerMatrix)
            widthsum += width
            infoList += info

            newsupersetsList.append(newsupersets)
            absHierarchyList.append(absHierarchy)

        print("Trying ", counter, "th time, reaching width: ", widthsum, " goal: ", goal)
        if goal < 0:
            goal = widthsum - 1
        if counter > 20:
            raise Exception("Failing to meet the goal after ", counter, " trials!")
        counter += 1
    print(infoList)
    return newsupersetsList, absHierarchyList

# TODO
#absmap.keys()


