#!/usr/bin/env python3.7
import argparse, os, sys
import logging
from MRSets import MRCode
import pickle, json
import time
import random
from collections import Counter, defaultdict
from analyze import kraftsBound, transposeMatrix, groupOverlappingRows
from util import printShellDivider, getShellWidth, shellHistogram
from ClusterCodes import OriginalCodeStatic
import multiprocessing
from typing import Callable, List
import itertools
from RSets import RCode
import numpy as np

""" 
    Need to write prerequisite installation script.

    Distribution of number of ASes per prefix
    Run algorithm multiple times with parameters: 
        - Change the percentage of the matrix for which we will run the encoding
        - Max cluster size threshold
        - Disable "approximation" on min cut algorithm, check if code width changes
    Per Code output:
        - Number of absolutes per clusters avg
        - Absolute hierarchy?
        - Get partition sizes, stats for each subcode
        - Running time


    Bonus Info (if we have time to write it):
        - Get min-cut sizes for each extraction iteration
        - Churn (havent written at all yet)
"""


def get_args():
    parser = argparse.ArgumentParser(description="MEME Evaluation Script")
    parser.add_argument('matrix_pickle', default='route_matrix.pickle', 
                        help='Pickle file that contains an attribute matrix')
    parser.add_argument('-o', '--outfile', default=None,
                        help='Destination file to which evaluation results will be written. If no file is given, stdout is used.')
    return parser.parse_args()


# Not used
def deduplicateMatrix(matrix):
    # make the matrix rows hashable
    matrix = [frozenset(row) for row in matrix]
    # count row occurrences
    return [(list(row), count) for row,count in Counter(matrix).items()]



def reduplicateMatrix(matrixWithCounts):
    matrix = []
    for row, count in matrixWithCounts:
        matrix.extend([set(row)]*count)
    return matrix


def allPairs(list1, list2):
    return [(item1, item2) for item1 in list1 for item2 in list2]


def getMatrixStatistics(matrixWithCounts, **extraInfo):
    matrix = reduplicateMatrix(matrixWithCounts)
    logger = logging.getLogger("eval.matrixStats")
    allCols = set()
    for row, count in matrixWithCounts:
        allCols.update(row)

    width = len(allCols)
    height = sum([count for row, count in matrixWithCounts])
    numDistinctRows = len(matrixWithCounts)

    rowSizes = []
    for row, count in matrixWithCounts:
        rowSizes.extend([len(row)]*count)

    avgRowSizeDup = sum(rowSizes) / len(rowSizes)
    densityDup = avgRowSizeDup / width

    avgRowSizeNoDup = sum([len(row) for row, count in matrixWithCounts]) / len(matrixWithCounts)
    densityNoDup = avgRowSizeNoDup / width

    info = dict(extraInfo)
    info["width"] = width
    info["height"] = height
    info["distinct rows"] = numDistinctRows
    info["avg row size dup"] = avgRowSizeDup
    info["avg row size no dup"] = avgRowSizeNoDup
    info["density dup"] = densityDup
    info["density no dup"] = densityNoDup
    info["max row size"] = max(rowSizes)

    info["row size counts"] = dict(Counter([len(row) for row in matrix]))

    tmatrix = transposeMatrix(matrix)
    #info["col size counts"] = dict(Counter([len(row) for row in tmatrix]))

    clusters = groupOverlappingRows(matrix, asRows=False)
    info["cluster size counts"] = dict(Counter([len(cluster) for cluster in clusters]))

    maxcluster = max(clusters, key = lambda x:len(x))
    numRowsBridged = 0
    numRowsBridgedDup = 0
    for row, count in matrixWithCounts:
        if len(set(row).intersection(maxcluster)) != 0:
            numRowsBridgedDup += count
            numRowsBridged += 1
    info["num rows in max cluster dup"] = numRowsBridgedDup
    info["num rows in max cluster no dup"] = numRowsBridged
    return info



def randomSubmatrix(matrixWithCounts, allCols = None, percent=0.10):
    """ Return a submatrix made of a random subset of the columns.
    """
    if allCols == None:
        # genrate allCols from scratch
        allCols = set()
        for row, count in matrixWithCounts:
            allCols.update(row)

    allCols = list(allCols)
    random.shuffle(allCols)

    subset = set(allCols[:int(len(allCols) * percent)])

    submatrix = [subset.intersection(row) for row, count in matrixWithCounts]
    
    return submatrix



def plotRowSizeDistribution(matrixWithCounts):
    rowSizes = []
    for row, count in matrixWithCounts:
        rowSizes.extend([len(row)]*count)

    shellHistogram(rowSizes, title="Distribution of row sizes")



def onlyDensestColumns(matrixWithCounts, frac : float, rev = True):
    """ Return a submatrix which contains only the densest 'frac' of the columns.
    """
    counts = defaultdict(int)

    for row, count in matrixWithCounts:
        for col in row:
            counts[col] += count


    allCols = list(counts.keys())
    allCols.sort(reverse=rev, key=lambda x:counts[x])

    k = int(len(allCols) * frac)

    kDensest = set(allCols[:k])

    kDensestRows = [(frozenset(kDensest.intersection(row)), count) for row, count in matrixWithCounts]
    results = {}
    for row, count in kDensestRows:
        if row in results:
            results[row] += count
        else:
            results[row] = count
    return [(set(row), count) for row, count in results.items()]



def parallelTrials(matrix, 
                  submatrixEvaluator : Callable, # function that evaluates a code on a given submatrix
                  submatrixEvaluationParams : List[List] = [[]], # list of parameter lists fo evaluating submatrices
                  submatrixProducer : Callable = onlyDensestColumns, # function that produces a submatrix from a given matrix
                  submatrixProductionParams : List[List] = [[]], # list of parameter lists for producing submatrices
                   parallel = True):
    """ Run evaluation experiments in parallel. Runs every combination of matrix evaluation parameters and production parameters.
    """

    # produce submatrices using the given production function and the given list of production parameters
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        submatrices = p.starmap(submatrixProducer, [[matrix] + paramList for paramList in submatrixProductionParams])

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p: 
        submatrixInfos = p.map(getMatrixStatistics, submatrices)

    for i, submatrixInfo in enumerate(submatrixInfos):
        submatrixInfo["Matrix ID"] = i
        submatrixInfo["ParamList"] = submatrixProductionParams[i]

    # now that we've gotten submatrix infos, the row duplicate counts are unneeded
    submatrices = [[row for row,count in submatrix] for submatrix in submatrices]
    # take cross product of evaluation parameters and submatrices (evaluate every parameter list on every submatrix)
    paramPairs = allPairs(range(len(submatrices)), submatrixEvaluationParams)
    newEvalParams = [[submatrices[pair[0]]] + pair[1] for pair in paramPairs]
    matrixIndices = [pair[0] for pair in paramPairs]

    print("Evaluation starts...")
    # evaluate an encoding on each submatrix using the given evaluation function and the given list of evaluation parameters
    if parallel:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            evalResults = p.starmap(submatrixEvaluator, newEvalParams)
    else:
        evalResults = [submatrixEvaluator(*paramSet) for paramSet in newEvalParams]

    for i, matrixIndex in enumerate(matrixIndices):
        evalResults[i]['Matrix ID'] = matrixIndex

    # return the list of results and matrix descriptions
    return submatrixInfos, evalResults



def evaluatePathSetsSingle(matrix, **extraInfo):
    """ Evaluate a single instance of a pathsets encoding on the given matrix
    """

    optMerging = False # merge any overlapping clusters
    optimizeInitialWidth = True
    # if maxWidth is -1, pathsets greedily minimizes memory instead of number of match strings
    maxWidth = -1

    code = OriginalCodeStatic(matrix)
    startTime = time.time()    
    code.make(optWidth=optimizeInitialWidth,
                maxWidth = maxWidth,
                mergeOverlaps=optMerging)
    runningTime = time.time() - startTime

    width = code.width()
    memory = code.memoryRequired()[1]

    numMatchStrings = code.numMatchStrings()
    tagWidth = code.width()

    info = dict(extraInfo)
    info['maxWidth'] = maxWidth
    info['optMerging'] = optMerging
    info['Running time'] = runningTime
    info['Number of match strings'] = numMatchStrings
    info['Tag width'] = width
    info['Total memory'] = numMatchStrings * tagWidth

    return info

def evaluatePathSetsPISASingle(matrix, numTable=5, **extraInfo):
    """ Evaluate a single instance of a pathsets encoding on the given matrix                                                    
    """

    optMerging = False # merge any overlapping clusters                                                                          
    optimizeInitialWidth = True
    # if maxWidth is -1, pathsets greedily minimizes memory instead of number of match strings
    maxWidth = -1
    info = dict(extraInfo)
    info['maxWidth'] = maxWidth
    info['optMerging'] = optMerging
    info['Running time'] = 0
    info['Number of match strings'] = 0
    info['Tag width'] = []
    info['Total memory'] = 0

    allCols = set()
    for row in matrix:
        allCols.update(row)
    allCols = list(allCols) 
    random.shuffle(allCols)

    partitionSize = int(len(allCols)/numTable)
    for i in range(numTable):
        start = i * partitionSize
        if i == numTable-1:
            end = len(allCols)+1
        else:
            end = (i+1) * partitionSize
        subset = set(allCols[start:end])
        submatrix = [subset.intersection(row) for row in matrix]

        code = OriginalCodeStatic(submatrix)
        startTime = time.time()
        code.make(optWidth=optimizeInitialWidth,
                  maxWidth = maxWidth,
                  mergeOverlaps=optMerging)
        runningTime = time.time() - startTime

        width = code.width()
        memory = code.memoryRequired()[1]

        numMatchStrings = code.numMatchStrings()
        tagWidth = code.width()

        info['Running time'] += runningTime
        info['Number of match strings'] += numMatchStrings
        info['Tag width'].append(width)
        info['Total memory'] += numMatchStrings * tagWidth

    return info



def evaluatePathSetsParallel(matrix):
    # generate a bunch of submatrices and evaluate pathsets on all those submatrices
    policyPercentages = [[0.05*i] for i in range(1, 21)]

    return parallelTrials(matrix = matrix, 
                          submatrixEvaluator = evaluatePathSetsSingle, 
                          submatrixProducer = onlyDensestColumns, 
                          submatrixProductionParams = policyPercentages) 


def evaluateMemeParallel(matrix, minThreshold=5, maxThreshold=6):
    # generate a bunch of 
    thresholds = [[threshold] for threshold in range(minThreshold, maxThreshold)]
    policyPercentages = [[0.05*i] for i in range(1, 21)]

    return parallelTrials(matrix=matrix,
                          submatrixEvaluator = evaluateMemeSingle,
                          submatrixEvaluationParams = thresholds,
                          submatrixProducer = onlyDensestColumns,
                          submatrixProductionParams = policyPercentages)


def evaluatePathSetsPISAParallel(matrix, minTable = 3, maxTable=12):
    # generate a bunch of
    numtables = [[numTable] for numTable in range(minTable, maxTable)]
    policyPercentages = [[0.05*i] for i in range(1, 21)]

    return parallelTrials(matrix=matrix,
                          submatrixEvaluator = evaluatePathSetsPISASingle,
                          submatrixEvaluationParams = numtables,
                          submatrixProducer = onlyDensestColumns,
                          submatrixProductionParams = policyPercentages)

def evaluateMemeUpdateParallel(matrix, minThreshold=5, maxThreshold=6):
    # generate a bunch of 
    thresholds = [[threshold] for threshold in range(minThreshold, maxThreshold)]
    policyPercentages = [[0.05*i] for i in range(20, 21)]

    return parallelTrials(matrix=matrix,
                          submatrixEvaluator = evaluateMemeUpdateSingle,
                          submatrixEvaluationParams = thresholds,
                          submatrixProducer = onlyDensestColumns,
                          submatrixProductionParams = policyPercentages)



def evaluateMemeSingle(matrix, threshold=4, **extraInfo):
    # hierarchy = True makes the MRCode uses biclutering hierarchy algorithm
    mrcode = MRCode(matrix, hierarchy = True)
    # parameters is curretly a tuple of (Threshold of bicluster size, Goal of tag width)

    startTime = time.time()
    mrcode.optimize(parameters = (threshold, None))
    runningTime = time.time() - startTime
    mrcode.verifyCompression()

    codeWidths = [code.widthUsed() for code in mrcode.rcodes]
    numMatchStrings = len(mrcode.matchStrings())

    info = dict(extraInfo)
    info["threshold"] = threshold
    info["Num shadow Elements"] = len(mrcode.shadowElements.keys())
    info["Number of match strings"] = numMatchStrings
    info["Running time"] = runningTime
    info["Subcode widths"] = sorted(codeWidths)
    info["Tag width"] = sum(codeWidths)
    info["Shadow count"] = len(set(mrcode.shadowElements.keys()))
    info["Total memory"] = sum(codeWidths) * numMatchStrings
    info["Total memory(PISA)"] = sum([rcode.widthUsed()*len(rcode.elements) for rcode in mrcode.rcodes])

    return info

def evaluateMemeUpdateSingle(matrix, threshold=4, **extraInfo):
    allCols = set()
    for row in matrix:
        allCols.update(row)
    print(allCols)

    # hierarchy = True makes the MRCode uses biclutering hierarchy algorithm
    # Disabling siblings
    # extra 1 bit in each subtag
    mrcode = MRCode(matrix, hierarchy = True, shadow = False, extraBits = 1)
    # parameters is curretly a tuple of (Threshold of bicluster size, Goal of tag width, Disabling ancestor)    
    mrcode.optimize(parameters = (threshold, None, False))

    print("Start adding....")
    runningTime = []
    tags = []
    for i in range(300):
        startTime = time.time()
        newSS = random.sample(matrix, k=1)[0]
        newSS.update(random.sample(allCols, k=1))
        tag = mrcode.addSuperset(newSS)
        # mrcode.addSuperset(random.sample(allCols, k=10))
        runningTime.append(time.time() - startTime)
        tags.append(tag)
        print(i, tag, time.time() - startTime)
        mrcode.verifyCompression()
    runningTime = np.asarray(runningTime)
    tags = np.asarray(tags)

    print("Actual Encoding", (tags == 1).sum(), np.average(runningTime[tags == 1]))
    print("No Encoding", (tags == 0).sum(), np.average(runningTime[tags == 0]))
    print("Both", tags.shape[0], np.average(runningTime))

    info = dict(extraInfo)
    info["Running time"] = runningTime
    return info



def main():
    args = get_args()

    print("Using matrix pickle filename:", args.matrix_pickle)
    if args.outfile != None:
        print("Evaluation results will be written to", args.outfile)
    else:
        print("No log file given. Evaluation results going to stdout.")

    print("Loading matrix from file.")
    with open(args.matrix_pickle, 'rb') as fp:
        matrixWithCounts = pickle.load(fp)
        if type(matrixWithCounts) is set:
            matrixWithCounts = [(row, 1) for row in matrixWithCounts]
    print("Done loading")

    print("Getting overall matrix stats")
    plotRowSizeDistribution(matrixWithCounts)
    getMatrixStatistics(matrixWithCounts, matrixName="FullMatrix")
    print("Done.")

    resultDict = {}

    # # Meme
    # submatrixInfos, evalResults = evaluateMemeParallel(matrixWithCounts)
    # printShellDivider("Submatrix Properties")
    # for item in submatrixInfos:
    #     print(item)
    # printShellDivider("Meme Eval Results")
    # for item in evalResults:
    #     print(item)

    # resultDict['MemeMatrixInfos'] = submatrixInfos
    # resultDict['MemeEvalResults'] = evalResults

    # Meme update
    evaluateMemeUpdateParallel(matrixWithCounts)

    # # PathSets PISA
    # submatrixInfos, evalResults = evaluatePathSetsPISAParallel(matrixWithCounts)
    # printShellDivider("Submatrix Properties")
    # for item in submatrixInfos:
    #     print(item)
    # printShellDivider("PathSets PISA Eval Results")
    # for item in evalResults:
    #     print(item)

    # resultDict['PathSetsPISAMatrixInfos'] = submatrixInfos
    # resultDict['PathSetsPISAEvalResults'] = evalResults

    
    # # PathSets
    # submatrixInfos, evalResults = evaluatePathSetsParallel(matrixWithCounts)
    # printShellDivider("Submatrix Properties")
    # for item in submatrixInfos:
    #     print(item)
    # printShellDivider("PathSets Eval Results")
    # for item in evalResults:
    #     print(item)

    # resultDict['PathSetsMatrixInfos'] = submatrixInfos
    # resultDict['PathSetsEvalResults'] = evalResults

    print("Writing results to pickle")
    with open(args.outfile, 'wb') as fp:
        pickle.dump(resultDict, fp)
    print("Done.")


if __name__ == "__main__":
    main()






