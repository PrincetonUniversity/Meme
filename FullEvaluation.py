#!/usr/bin/env python3.7
import argparse, os, sys
import logging
from MRSets import MRCode
import pickle, json
import time
import random
from collections import Counter
from analyze import kraftsBound, transposeMatrix, groupOverlappingRows
from util import printShellDivider, getShellWidth, shellHistogram

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

def reduplicateMatrix(matrixWithCounts):
    matrix = []
    for row, count in matrixWithCounts:
        matrix.extend([set(row)]*count)
    return matrix


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

    avgRowSize = sum(rowSizes) / len(rowSizes)

    density = avgRowSize / width


    info = dict(extraInfo)
    info["width"] = width
    info["height"] = height
    info["distinct rows"] = numDistinctRows
    info["avg row size"] = avgRowSize
    info["density"] = density
    info["max row size"] = max(rowSizes)

    info["row size counts"] = dict(Counter([len(row) for row in matrix]))
    plotRowSizeDistribution(matrixWithCounts)

    tmatrix = transposeMatrix(matrix)
    #info["col size counts"] = dict(Counter([len(row) for row in tmatrix]))

    clusters = groupOverlappingRows(matrix, asRows=False)
    info["cluster size counts"] = dict(Counter([len(cluster) for cluster in clusters]))

    logger.info(json.dumps(info))



def randomSubmatrix(matrix, allCols = None, percent=0.10):
    """ Return a submatrix made of a random subset of the columns.
    """
    if allCols == None:
        # genrate allCols from scratch
        allCols = set()
        for row in matrix:
            allCols.update(row)

    allCols = list(allCols)
    random.shuffle(allCols)

    subset = set(allCols[:int(len(allCols) * percent)])

    submatrix = [subset.intersection(row) for row in matrix]
    
    return submatrix


def plotRowSizeDistribution(matrixWithCounts):
    rowSizes = []
    for row, count in matrixWithCounts:
        rowSizes.extend([len(row)]*count)

    shellHistogram(rowSizes, title="Distribution of row sizes")


def getCodeStatistics(supersets, **extraInfo):
    logger = logging.getLogger("eval.codeStats")

    #logger.info("Total num groups" + str(len(set(supersets))))
    originalElements = list(set.union(*[set(superset) for superset in supersets]))
    originalSets = [frozenset(superset) for superset in supersets]
    #logger.info("Total num elements" + str(len(originalElements)))

    # orderedSets = [supersets[i*2 + 1] for i in range(len(supersets)/2)]
    # maxElement = max(max(superset) for superset in orderedSets)
    # ordering = {i:i for i in range(maxElement+1)}
    # rcode = RCode(orderedSets)
    # rcode.optimizeWidth()

    # hierarchy = True makes the MRCode uses biclutering hierarchy algorithm
    mrcode = MRCode(originalSets, hierarchy = True)
    # parameters is curretly a tuple of (Threshold of bicluster size, Goal of tag width)

    minThreshold = 4
    maxThreshold = 15
    for threshold in range(minThreshold, maxThreshold+1):
        print("Iteration %d of %d.." % (threshold - minThreshold + 1, maxThreshold-minThreshold+1))
        info = dict(extraInfo)
        info["threshold"] = threshold
        cur_time = time.time()
        mrcode.optimize(parameters = (threshold, None))
        running_time = time.time() - cur_time
        info["Shadow Elements"] = len(mrcode.shadowElements.keys())
        mrcode.verifyCompression()
        totalMemory = [len(rule) for rules in mrcode.matchStrings().values() for rule in rules]
        info["Total number of rules"] = len(totalMemory)
        info["Length of rule"] = totalMemory[0]
        info["Total memory"] = sum(totalMemory)
        info["Running time"] = running_time
        info["subcode widths"] = [rcode.widthUsed() for rcode in mrcode.rcodes]
        logger.info(json.dumps(info))



    # print("Tag width: ", len(mrcode.tagString(frozenset([]))))
    # print("Tag width: ", len(mrcode.matchStrings(frozenset(['AS8283']))['AS8283'][0]))
    # print("Tag width: ", mrcode.matchStrings(frozenset(['AS8283']))['AS8283'][0])


def matrixTrials(matrix, numTrials, percents):
    allCols = set()
    for row in matrix:
        allCols.update(row)
    for p, percent in enumerate(percents):
        print("Percent %d of %d" % (p+1, len(percents)))
        for trial in range(numTrials):
            print("Trial %d of %d" % (trial+1, numTrials))
            submatrix = randomSubmatrix(matrix, allCols=allCols, percent=percent)
            #getMatrixStatistics(submatrix, trialNum=trial, percent=percent)
            getCodeStatistics(submatrix, trialNum=trial, percent=percent)
    


def main():
    args = get_args()

    print("Using matrix pickle filename:", args.matrix_pickle)
    if args.outfile != None:
        print("Evaluation results will be written to", args.outfile)
        logging.basicConfig(filename=args.outfile, level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.DEBUG)
        print("No log file given. Evaluation results going to stdout.")

    print("Loading matrix from file.")
    with open(args.matrix_pickle, 'rb') as fp:
        matrixWithCounts = pickle.load(fp)
    print("Done loading")

    print("Getting overall matrix stats")
    getMatrixStatistics(matrixWithCounts, matrixName="FullMatrix")
    print("Done.")
  
    matrix = [row for row,count in matrixWithCounts]

    #getMatrixStatistics(matrix, matrixName="deduplicated")

    matrixTrials(matrix, numTrials=1, percents=[1.0])


if __name__ == "__main__":
    main()






