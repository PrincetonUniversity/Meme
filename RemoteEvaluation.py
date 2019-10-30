#!/usr/bin/env python3.7
from parse_mrt import mrtCsvToMatrix
import argparse, os, sys
import logging
from MRSets import MRCode
import pickle
import time


logging.basicConfig(level=logging.DEBUG)

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
    parser.add_argument('-m', '--matrix-pickle', default='announcement_matrix.pickle', 
                        help='Pickle file that contains an attribute matrix')
    parser.add_argument('-i', '---mrt-csv', default=None,
                        help='CSV that contains RIBs in MRT format')
    parser.add_argument('-o', '--outfile', default='meme_results.pickle',
                        help='Destination pickle  to which evaluation results will be written')
    return parser.parse_args()




def getCodeStatistics(supersets):

    print("Total number of groups: ", len(set(supersets)))
    originalElements = list(set.union(*[set(superset) for superset in supersets]))
    originalSets = [frozenset(superset) for superset in supersets]
    print("Total number of elements: ", len(originalElements))

    # orderedSets = [supersets[i*2 + 1] for i in range(len(supersets)/2)]
    # maxElement = max(max(superset) for superset in orderedSets)
    # ordering = {i:i for i in range(maxElement+1)}
    # rcode = RCode(orderedSets)
    # rcode.optimizeWidth()

    # hierarchy = True makes the MRCode uses biclutering hierarchy algorithm
    mrcode = MRCode(originalSets, hierarchy = True)
    # parameters is curretly a tuple of (Threshold of bicluster size, Goal of tag width)

    threshold = 4
    while threshold < 15:
        cur_time = time.time()
        mrcode.optimize(parameters = (threshold, None))
        print("Shadow Elements:", len(mrcode.shadowElements.keys()))
        mrcode.verifyCompression()
        totalMemory = [len(rule) for rules in mrcode.matchStrings().values() for rule in rules]
        print("Chosen threshold: ", threshold)
        print("Total number of rules: ", len(totalMemory))
        print("Length of rule:", totalMemory[0])
        print("Total memory:", sum(totalMemory))
        threshold += 1
        print("Running time:", time.time() - cur_time, "seconds")

    # print("Tag width: ", len(mrcode.tagString(frozenset([]))))
    # print("Tag width: ", len(mrcode.matchStrings(frozenset(['AS8283']))['AS8283'][0]))
    # print("Tag width: ", mrcode.matchStrings(frozenset(['AS8283']))['AS8283'][0])



def main():
    args = get_args()

    needMatrix = False
    if (args.mrt_csv == None) or (not os.path.exists(args.mrt_csv)):
        if os.path.exists(args.matrix_pickle):
            print("No MRT file given. Using existing matrix file.")
        else:
            print("ERROR: MRT and matrix files could not be found! Run script with -h for usage info.")
            exit(1)
    elif not os.path.exists(args.matrix_pickle):
        print("Matrix file not found.")
        needMatrix = True
    else:
        mrt_modtime = os.path.getmtime(args.mrt_csv)
        matrix_modtime = os.path.getmtime(args.matrix_pickle)

        if mrt_modtime > matrix_modtime:
            print("Given MRT file newer than matrix file.")
            needMatrix = True

    if needMatrix:
        print("Generating matrix from given MRT file...")
        matrix = mrtCsvToMatrix(args.mrt_csv, args.matrix_pickle)
        print("Done.")
    else:
        print("Loading matrix from", args.matrix_pickle)
        with open(args.matrix_pickle, 'rb') as fp:
            matrix = pickle.load(fp)
        print("Done loading")
    
    getCodeStatistics(matrix)


if __name__ == "__main__":
    main()






