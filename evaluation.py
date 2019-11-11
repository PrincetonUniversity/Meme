import os, sys
import json
import time
import pickle
try:
    from .MRSets import MRCode
except:
    from MRSets import MRCode

REPO_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(REPO_PATH, "data/")

def compress(filename):
    filepath = os.path.join(DATA_PATH, filename)

    with open(filepath, 'rb') as fp:
        matrixRowCounts = pickle.load(fp)

    print(matrixRowCounts)
    allCols = set()
    numRowsTotal = 0
    numEntries = 0
    for row, count in matrixRowCounts:
        allCols.update(row)
        numRowsTotal += count
        numEntries += (len(row) * count)

    allCols = list(allCols)
    allCols.sort()

    avgRowSize = numEntries / numRowsTotal
    density = numEntries / (len(allCols) * numRowsTotal)

    matrixInfo = {".Note":"Although the matrix is binary, it is represented as rows of column IDs to save memory because BGP route matrices are typically extremely sparse. The column IDs denote non-zero positions.",
                  ".Num Unique Rows": len(matrixRowCounts),
                  ".Num Total Rows (prefixes)": numRowsTotal,
                  ".Num Columns (peers)": len(allCols),
                  ".Density (between 0 and 1)" : density,
                  ".Average row size" : avgRowSize,
                  "Columns (this should just be an contiguous integer range due to anonymization)" : str(allCols)}

    print(matrixInfo)

    count = {}
    accum = [(len(row),co) for row, co in matrixRowCounts]
    for row, co in accum:
        if row in count:
            count[row] += co
        else:
            count[row] = co
    print(count)

    count = {}
    for col in allCols:
        for row,co in matrixRowCounts:
            if col in row:
                if col in count:
                    count[col] += co
                else:
                    count[col] = co

    count = sorted([(col, co) for col, co in count.items()], key = lambda x:x[1], reverse = True)
    selection = [col for col, co in count]
    supersets = [frozenset(set(row).intersection(selection)) for row, count in matrixRowCounts]


    print("Total number of groups: ", len(set(supersets)))
    originalElements = list(set.union(*[set(superset) for superset in supersets]))
    originalSets = [frozenset(superset) for superset in supersets]
    print("Total number of elements: ", len(originalElements))

    # Hierarchy
    # setList = [set([i for i, ori in enumerate(originalSets) if elem in ori]) for elem in originalElements]
    # finalAnswer = []                        # matrix after subset merging
    # setList.sort(key=len, reverse=True)
    # i = 0

    # while i < len(setList):
    #     if len(setList[i]) != 0:
    #         finalAnswer.append(setList[i])
    #     for j in reversed(range(i+1, len(setList))):
    #         if setList[j].issubset(setList[i]):
    #             print(len(setList[j]), len(setList[i]))
    #     i += 1


    # hierarchy = True makes the MRCode uses biclutering hierarchy algorithm
    mrcode = MRCode(originalSets, hierarchy = True)
    # parameters is curretly a tuple of (Threshold of bicluster size, Goal of tag width)

    threshold = 4
    while threshold < 15:
        cur_time = time.time()
        mrcode.optimize(parameters = (threshold, None))
        print("Shadow Elements:", len(mrcode.shadowElements.keys()))
        #mrcode.verifyCompression()
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



if __name__ == '__main__':
    compress("route_matrix_small.pickle")
