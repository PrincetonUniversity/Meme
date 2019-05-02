import copy
from collections import Counter
from .analyze import bitsRequiredVariableID
from .optimize import removeSubsets
from bisect import bisect_left
import numpy as np


# Convert a list of supersets to a matrix of the number of tags
def supersets2Matrix(supersetsList1, supersetsList2 = None, colSet = None):
    if bool(supersetsList2) != bool(colSet):
        raise Exception("Expected Usage:\n  supersets2Matrix [supersetsList1] for initial tagNumMatrix\n  supersets2Matrix [supersetsList1] [supersetsList2] [colSet] in the grouping loop")

    # flatten a list of lists of supersets into a list of flat lists
    flatlist = [[item for superset in supersets for item in superset] for supersets in supersetsList1]
    numDFAs = len(flatlist)

    # get and check the set of columns are equivalent across all subDFAs
    if not colSet:
        colSet = set(flatlist[0])
        for l in flatlist:
            if colSet != set(l):
                raise Exception("Not all subDFAs have the same initial set of columns!")

    # count the distribution of columns in each sub-DFA
    counts = [Counter(l) for l in flatlist]
    # prepend the column ID
    tagNumMatrix = [[col] + [counts[i][col]  if col in counts[i] else 1 for i in range(numDFAs)] for col in colSet]
    # append the total number of rules
    tagNumMatrix = [l + [np.prod(l[1 : ])] for l in tagNumMatrix]
    tagNumMatrix.sort(key = lambda l : l[numDFAs + 1], reverse =True)

    if supersetsList2:
        # merge the flatlist from supersetsList2
        flatlist = [flatlist[i] + [item for superset in supersetsList2[i] for item in superset] for i in range(numDFAs)]
        counts = [Counter(l) for l in flatlist]
        numTags = np.sum(np.prod([[counts[i][col] for i in range(numDFAs)] for col in colSet], axis = 1))
    else:
        numTags = np.sum([l[numDFAs + 1] for l in tagNumMatrix])

    return tagNumMatrix, numTags

# objective function to optimize over, which is the total memory in bits here
def objectiveFunction(supersetsList1, supersetsList2 = None, colSet = None):
    # remove subsets
    supersetsList1rmss = [removeSubsets(supersets) for supersets in supersetsList1]
    # calculate the tag width
    widthTag1 = [bitsRequiredVariableID(supersets) for supersets in supersetsList1rmss]

    if supersetsList2:
        supersetsList2rmss = [removeSubsets(supersets) for supersets in supersetsList2]
        widthTag2 = [bitsRequiredVariableID(supersets) for supersets in supersetsList2rmss]
    else:
        supersetsList2rmss = None
        widthTag2 = 0

    # calculate the total number of tags and obtain the updated TagNumMatrix heap
    newTagNumMatrix, numTags = supersets2Matrix(supersetsList1rmss, supersetsList2rmss, colSet)

    print('Total Number of Tags: %d \tTag Width: %s \tTag Width: %s ' % (numTags, widthTag1, widthTag2))
    return newTagNumMatrix, numTags * (np.sum(widthTag1) + np.sum(widthTag2))


def extractOverlaps(supersetsList):
    # removing subsets (okay for objectiveFunction since neither widthTag nor numTags would be affected)
    supersetsList = [removeSubsets(supersets) for supersets in supersetsList]

    # tagNumMatrix is a heap sorted by the total number of tags for every column
    tagNumMatrix, cost = objectiveFunction(supersetsList)

    numCols = len(tagNumMatrix)
    numDFAs = len(tagNumMatrix[0]) - 2
    prodID = numDFAs + 1

    # get the set of column IDs and all thresholds of the number of tags
    colSet = [l[0] for l in tagNumMatrix]
    thresholdList = sorted(list(set([l[prodID] for l in tagNumMatrix])), reverse =True)
    if thresholdList[-1] == 1:
        del thresholdList[-1]
    fwidth = len(str(int(thresholdList[0])))

    # save all supersets for removeSubsets
    supersetsList1 = supersetsList
    supersetsList2 = [[set()] * numCols] * numDFAs

    # optimizing record
    colList = [[[] * numDFAs]]
    costList = [ cost ]

    print('Threshold: %*s \tCost: %d' % (fwidth, "None", cost))

    # iteratively find the threshold on the number of tags to optimize the memory consumption
    for t in thresholdList:
        newCol = [[]] * numDFAs
        i = 0
        while tagNumMatrix[i][prodID] >= t:
            colID = tagNumMatrix[i][0]
            numTags = tagNumMatrix[i][1 : prodID]
            index, sortedNumTags = zip(*sorted(zip(range(numDFAs), numTags), key = lambda x : x[1], reverse = True))

            # extract columns till the total number of tags is below the threshold
            for j in range(numDFAs):
                if np.prod(sortedNumTags[j : ]) >= t:
                    newCol[index[j]].append(colID)
                else:
                    break
            # check the next item
            i += 1

        # update supersetsList1 and supersetsList2
        supersetsList2new = [[s.intersection(newCol[i]) for s in supersetsList1[i]] for i in range(numDFAs)]
        supersetsList2 = [[s1.union(s2) for s1, s2 in zip(supersetsList2[i], supersetsList2new[i])] for i in range(numDFAs)]
        supersetsList1 = [[s.difference(newCol[i]) for s in supersetsList1[i]] for i in range(numDFAs)]

        # get the cost from the objective function and update the tagNumMatrix heap
        tagNumMatrix, cost = objectiveFunction(supersetsList1, supersetsList2, colSet)
        colList.append(newCol)
        costList.append(cost)
        print('Threshold: %*d \tCost: %d' % (fwidth, t, cost))







