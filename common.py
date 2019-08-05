import os, sys
import pickle
import math
import time

REPO_PYTHON_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(REPO_PATH, 'data/')
PKL_PATH = os.path.join(REPO_PATH, 'pickles/')

from collections.abc import Iterable


VERBOSITY = 2
def logger(*args, **kwargs):
    lev = kwargs.get('lev', 2)
    if lev <= VERBOSITY:
        print(' '.join([str(x) for x in args]))


def to_pkl_path(filename):
    if not filename.endswith(".pickle"):
        filename += ".pickle"
    filepath = os.path.join(PKL_PATH, filename)
    return filepath

def pickle_exists(filename):
    filepath = to_pkl_path(filename)
    return os.path.isfile(filepath)


def from_pickle(filename, **kwargs):
    filepath = to_pkl_path(filename)
    logger("Loading pickle file", filepath)
    with open(filepath, 'rb') as fp:
        obj = pickle.load(fp, **kwargs)
    logger("Done loading", filepath)
    return obj


def to_pickle(obj, filename, **kwargs):
    filepath = to_pkl_path(filename)
    logger("Writing to pickle file", filepath)
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp, **kwargs)
    logger("Done writing pickle", filepath)




def ternary_compare(str1, str2):
    if len(str1) != len(str2):
        raise Exception("strings of unequal length compared: %s and %s (len %d and %d)" %(str1, str2, len(str1), len(str2)))
    for (c,d) in zip(str1,str2):
        # compare every pair of bits. failure short-circuits
        if not ((c=='*') or (d=='*') or (c==d)):
            return False
    return True

def longest_prefix_match(table, tag):
    matches = [prefix for prefix in table if ternary_compare(prefix, tag)]
    if len(matches) == 0: return -1
    return table.index(max(matches, key=lambda s:s.index('*') if '*' in s else len(s)))



def recoverRow(tag, queryDict):
    recovered = set()
    for col, queries in queryDict.items():
        if type(queries) == str or not isinstance(queries, Iterable):
            queries = [queries]
        for query in queries:
            if ternary_compare(tag, query):
                recovered.add(col)
                break
    return recovered

def verifyCompression(tagDict, queryDict, matrix=None):
    if matrix != None:
        for row in matrix:
            if frozenset(row) not in tagDict:
                raise Exception("A matrix row was not in the tag dictionary!")
    for row, tag in tagDict.items():
        recovered = recoverRow(tag, queryDict)
        if set(recovered) != set(row):
            print("Original row is", row)
            print("Recovered row is", recovered)
            raise Exception("Compression failed to verify")
    print("Compression verified successfully")
    return True




def prettySet(s):
    return  "{%s}" % ', '.join(str(i) for i in s)


def getShellWidth():
    return int(os.popen("tput cols", 'r').read())


def printShellDivider(text="", divChar="=", width=None):
    if width == None:
        width = getShellWidth()

    if len(text) > 0:
        numDivChars = max(width - len(text) - 2, 0)
        lWidth = numDivChars // 2
        rWidth = numDivChars - lWidth
        print(divChar * lWidth, text, divChar * rWidth)
    else:
        print(divChar * width)


def printAsColumns(items, title='', delim=", "):
    shellWidth = getShellWidth()


    printShellDivider(title)

    items = [str(item) for item in items]
    items.sort(key=len, reverse=True)

    delimWidth = len(delim)

    i = 0
    while i < len(items):
        if len(items[i]) + delimWidth > shellWidth:
            print(items[i] + ",")
            i += 1
        else:
            numCols = shellWidth // (len(items[i])+delimWidth)
            colWidth = shellWidth // numCols
            for item in items[i : i+numCols]:
                print(item + delim + " "*(colWidth-len(item)-delimWidth), end="")
            print()
            i += numCols

    printShellDivider()


COMMON_TIMER_CLOCK = None
def printTimer(init=False):
    global COMMON_TIMER_CLOCK
    currTime = time.time()
    if COMMON_TIMER_CLOCK == None or init:
        print("Timer initialized.")
    else:
        elapsedTime = currTime - COMMON_TIMER_CLOCK
        elapsedDiscrete = int(math.floor(elapsedTime))
        print("%2d min %5.2f sec elapsed since last timer call." % (elapsedDiscrete // 60, elapsedTime % 60))
    COMMON_TIMER_CLOCK = currTime


def pointerBitsFor(numItems):
    """ Returns the number of bits needed to distinctly identify every element in a set of size 'count'.
    """
    return  (numItems-1).bit_length()

