#!/usr/bin/env python3.7
import random, pickle, json, argparse
from collections import Counter, defaultdict


def get_args():
    """ Parse command line arguments.
    """
    defaultOutfile = "route_matrix.pickle"
    defaultJsonFile = "matrix_readable.json"
    parser = argparse.ArgumentParser(description="Anonymous route matrix generation script.")
    parser.add_argument('infile',
                        help='File that contains RIBs (as announcement messages) in bgpdump format')
    parser.add_argument('-o', '--outfile', default=defaultOutfile,
            help='Destination file to write matrix. Default: ' + defaultOutfile)
    parser.add_argument('-j', '--jsonfile', default=defaultJsonFile,
            help="Destination file to write matrix in human-readable format. Default: " + defaultJsonFile)
    return parser.parse_args()



def anonymizeMatrix(matrix):
    """ Replace the column identifiers with random indices.
    """
    # get all columns
    allCols = set()
    for row in matrix:
        allCols.update(row)
    # shuffle all columns
    allCols = list(allCols)
    random.shuffle(allCols)
    # map the shuffled columns to indices
    oldColToNewCol = {oldCol:i for i,oldCol in enumerate(allCols)}
    # convert the columns to their indices
    newMatrix = [[oldColToNewCol[oldCol] for oldCol in row] for row in matrix]
    return newMatrix



def reduceRows(matrix):
    """ Convert a list of rows to a dictionary mapping rows to their number of occurrences.
        If a matrix has a large number of duplicate rows, this will greatly decrease the memory usage.
    """
    originalLen = len(matrix)
    # make the matrix rows hashable
    matrix = [frozenset(row) for row in matrix]
    # count row occurrences
    rowCounts = [(list(row), count) for row,count in Counter(matrix).items()]
    newLen = len(rowCounts)
    print("Matrix was reduced from %d rows to %d rows via deduplication" % (originalLen, newLen))
    return rowCounts



def makeReadable(matrixRowCounts):
    """ Given a list of tuples (row, row_count) which represent a reduced matrix, create a list
        with annotations and matrix information for dumping to a json for human readability.
    """
    allCols = set()
    numRowsTotal = 0
    for row, count in matrixRowCounts:
        allCols.update(row)
        numRowsTotal += count

    allCols = list(allCols)
    allCols.sort()

    matrixInfo = {".Note":"Although the matrix is binary, it is represented as lists of column IDs because BGP route matrices are typically extremely sparse. The column IDs denote non-zero positions.",
                  ".Num Unique Rows": len(matrixRowCounts),
                  ".Num Total Rows (prefixes)": numRowsTotal,
                  ".Num Columns (peers)": len(allCols),
                  "Columns (this should just be an contiguous integer range due to anonymization)" : str(allCols)}
    annotatedRowCounts = [{"Row Members":row, "Row Occurrences":count} for row,count in matrixRowCounts]

    matrixInfo['Matrix rows and their counts'] = annotatedRowCounts
    return matrixInfo
    


def bgpdumpToMatrix(filename):
    """ Takes as input a BGPdump-parsed MRT file. Returns a list of list of prefix announcers.
    """
    prefixToAS = {}
    with open(filename, 'r') as fp:
        AS = None
        inAnnounceList = False
        for line in fp:
            line = line.strip()
            if inAnnounceList:
                if not line.startswith("TIME:") and len(line) > 1: # not an empty line, should be a prefix
                    assert('.' in line)
                    announcers = prefixToAS.setdefault(line, [])
                    announcers.append(AS)
                else:
                    inAnnounceList = False
            elif line.startswith("ASPATH:"):
                AS = int(line.split()[1])
            elif line.startswith("ANNOUNCE"):
                inAnnounceList = True

    return list(prefixToAS.values())




def main():
    args = get_args()

    print("Using bgpdump filename:", args.infile)
    print("Results will be written to pickle file:", args.outfile)
    numSteps = 5

    print("Parsing bgpdump file.. (step 1 of %d)" % numSteps)
    matrix = bgpdumpToMatrix(args.infile)

    print("Done parsing. Anonymizing matrix.. (step 2 of %d)" % numSteps)
    matrix = anonymizeMatrix(matrix)

    print("Done anonymizing. Reducing matrix.. (step 3 of %d)" % numSteps)
    matrix = reduceRows(matrix)

    print("Done reducing. Writing matrix to pickle file.. (step 4 of %d)" % numSteps)
    with open(args.outfile, 'wb') as fp:
        pickle.dump(matrix, fp)
    print("Done writing pickle")


    print("The final matrix will now also be written to file %s for you to review for privacy violations in human-readable format." % args.jsonfile)
    print("Writing to JSON (step 5 of %d)" % numSteps)
    readableDump = makeReadable(matrix)
    with open(args.jsonfile, 'w') as fp:
        json.dump(readableDump, fp, ensure_ascii=True, indent=4, sort_keys=True)

    print("Done.")
    exit(0)



if __name__ == "__main__":
    main()
