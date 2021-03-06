from BaseCodes import BaseCodeStatic, ColumnID, Row, Matrix, FixedMatrix, BinaryString
from util import generateIdentifiers, kraftsInequality, printShellDivider
from optimize import removeSubsets, minimizeRulesGreedy, minimizeVariableWidthGreedy, mergeIntersectingSets, minimizeMemoryGreedy
import analyze

from typing import List, Set, Dict, FrozenSet, Collection

from MatrixParameters import randomMatrix

# memory requirements are
#   sum( sum(tagWidth + overhead for overhead in ruleOverheads[elem])*occurrences[elem] for elem in elements)

# compare the minimum possible tag size that the schemes can achieve
# then, compare the minimum rule inflation the schemes achieve below different field widths: 16, 32, 48, 64 bits


class ClusterCode(BaseCodeStatic):

    clustersMade : bool = False
    clusters : Matrix = None
    identifiersMade : bool = False
    identifiers : List[BinaryString] = None
    freeIdentifiers : List[BinaryString] = None

    hideImplicits : bool = True
    clusterImplicits : List[Set[ColumnID]] = None

    def __init__(self, matrix=None, hideImplicits=True, **kwargs):
        super().__init__(matrix=matrix, **kwargs)

        self.clusters = []
        self.identfiers = []
        self.hideImplicits = hideImplicits
        if self.hideImplicits:
            self.clusterImplicits = []



    def makeClusters(self, rowGroupings : List[List[Row]]):

        rowGroupings = [[set(row) for row in group] for group in rowGroupings]

        clusters = [set.union(*group) for group in rowGroupings]
        implicits = []
        if self.hideImplicits:
            implicits = [set.intersection(*group) for group in rowGroupings]

        self.setClusters(clusters, implicits)


    def setClusters(self, clusters, implicits=None):
        self.made = False
        self.clusters = [set(cluster) for cluster in clusters]
        if implicits == None:
            self.clusterImplicits = [set() for _ in self.clusters]
        else:
            self.clusterImplicits = [set(implSet) for implSet in implicits]

        "Ensure some cluster can produce a tag for the empty set"
        for implicitSet in self.clusterImplicits:
            if len(implicitSet) == 0:
                break
        else:
            self.clusters.append(set())
            self.clusterImplicits.append(set())

        self.clustersMade = True



    def makeIdentifiers(self, identifiers=None):
        self.made = False
        if identifiers != None:
            if len(identifiers) != len(self.clusters):
                raise exception("Given set of identifiers is not same length as set of clusters!")
            self.identifiers = list(identifiers)
            return

        idLengths = [len(cluster) for cluster in self.clusters]
        if self.hideImplicits:
            idLengths = [idLen - len(implicits) for idLen, implicits in zip(idLengths, self.clusterImplicits)]
        self.identifiers, self.freeIdentifiers = generateIdentifiers(idLengths)

        self.identifiersMade = True



    def make(self, *args, **kwargs):
        assert self.clustersMade and self.identifiersMade

        self.made = True


    def unmake(self, *args, **kwargs):
        self.clusters, self.clusterImplicits = None, None
        self.clustersMade = False

        self.identifiers, self.freeIdentifiers = None, None
        self.identifiersMade = False
        self.made = False


    def extractColumns(self, colsToExtract : Collection[ColumnID]) -> FixedMatrix:
        self.unmake()

        colsToExtract = frozenset(colsToExtract)
        extractedMatrix = [colsToExtract.intersection(row) for row in self.matrix]
        self.matrix = [row.difference(colsToExtract) for row in self.matrix]

        self.columnIDs = [colID for colID in self.columnIDs if colID not in colsToExtract]
        
        return extractedMatrix



    def width(self):
        if not self.made:
            return -1
        return max(len(cluster) + len(ID) for cluster,ID in zip(self.clusters, self.identifiers))



    def __indexOfCluster(self, subset):
        for i, cluster in enumerate(self.clusters):
            if cluster.issuperset(subset) and \
                    ((not self.hideImplicits) or self.clusterImplicits[i].issubset(subset)):
                return i
        return None



    def buildString(self, ssIndex, mask=None, subset=None, padChar='0', maskNegChar='', partChar='', width=None):
        """ padChar can be '0' or '*'
            maskNegChar is the characters for columnIDs absent from a mask. Can be '0' or '*'
            partChar can be '' or '|', and goes between the identifier and padding, and the padding and mask
        """
        assert not (mask == None and subset == None)
        if maskNegChar == '':
            maskNegChar = padChar
        if width == None:
            width = self.width()


        identifier = self.identifiers[ssIndex]

        cluster = self.clusters[ssIndex]
        if self.hideImplicits:
            cluster = cluster.difference(self.clusterImplicits[ssIndex])
        
        if mask == None:
            mask = ''.join(['1' if colID in subset else maskNegChar for colID in sorted(cluster)])

        padding = padChar * (width - (len(identifier) + len(mask)))

        return identifier + partChar + padding + partChar + mask



    def tag(self, row, decorated=False):
        assert self.made
        index = self.__indexOfCluster(row)
        if index == None:
            raise Exception("COULD NOT FIND INDEX FOR " + str(row))
        partChar = '_' if decorated else ''
        return self.buildString(index, subset=row, padChar='0', partChar=partChar)


    def matchStrings(self, columnID, decorated=False):
        assert self.made
        tagWidth = self.width()
        partChar = '_' if decorated else ''
        outStrings = []
        for i, cluster in enumerate(self.clusters):
            if columnID in cluster:
                subset = [columnID]
                if columnID in self.clusterImplicits[i]:
                    subset = []
                outStrings.append(self.buildString(i, subset=[columnID], width=tagWidth, padChar='*', partChar=partChar))

        return outStrings


    def allMatchStrings(self, decorated=False):
        assert self.made
        tagWidth = self.width()
        partChar = '_' if decorated else ''

        outStrings = {colID : [] for colID in self.columnIDs}
        for cID, cluster in enumerate(self.clusters):
            explicits = list(cluster.difference(self.clusterImplicits[cID]))
            explicits.sort()
            maskBits = ['*'] * len(explicits)
            for i, columnID in enumerate(explicits):
                maskBits[i] = '1'
                mask = ''.join(maskBits)
                outStrings[columnID].append(self.buildString(cID, mask=mask, width=tagWidth, padChar='*', partChar=partChar))
                maskBits[i] = '*'
            # now make strings for implicits
            mask = '*' * len(explicits)
            for columnID in self.clusterImplicits[cID]:
                outStrings[columnID].append(self.buildString(cID, mask=mask, width=tagWidth, padChar='*', partChar=partChar))


        return outStrings


class OriginalCodeStatic(ClusterCode):

    def __init__(self, matrix=None, **kwargs):
        super().__init__(matrix=matrix, hideImplicits=False, **kwargs)


    def make(self, optWidth=True, maxWidth=-1, mergeOverlaps=False):
        clusters = removeSubsets(self.matrix)

        if optWidth:
            self.logger.debug("Optimizing width")
            clusters = minimizeVariableWidthGreedy(clusters)
            self.logger.debug("Done optimizing width.")

        minWidth = kraftsInequality([len(cluster) for cluster in clusters])
        if mergeOverlaps:
            clusters = mergeIntersectingSets(clusters)
        elif maxWidth != -1:
            if maxWidth >= minWidth:
                self.logger.debug("Optimizing rule count given a maximum tag width")
                clusters = minimizeRulesGreedy(clusters, {colID:1 for colID in self.columnIDs}, maxWidth)
                self.logger.debug("Done optimizing rule count.")
            else:
                self.logger.warning("Given maximum tag width is too small! Max given is %d, min is %d" % (maxWidth, minWidth))
        else:
            self.logger.debug("Optimizing memory")
            clusters = minimizeMemoryGreedy(clusters)
            self.logger.debug("Done optimizing memory")

        self.setClusters(clusters)

        self.makeIdentifiers()

        self.made = True



class NewerCodeStatic(ClusterCode):
    def __init__(self, matrix=None, **kwargs):
        super().__init__(matrix=matrix, hideImplicits=True, **kwargs)

    def make(self):
        clusters, implicits = analyze.groupOverlappingRows(self.matrix, asRows=False, withImplicits=True)

        self.setClusters(clusters, implicits)

        self.makeIdentifiers()

        self.made = True

"""
 have to handle the empty set in the event of implicits
"""



def main():
    matrix = [[1,2,3], [3,4,5], [6,7], []]
    """
    code = OriginalCodeStatic(matrix=matrix)
    code.timeMake(maxWidth=4, optWidth=True)
    code.verifyCompression()
    print("Memory of original code:", code.memoryRequired())
    print("Tag for empty row is", code.tag([]))
    """

    for i in range(5):
        try:
            randMatrix = randomMatrix(rows=1000, columns=100, density=0.01)

            code = OriginalCodeStatic(matrix=randMatrix)
            code.timeMake()
            code.verifyCompression()
        except:
            print("Code verification FAILED")
            printShellDivider("Original Matrix")
            print(randMatrix)
            printShellDivider("Clusters")
            print(code.clusters)
            printShellDivider("Tags")
            print(code.allTags(decorated=True))
            printShellDivider("Strings")
            print(code.allMatchStrings(decorated=True))
            raise Exception()


if __name__ == "__main__":
    main()
