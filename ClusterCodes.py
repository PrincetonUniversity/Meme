from BaseCodes import BaseCodeStatic, ColumnID, Row, Matrix, FixedMatrix, BinaryString
from util import generateIdentifiers, kraftsInequality
from optimize import removeSubsets, minimizeRulesGreedy, minimizeVariableWidthGreedy

from typing import List, Set, Dict, FrozenSet, Collection

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
        if implicits == None or not self.hideImplicits:
            implicits = [[] for _ in self.clusters]
        else:
            self.clusterImplicits = list(implicits)
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
            if cluster.issuperset(subset):
                return i
        return -1



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
            cluster = cluster.difference_update(self.clusterImplicits[ssIndex])


        if mask == None:
            mask = ''.join(['1' if colID in subset else maskNegChar for colID in cluster])

        padding = padChar * (width - (len(identifier) + len(mask)))

        return identifier + partChar + padding + partChar + mask



    def tag(self, row, decorated=False):
        assert self.made
        index = self.__indexOfCluster(row)

        partChar = '_' if decorated else ''
        return self.buildString(index, subset=row, padChar='0', partChar=partChar)


    def matchStrings(self, columnID, decorated=False):
        assert self.made
        tagWidth = self.width()
        partChar = '_' if decorated else ''
        outStrings = []
        for i, cluster in enumerate(self.clusters):
            if columnID in cluster:
                outStrings.append(self.buildString(i, subset=[columnID], width=tagWidth, padChar='*', partChar=partChar))

        return outStrings


    def allMatchStrings(self, decorated=False):
        assert self.made
        tagWidth = self.width()
        partChar = '_' if decorated else ''

        outStrings = {colID : [] for colID in self.columnIDs}
        for cID, cluster in enumerate(self.clusters):
            maskBits = ['*'] * len(cluster)
            for i, columnID in enumerate(cluster):
                maskBits[i] = '1'
                mask = ''.join(maskBits)
                outStrings[columnID].append(self.buildString(cID, mask=mask, width=tagWidth, padChar='*', partChar=partChar))
                maskBits[i] = '*'

        return outStrings


class OriginalCodeStatic(ClusterCode):

    def __init__(self, matrix=None, **kwargs):
        super().__init__(matrix=matrix, hideImplicits=False, **kwargs)


    def make(self, optWidth=False, maxWidth=-1, mergeOverlaps=True):
        clusters = removeSubsets(self.matrix)

        if optWidth:
            clusters = minimizeVariableWidthGreedy(clusters)

        minWidth = kraftsInequality([len(cluster) for cluster in clusters])
        if maxWidth != -1:
            if maxWidth >= minWidth:
                clusters = minimizeRulesGreedy(clusters, {colID:1 for colID in self.columnIDs}, maxWidth)
            else:
                self.logger.warning("Given maximum tag width is too small! Max given is %d, min is %d" % (maxWidth, minWidth))

        self.setClusters(clusters)

        self.makeIdentifiers()

        self.made = True




def main():
    matrix = [[1,2,3], [3,4,5], [6,7]]
    code = OriginalCodeStatic(matrix=matrix)
    code.timeMake(maxWidth=4, optWidth=True)
    code.verifyCompression()
    print("Memory of original code:", code.memoryRequired())
    print("Tag for empty row is", code.tag([]))



if __name__ == "__main__":
    main()
