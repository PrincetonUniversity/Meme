from BaseCodes import BaseCode, BaseCodeStatic, ColumnID, Row, FixedRow, Matrix, FixedMatrix, BinaryString, TernaryStrings
from typing import List, Collection, Callable, Dict
from ClusterCodes import OriginalCodeStatic, NewerCodeStatic
import networkx as nx
import itertools
import util
import analyze

from HierarchicalCode import HCode
from RSets import RCode
from analyze import groupIdenticalColumns, groupOverlappingRows
from graphAlgorithm import graphHierarchy, allOneCuts

class MultiCode(BaseCodeStatic):
    subCodeClass    : BaseCode = None
    subCodes        : List[BaseCode] = None
    shadowElements  : Dict[ColumnID, ColumnID] = None # maps shadow elements to the elements they mimic
    optimized       : bool = False


    def __init__(self, matrix : Matrix = None, subCodeClass : BaseCode = NewerCodeStatic, **kwargs) -> None:
        super().__init__(matrix=matrix, **kwargs)


        self.matrix, self.shadowElements = self.stripShadows(self.originalRows)

        self.subCodeClass = subCodeClass
        self.subCodes = [subCodeClass(matrix=self.matrix)]


    def printInfo(self):
        if not self.made: return
        #print("Number of Subcodes: %d, Subcode Widths:" % len(self.subCodes), [subCode.width() for subCode in self.subCodes])
        line = '-'*20
        header = line + ' %d SubCodes ' % len(self.subCodes) + line
        print(header)
        for i, subCode in enumerate(self.subCodes):
            print("--- SubCode %d " %i, line)
            subCode.printInfo()
        print("--- Overall", line)
        print("Subcode widths:", [subCode.width() for subCode in self.subCodes])
        super().printInfo()
        print('-' * len(header))



    def stripShadows(self, matrix) -> (Matrix, Dict[ColumnID, ColumnID]):
        """ Remove identically behaving elements from the given matrix.
            Return the stripped matrix, and a dictionary mapping the removed 
            elements to the elements they mimic.
        """
        shadowElementMap : Dict[ColumnID, ColumnID] = {}
        identicalGroups = groupIdenticalColumns(self.originalRows)
        for group in identicalGroups:
            nonShadow = group[0]
            for shadow in group[1:]:
                shadowElementMap[shadow] = nonShadow

        shadows = set(shadowElementMap.keys())

        newMatrix = [frozenset(row).difference(shadows) for row in matrix]

        return newMatrix, shadowElementMap



    def optimize(self, variant='vcut', **kwargs):
        self.logger.info("Optimizing with variant " + variant)
        if variant == 'vcut':
            # Vertex cut
            self.optimizeVertexCuts(**kwargs)
        elif variant == 'hier':
            # Hierarchy
            self.optimizeHierarchy(**kwargs)
        elif variant == '1cut':
            self.extractOneCuts(**kwargs)
        self.optimized = True


    def extractOneCuts(self, **parameters):
        def _getOneCuts(_code):
            submatrices = groupOverlappingRows(_code.matrix)
            oneCuts = []
            for submatrix in submatrices: oneCuts.extend(allOneCuts(submatrix))
            return oneCuts

        extractions = _getOneCuts(self.subCodes[-1])
        while len(extractions) > 0:
            self.spawnSubCode(extractions)
            if len(self.subCodes[-1].columnIDs) <= 1: break
            extractions = _getOneCuts(self.subCodes[-1])


    def optimizeHierarchy(self, **parameters):
        supersetsList, absHierarchyList = graphHierarchy(self.matrix, **parameters)

        self.codes = [self.subCodeClass(supersets, absHierarchy=absHierarchy) for supersets,absHierarchy in zip(supersetsList, absHierarchyList)]

        for code in self.codes:
            code.buildCode()



    def spawnSubCode(self, subCodeColumns, sourceCodeIndex=-1) -> None:
        sourceCode = self.subCodes[sourceCodeIndex]
        sourceCode.unmake()
        self.made = False
        newMatrix = sourceCode.extractColumns(subCodeColumns)

        
        self.subCodes.append(self.subCodeClass(newMatrix))

    

    def optimizeVertexCuts(self, **kwargs):


        codeCostFunc = lambda m: analyze.bitsRequiredVariableID(analyze.groupOverlappingRows(m, asRows=False))
        #codeCostFunc = lambda m: util.longestLen(analyze.groupOverlappingRows(m, asRows=False))

        submatrices = analyze.dissectMatrix(self.matrix, matrixCostFunc=codeCostFunc)
        # uncomment the line below this to force all subcodes to have a maximum mask size
        #submatrices = analyze.dissectMatrix(self.matrix, maxComponentSize=5)
        self.subCodes = [self.subCodeClass(submatrix) for submatrix in submatrices]




    def make(self, *args, dontOptimize=False, **kwargs) -> None:
        if (not self.optimized) and (not dontOptimize): self.optimize()

        for subCode in self.subCodes:
            subCode.make(*args, **kwargs)
        self.made = True

    def unmake(self, *args, **kwargs) -> None:
        for subCode in self.subCodes:
            subCode.unmake(*args, **kwargs)
        self.made = False


    def width(self):
        return sum([subCode.width() for subCode in self.subCodes])

    def numSubCodes(self):
        return len(self.subCodes)

    
    def tag(self, row, decorated=False) -> BinaryString:
        row = set([self.shadowElements.get(columnID, columnID) for columnID in row]) # convert any shadows to parents

        splitChar = '|' if decorated else ''
        outSub = ''
        subTags = [subCode.tag(row.intersection(subCode.columnIDs), decorated=decorated) for subCode in self.subCodes]
        return splitChar.join(subTags)


    def matchStrings(self, columnID, decorated=False) -> TernaryStrings:
        columnID = self.shadowElements.get(columnID, columnID) # if its a shadow, grab its parent

        splitChar = '|' if decorated else ''
        codeIdx = 0
        while columnID not in self.subCodes[codeIdx].columnIDs:
            codeIdx += 1

        # one piece per subCode, initially all wildcards
        pieces = ['*'*subCode.width() for subCode in self.subCodes]
        outStrs = []
        for subMatchString in self.subCodes[codeIdx].matchStrings(columnID, decorated=decorated):
            pieces[codeIdx] = subMatchString
            outStrs.append(splitChar.join(pieces))

        return outStrs


    def verifyCompression(self):
        for i, subCode in enumerate(self.subCodes):
            self.logger.info("Verifying subcode %d" % i)
            subCode.verifyCompression()
        self.logger.info("All subcodes idependently verified")
        super().verifyCompression()



def testMultiCode():
    matrix = [[1,2,3,4],
              [3,4,5],
              [3,4,6,7]]

    code = MultiCode(matrix)
    code.optimize(variant='vcut')
    code.make()
    code.verifyCompression()
    print("Num subcodes:", code.numSubCodes())
    code = MultiCode(matrix)
    code.optimize(variant='vcut')
    code.make()
    code.verifyCompression()
    print("Num subcodes:", code.numSubCodes())


if __name__=="__main__":
    testMultiCode()
