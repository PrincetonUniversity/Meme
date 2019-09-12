from BaseCodes import BaseCode, BaseCodeStatic, ColumnID, Row, FixedRow, Matrix, FixedMatrix, BinaryString, TernaryStrings
from typing import List, Collection, Callable
from ClusterCodes import OriginalCodeStatic
import networkx as nx
import itertools


class MultiCode(BaseCodeStatic):
    subCodeClass     : BaseCode = None
    subCodes      : List[BaseCode] = None


    def __init__(self, matrix : Matrix = None, subCodeClass : BaseCode = OriginalCodeStatic, **kwargs) -> None:
        super().__init__(matrix=matrix, **kwargs)

        self.subCodeClass = subCodeClass
        self.subCodes = [subCodeClass(matrix=matrix, **kwargs)]


    def optimize(self, variant='vcut', **kwargs):
        if variant == 'vcut':
            self.optimizeVertexCuts(**kwargs)


    def spawnSubCode(self, subCodeColumns, sourceCodeIndex=-1) -> None:
        sourceCode = self.subCodes[sourceCodeIndex]
        sourceCode.unmake()
        self.made = False
        newMatrix = sourceCode.extractColumns(subCodeColumns)

        
        self.subCodes.append(self.subCodeClass(newMatrix))



    def optimizeVertexCuts(self, threshold = 10, **kwargs):
        """ Threshold is the maximum size of a connected component we allow.
        """

        while True:
            subCode = self.subCodes[-1]
            matrix = subCode.originalRows

            # a node for every column, and an edge for every pair of columns that occur together in any row
            G = nx.Graph()
            for row in matrix:
                for i1, i2 in itertools.combinations(row, 2):
                    G.add_edge(i1, i2)

            extractions = set()
            queue = [G.subgraph(nodes) for nodes in nx.connected_components(G)]
            while len(queue) > 0:
                cc = queue.pop()
                if len(cc) < threshold: continue
                cut = nx.minimum_node_cut(cc)
                extractions.update(cut)
                subG = cc.subgraph([n for n in cc if n not in set(cut)])
                queue.extend([subG.subgraph(nodes) for nodes in nx.connected_components(subG)])

            if len(extractions) == 0:
                break
            self.spawnSubCode(subCodeColumns=extractions)

        #for subCode in self.subCodes:
        #    subCode.mergeOverlaps()


    def make(self, *args, **kwargs) -> None:
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
        splitChar = '|' if decorated else ''
        outSub = ''
        subTags = [subCode.tag(row.intersection(subCode.columnIDs), decorated=decorated) for subCode in self.subCodes]
        return splitChar.join(subTags)


    def matchStrings(self, columnID, decorated=False) -> TernaryStrings:
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



def testMultiCode():
    matrix = [[1,2,3,4],
              [3,4,5],
              [3,4,6,7]]

    code = MultiCode(matrix)
    code.optimize(threshold=3)
    code.make()
    code.verifyCompression()
    print("Num subcodes:", code.numSubCodes())


if __name__=="__main__":
    testMultiCode()
