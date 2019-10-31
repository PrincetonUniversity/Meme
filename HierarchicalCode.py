from optimize import minimizeVariableWidthGreedy, minimizeRulesGreedy, mergeIntersectingSets, generateCodeWords
from analyze import getSupersetIndex, bitsRequiredVariableID, ternary_compare
from AbsNode import AbsNode
from BaseCodes import BinaryString, TernaryStrings, BaseCodeStatic

import math
from bidict import bidict
from collections import deque as queue

from typing import List,Set,Dict,NewType,FrozenSet

String = NewType('String', str)


class HCode(BaseCodeStatic):
    supersets       : List[Set] = None
    identifiers     : List[BinaryString] = None
    freeCodes       : List[String] = None

    absHierarchy : List = None # list of encoding
    absCodes : Dict = None # prefix coding for absolute elements
    emptyCode = ''



    def __init__(self, matrix, absHierarchy, **kwargs):
        super().__init__(matrix=matrix, **kwargs)

        self.absHierarchy = absHierarchy
        self.absCodes = {}
        self.emptyCode = ''

        self.supersets = [set(row) for row in matrix]


    def groupingStrategy(self, groups):
        self.codeBuilt = False
        self.supersets = [set(g) for g in groups]



    def extract(self, elements):
        """ Remove the given elements'columns from the matrix, and return
            the removed columns as a submatrix
        """
        self.codeBuilt = False
        elements = set(elements)
        submatrix = [elements.intersection(row) for row in self.matrix]
        for superset in self.supersets:
            superset.difference_update(elements)
        self.columnIDs.difference_update(elements)
        return submatrix



    def matchStrings(self, element, decorated=False, printD=False) -> TernaryStrings:
        """ Given an element, return a wildcard string for every superset that element appears in. This
            set of wildcard strings, when matched against a packet's tag, determines if that element is present.
        """
        if not self.made:
            self.make()

        separator, padChar = '', '*'
        if decorated:
            separator, padChar = '-', '~'

        idMaskPairs = []
        if  element in self.absCodes:
            identifier = self.absCodes[element][0]
            mask = ''
            idMaskPairs.append((identifier, mask))

        else:
            for superset in self.supersets:
                if element not in superset:
                    continue

                identifier = superset.codeword
                mask = superset.queryMask(element)
                idMaskPairs.append((identifier, mask))

        width = self.width()
        strings = [''.join([id, separator, padChar*(width - (len(id) + len(mask))), mask]) for (id,mask) in idMaskPairs]
        if len(strings) == 0:
            self.logger.warning("No match strings generated for element:", element)
        return strings


    def getSupersetIndex(self, elements):
        """ Return the index of the first superset that contains all the given elements.
            Returns -1 if no superset exists.
        """
        absolutes = self.absCodes.keys()
        subset = set(elements).difference(absolutes)
        for i, superset in enumerate(self.supersets):
            if superset.issuperset(subset):
                return i
        return -1


    def width(self, includePadding : bool = False) -> int:
        if not self.made: self.make()
        return  max([len(superset) + len(identifier) for superset,identifier in zip(self.supersets,self.identifiers)]) 


    def tag(self, elements, decorated=False):
        """ Given an element set, returns a binary string to be used as a packet tag.
            If decorated, padding bits are replaced with ~, and the identifier and mask are
            separated by a - character.
        """
        if not self.codeBuilt:
            self.buildCode()

        subset = set(elements)

        if len(subset) == 0:
            identifier = self.emptyCode
            mask = ''
        elif subset.issubset(self.absCodes.keys()):
            identifier = max([self.absCodes[element] for element in subset], key=lambda x : len(x[0]))[1]
            mask = ''
        else:
            ssIndex = self.getSupersetIndex(subset)
            if ssIndex == -1:
                self.logger.warning("Asking for a tag for an unseen element set: ", elements)
                return ""

            superset = self.supersets[ssIndex]
            mask = ''.join([1 if elem in subset else 0 for elem in superset])
            identifier = self.identifiers[ssIndex]


        width = self.width()
        paddingLen = width - ((len(identifier) + len(mask)))
        padding = ('~' if decorated else '0') * paddingLen

        separator = '-' if decorated else ''

        return identifier + separator + padding + mask


    def make(self):
        if self.made: return
        self.buildCode()
        self.made = True

    def unmake(self):
        self.made = False

    def buildCode(self):
        """ Rebuilds all identifiers. Sets codeBuilt to True.
        """
        self.logger.debug("Building codewords... ")

        codewordMap, self.freeCodes, self.absCodes = generateCodeWords(self.absHierarchy, absHierarchy=True)
        supersetMap = {frozenset(superset.variables) : superset for superset in self.supersets}
        for k,v in codewordMap.items():
            if k == frozenset(["E"]):
                self.emptyCode = v
            else:
                supersetMap[k].codeword = v
        for ss in self.supersets:
            if ss.codeword == None:
                print("abs", ss.absolutes)
                print("var", ss.variables)
                print()

        self.codeBuilt = True
        self.logger.debug("Done building codewords.")






def unit_test():
    matrix = [[1,2,3],
              [2,3,4],
              [6,7,8],
              [7,8],
              [1,2],
              [1]]
    matrix = [[1,2,3],
              [2,3],
              [3,1],
              [3,4],
              [4,5],
              [5,3]]
    #rcode = RCode(matrix)
    #rcode.verifyCompression()

if __name__=="__main__":
    unit_test()
