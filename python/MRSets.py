from typing import List,Set,Dict
from RSets import RCode



class MRCode(object):
    rcodes : List = None
    elements : Set = None


    def __init__(self, elementSets):
        rcode = RCode(elementSets)
        rcode.optimizeWidth()

        self.elements = rcode.elements.copy()

        self.rcodes = [rcode]


    def spawnSubCode(self):
        rcode = self.rcodes[-1]
        occurrences = rcode.elementOccurrences()
        expensiveElements = [elem for elem in rcode.elements if occurrences[elem] > 2]
        if len(expensiveElements) == 0: return

        submatrix = rcode.extract(expensiveElements)
        rcode.removeSubsets()
        rcode.optimizeWidth()
        rcode.removePadding()

        newCode = RCode(submatrix)
        newCode.optimizeWidth()
        newCode.removePadding()

        self.rcodes.append(newCode)

    def tagWidth(self):
        return sum(code.widthUsed() for code in self.rcodes)



    def tagString(self, elements, decorated=False):
        elements = set(elements)
        subtags = []
        for code in self.rcodes:
            subelements = elements.intersection(code.elements)
            subtags.append(code.tagString(subelements, decorated))

        separator = '|' if decorated else ''
        return separator.join(subtags)


    def allMatchStrings(self, elements):
        padChar = '*'
        totalWidth = self.tagWidth()
        lPaddingLen = 0
        strings = {}
        for code in self.rcodes:
            substrings = code.allMatchStrings(code.elements)
            subTagWidth = code.widthUsed()
            rPaddingLen = totalWidth - subTagWidth - lPaddingLen

            for elem in code.elements:
                strings[elem] = [padChar*lPaddingLen + subString + padChar*rPaddingLen for substring in substrings[elem]]

            lPaddingLen += subTagWidth

        return strings




if __name__=="__main__":
    matrix = [[1,2,3],
              [3,4,5],
              [3,6,7]]

    code = MRCode(matrix)
    for row in matrix:
        print(code.tagString(row, True))

    code.spawnSubCode()
    for row in matrix:
        print(code.tagString(row, True))


