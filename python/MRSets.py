from typing import List,Set,Dict
from RSets import RCode



class MRCode(object):
    rcodes : List = None
    elements : Set = None


    def __init__(self, elementSets):

        # we start off with a single code
        rcode = RCode(elementSets)
        rcode.optimizeWidth()
        self.rcodes = [rcode]

        self.elements = rcode.elements.copy()



    def spawnSubCode(self):
        # grab the 'latest' code
        rcode = self.rcodes[-1]

        # find any elements in the code that occur more than twice
        occurrences = rcode.elementOccurrences()
        expensiveElements = [elem for elem in rcode.elements if occurrences[elem] > 2]
        if len(expensiveElements) == 0: return

        # remove them from the code, and then re-optimize that code
        submatrix = rcode.extract(expensiveElements)
        rcode.removeSubsets()
        rcode.optimizeWidth()
        rcode.removePadding()

        # place the removed elements into a new code
        newCode = RCode(submatrix)
        newCode.optimizeWidth()
        newCode.removePadding()

        # add the new code to the list of codes
        self.rcodes.append(newCode)

    def tagWidth(self):
        return sum(code.widthUsed() for code in self.rcodes)



    def tagString(self, elements, decorated=False):
        """ Given a set of elements, returns a string which represents
            that set in compressed form.
        """
        elements = set(elements)
        subtags = []
        for code in self.rcodes:
            subelements = elements.intersection(code.elements)
            subtags.append(code.tagString(subelements, decorated))

        separator = '|' if decorated else ''
        return separator.join(subtags)


    def allMatchStrings(self, elements):
        """ For every element in elements, return a list of of ternary strings which,
            when compared to a compressed tag, will determine if the respective
            element is present in the tag or not.
        """
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


