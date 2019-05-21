from typing import List,Set,Dict
try:
    from .RSets import RCode
    from .analyze import groupIdenticalColumns
except:
    from RSets import RCode
    from analyze import groupIdenticalColumns



class MRCode(object):
    rcodes : List = None
    elements : Dict = None # maps elements to rcode indices

    shadowElements : Dict = None # maps elements to elements for which they have identical behavior

    def __init__(self, elementSets):
        # remove duplicates
        elementSets = set([frozenset(es) for es in elementSets])


        # find shadow elements
        self.shadowElements = {}
        identicalElementGroups = groupIdenticalColumns(elementSets)
        # for every set of elements that behave identically, arbitrarily make the first element in the set the "parent" of all the rest. All elements but the parents are now shadows
        for elemGroup in identicalElementGroups:
            for elem in elemGroup[1:]:
                self.shadowElements[elem] = elemGroup[0]

        # remove shadow elements
        shadows = set(self.shadowElements.keys())
        elementSets = [es.difference(shadows) for es in elementSets]


        # we start off with a single code
        rcode = RCode(elementSets)
        rcode.optimizeWidth()
        self.rcodes = [rcode]

        # initially every element belongs to the 0th (and only) rcode
        self.elements = {element : 0 for element in rcode.elements}

    def optimize(self):
        """ TODO: Do something more clever.
        """
        while self.extractHeavyHitters(1):
            pass

    def extractHeavyHitters(self, threshold=2):
        """ Placeholder implementation. Not to be treated as an intelligent algorithm.
            Extracts all elements that appear in more than 'threshold' supersets.
        """
        # grab the 'latest' code
        rcode = self.rcodes[-1]

        # find any elements in the code that occur more than twice
        occurrences = rcode.elementOccurrences()
        expensiveElements = [elem for elem in rcode.elements if occurrences[elem] > threshold]
        if len(expensiveElements) == 0: return False

        self.spawnSubCode(extractionSet=expensiveElements)

        return True



    def spawnSubCode(self, extractionSet, rcodeSrcIndex=-1, rcodeDstIndex=None):
        """ Given a set of columns and two rcode indices, extract the columns from the src rset,
            and insert them into the dst rset.
        """
        # grab the src rcode
        rcode = self.rcodes[rcodeSrcIndex]


        # remove them from the code, and then re-optimize that code
        submatrix = rcode.extract(extractionSet)
        rcode.removeSubsets()
        rcode.optimizeWidth()
        rcode.removePadding()

        dstCode = None
        if rcodeDstIndex != None:
            # place the extractions in an existing rcode
            dstCode = self.rcodes[rcodeDstIndex]
            # TODO: this
            raise NotImplemented
        else:
            # place the removed elements into a new code
            dstCode = RCode(submatrix)
            # add the new code to the list of codes
            rcodeDstindex = len(self.rcodes)
            self.rcodes.append(dstCode)
        dstCode.optimizeWidth()
        dstCode.removePadding()


        # update the location of the elements
        for element in extractionSet:
            self.elements[element] = rcodeDstIndex


    def width(self):
        return sum(code.widthUsed() for code in self.rcodes)



    def tagString(self, elements, decorated=False):
        """ Given a set of elements, returns a string which represents
            that set in compressed form.
        """
        elements = set([self.shadowElements.get(e, e) for e in elements])
        subtags = []
        for code in self.rcodes:
            subelements = elements.intersection(code.elements)
            subtags.append(code.tagString(subelements, decorated))

        separator = '|' if decorated else ''
        return separator.join(subtags)




    def matchStrings(self, elements=None):
        """ For every element in elements, return a list of of ternary strings which,
            when compared to a compressed tag, will determine if the respective
            element is present in the tag or not.
        """
        if elements == None:
            elements = set(self.elements.keys())
            elements.update(self.shadowElements.keys())
        inputElements = elements.copy()
        elements = set([self.shadowElements.get(e, e) for e in elements])

        padChar = '*'
        totalWidth = self.width()
        lPaddingLen = 0
        strings = {}
        for code in self.rcodes:
            subset = elements.intersection(code.elements)
            substrings = code.allMatchStrings(subset)
            subTagWidth = code.widthUsed()
            rPaddingLen = totalWidth - subTagWidth - lPaddingLen

            for elem in subset:
                strings[elem] = [padChar*lPaddingLen + substring + padChar*rPaddingLen for substring in substrings[elem]]

            lPaddingLen += subTagWidth

        strings = {e:strings[self.shadowElements.get(e,e)] for e in inputElements}
        return strings




if __name__=="__main__":
    matrix = [[1,2,3,4],
              [3,4,5],
              [3,4,6,7]]

    cols = set()
    for row in matrix:
        cols.update(row)

    code = MRCode(matrix)
    for row in matrix:
        print(row, "->", code.tagString(row, True))

    code.extractHeavyHitters()
    for row in matrix:
        print(row, "->", code.tagString(row, True))

    print(code.matchStrings(cols))


