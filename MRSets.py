import networkx as nx
import itertools
from typing import List,Set,Dict
try:
    from .RSets import RCode
    from .analyze import groupIdenticalColumns
    from .biclustering import biclusteringHierarchy, graphHierarchy
except:
    from RSets import RCode
    from analyze import groupIdenticalColumns
    from biclustering import biclusteringHierarchy, graphHierarchy



class MRCode(object):
    rcodes : List = None
    elements : Dict = None # maps elements to rcode indices
    originalSets : List = None

    shadowElements : Dict = None # maps elements to elements for which they have identical behavior
    hierarchy : bool = False
    nonShadowElements : Set = None

    def __init__(self, elementSets, hierarchy = False):
        # remove duplicates
        elementSets = set([frozenset(es) for es in elementSets])

        self.originalSets = list(elementSets)
        self.hierarchy = hierarchy

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

        if not hierarchy:
            # we start off with a single code
            rcode = RCode(elementSets)
            #rcode.optimizeWidth()
            self.rcodes = [rcode]

            # initially every element belongs to the 0th (and only) rcode
            self.elements = {element : 0 for element in rcode.elements}
            self.nonShadowElements = None
        else:
            # Initialization for biclustering hierarchy algorithm
            self.rcodes = []
            self.elements = {}
            self.nonShadowElements = elementSets


    def useStrategy(self, strategy):
        """ A strategy is a list of groupings. A grouping is a list of supersets.
        """
        itemSets = [set.union(*[set(ss) for ss in grouping]) for grouping in strategy]
        matrices = [[os.intersection(itemSet) for os in self.originalSets] for itemSet in itemSets]

        self.rcodes = [RCode(matrix) for matrix in matrices]

        for rcode, grouping in zip(self.rcodes, strategy):
            rcode.groupingStrategy(grouping)


    def useHierarchyStrategy(self, supersetsList, absHierarchyList):
        """ A strategy is a list of groupings. A absHierarchy is a list of trees of absolute element dependency. 
            Build rcodes from supersetsList and absHierarchyList
        """
        self.rcodes = [RCode(supersets, absHierarchy=absHierarchy) for supersets, absHierarchy in zip(supersetsList, absHierarchyList)]
        self.elements = {element : i for i, rcode in enumerate(self.rcodes) for element in rcode.elements}

        for rcode in self.rcodes:
            rcode.buildCode()


    def optimize(self, parameters = None):
        """ TODO: Do something more clever.
        """
        if self.hierarchy:
            # Get the heirarchy from biclustering hierarchy algorithm
            supersetsList, absHierarchyList = graphHierarchy(self.nonShadowElements, parameters)
            return self.useHierarchyStrategy(supersetsList, absHierarchyList)
        else:
            return self.optimizeVertexCuts(parameters = parameters)
        #return self.optimizeRecursiveHeavyHitters()


    def verifyCompression(self):
        for i, rcode in enumerate(self.rcodes):
            try:
                rcode.verifyCompression()
            except:
                raise Exception("Code %d failed to verify!" % i)
        print("MRSets verified successfully")


    def optimizeRecursiveHeavyHitters(self, threshold=1):
        while self.extractHeavyHitters(1):
            pass


    def optimizeVertexCuts(self, parameters = None):
        """ Threshold is the maximum size of a connected component we allow.
        """
        if parameters == None:
            threshold = 10
            index = -1
        else:
            threshold = parameters[0]
            index = parameters[1]

        while True:
            rcode = self.rcodes[-1]
            matrix = rcode.originalSets

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
            self.spawnSubCode(extractionSet=extractions)

        for rcode in self.rcodes:
            rcode.mergeOverlaps()


    def extractHeavyHitters(self, threshold=2, index=-1):
        """ Placeholder implementation. Not to be treated as an intelligent algorithm.
            Extracts all elements that appear in more than 'threshold' supersets.
        """
        # grab the target rcode
        rcode = self.rcodes[index]

        # find any elements in the code that occur more than twice
        occurrences = rcode.elementOccurrences()
        expensiveElements = [elem for elem in rcode.elements if occurrences[elem] > threshold]
        if len(expensiveElements) == 0: return False

        self.spawnSubCode(extractionSet=expensiveElements)

        return True


    def spawnSubCode(self, extractionSet, srcIndex=-1, dstIndex=None):
        """ Given a set of columns and two rcode indices, extract the columns from the src rset,
            and insert them into the dst rset.
        """
        # grab the src rcode
        rcode = self.rcodes[srcIndex]


        # remove them from the code, and then re-optimize that code
        submatrix = rcode.extract(extractionSet)
        if sum(len(row) for row in submatrix) == 0:
            raise Exception("Attempting to spawn a subcode from an extraction that yields an empty matrix!")

        rcode.removeSubsets()
        #rcode.optimizeWidth()
        rcode.removePadding()

        dstCode = None
        if dstIndex != None:
            # TODO: this
            raise NotImplemented
            # place the extractions in an existing rcode
            dstCode = self.rcodes[dstIndex]
        else:
            # place the removed elements into a new code
            dstCode = RCode(submatrix)
            # add the new code to the list of codes
            dstindex = len(self.rcodes)
            self.rcodes.append(dstCode)
        #dstCode.optimizeWidth()
        dstCode.removePadding()
        dstCode.buildCode()


        # update the location of the elements
        for element in extractionSet:
            self.elements[element] = dstIndex


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


    def matchStrings(self, elements=None, decorated=False):
        """ For every element in elements, return a list of of ternary strings which,
            when compared to a compressed tag, will determine if the respective
            element is present in the tag or not.
        """
        if elements == None:
            elements = set(self.elements.keys())
            elements.update(self.shadowElements.keys())
        inputElements = elements.copy()
        elements = set([self.shadowElements.get(e, e) for e in elements])

        separator, padChar = '', '*'
        if decorated:
            separator, padChar = '|', '*'
        strings = {}
        blankCodes = [padChar*rcode.widthUsed() for rcode in self.rcodes]
        for i, code in enumerate(self.rcodes):
            subset = elements.intersection(code.elements)
            substrings = code.allMatchStrings(subset, decorated=decorated)

            lPadding = separator.join(blankCodes[:i])
            rPadding = separator.join(blankCodes[i+1:])

            if i != 0:
                lPadding = lPadding + separator
            if i == len(blankCodes):
                rPadding = separator + rPadding


            for elem in subset:
                strings[elem] = [lPadding + substring + rPadding for substring in substrings[elem]]

        strings = {e:strings[self.shadowElements.get(e,e)] for e in inputElements}
        return strings




if __name__=="__main__":
    matrix = [[1,2,3,4],
              [3,4,5],
              [3,4,6,7]]

    cols = set()
    for row in matrix:
        cols.update(row)

    print("===================")
    code = MRCode(matrix)
    for row in matrix:
        print(row, "->", code.tagString(row, True))

    print("===================")
    code.extractHeavyHitters()
    for row in matrix:
        print(row, "->", code.tagString(row, True))

    print(code.matchStrings(cols))

    code = MRCode(matrix)
    code.optimize()
    code = MRCode(matrix)
    code.optimizeRecursiveHeavyHitters()
    code = MRCode(matrix)
    code.optimizeVertexCuts()

    print("===================")
    code = MRCode(matrix)
    code.useStrategy([[[3,4,5]], [[1,2], [6,7]]])
    for row in matrix:
        print(row, "->", code.tagString(row, True))

