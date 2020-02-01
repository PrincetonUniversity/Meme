import networkx as nx
import itertools
from typing import List,Set,Dict
try:
    from .RSets import RCode, SuperSet
    from .analyze import groupIdenticalColumns
    from .graphAlgorithm import graphHierarchy
    from .optimize import addCodeWords
except:
    from RSets import RCode, SuperSet
    from analyze import groupIdenticalColumns
    from graphAlgorithm import graphHierarchy
    from optimize import addCodeWords

def ternary_compare(str1, str2):
    if len(str1) != len(str2):
        raise Exception("strings of unequal length compared: %s and %s" %(str1, str2))
    for (c,d) in zip(str1,str2):
        # compare every pair of bits. failure short-circuits
        if not ((c=='*') or (d=='*') or (c==d)):
            return False
    return True


class MRCode(object):
    rcodes : List = None
    elements : Dict = None # maps elements to rcode indices
    originalSets : List = None

    shadowElements : Dict = None # maps elements to elements for which they have identical behavior
    hierarchy : bool = False
    nonShadowElements : Set = None
    extraBits : int = 0

    def __init__(self, elementSets, hierarchy = False, shadow = True, extraBits = 0):
        # remove duplicates
        elementSets = set([frozenset(es) for es in elementSets])

        self.originalSets = list(elementSets)
        self.hierarchy = hierarchy
        self.parameters = None
        self.extraBits = extraBits

        # find shadow elements
        self.shadowElements = {}
        if shadow:
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
        self.rcodes = [RCode(supersets, absHierarchy=absHierarchy, extraBits=self.extraBits+i) for i, supersets, absHierarchy 
                       in zip(range(len(supersetsList)),supersetsList, absHierarchyList)]
        self.elements = {element : i for i, rcode in enumerate(self.rcodes) for element in rcode.elements}

        for rcode in self.rcodes:
            rcode.buildCode()


    def optimize(self, parameters = None):
        """ TODO: Do something more clever.
        """
        self.parameters = parameters
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

    # A prototype update function when there is no sibling (shadow = False in init()) 
    # or ancestor (parameter[2] = False in optimize())
    def addSuperset(self, superset, parameters = None):
        if parameters != None: self.parameters = parameters
        elements = set(superset)
        self.nonShadowElements.add(frozenset(superset))
        selCols = [] 
        # For simplicity, enforce the number of rcodes to be the same as before, but it can be bigger if desired
        actualChanging = False

        for i, rcode in enumerate(self.rcodes):
            intersectingClusters = set([ss for ss in rcode.originalSets if len(elements.intersection(ss)) != 0])
            if len(intersectingClusters) == 1: 
                intersectingClusters = set()
            newCols = [set(row) for row in self.nonShadowElements if len(row.intersection(selCols)) != 0]
            if len(newCols) == 0:
                newCols = set()
            else:
                newCols = set.union(*newCols)

            intersectingClusters.update([ss for ss in rcode.originalSets if len(newCols.intersection(ss)) != 0])
            if len(intersectingClusters) == 0: 
                continue

            actualChanging = True
            for ss in intersectingClusters:
                rcode.originalSets.remove(ss)
            for ss in intersectingClusters:
                ssIndex = rcode.getSupersetIndex(ss)
                ssCodeword = rcode.supersets[ssIndex].codeword
                rcode.freeCodes.append(ssCodeword)
                del rcode.supersets[ssIndex]
            # merge free codes when possible
            rcode.freeCodes.sort(key=lambda item: (-len(item), item))
            mergeFound = True
            while mergeFound:
                mergeFound = False
                codeIndex = 0
                while True:
                    if codeIndex + 1 >= len(rcode.freeCodes):
                        break
                    if rcode.freeCodes[codeIndex][:-1] == rcode.freeCodes[codeIndex+1][:-1]:
                        newCode = rcode.freeCodes[codeIndex][:-1]
                        del rcode.freeCodes[codeIndex]
                        del rcode.freeCodes[codeIndex]
                        rcode.freeCodes.insert(codeIndex, newCode)
                        mergeFound = True
                    else:
                        codeIndex += 1
            changeCols = set.union(*[set(ss) for ss in intersectingClusters])
            rcode.elements = rcode.elements.difference(changeCols)
            for elem in changeCols:
                del rcode.elementWeights[elem]
            rcode.codeBuilt = False

            changeCols.update(selCols)
            frozenMatrix = set([frozenset(row.intersection(changeCols)) for row in self.nonShadowElements])
            if i == len(self.rcodes) - 1:
                supersetsList, _ = graphHierarchy(frozenMatrix, tuple(list(self.parameters[:3]) + [1, True]))
                selCols = []
            else:
                supersetsList, _ = graphHierarchy(frozenMatrix, tuple(list(self.parameters[:3]) + [2, False]))
                selCols = supersetsList[1][0]

            newFrozenSupersets = [frozenset(set(vares).union(abses)) for vares, abses in supersetsList[0]]
            
            rcode.originalSets.extend(newFrozenSupersets)
            changeCols = changeCols.difference(selCols)
            rcode.elements.update(changeCols)
            for elem in changeCols:
                rcode.elementWeights[elem] = 1

            #[SuperSet(ss, absoluteGiven = True) for ss in supersetsList[0]]
            newSupersets = [SuperSet(ss) for ss in newFrozenSupersets]
            
            codewordMap, rcode.freeCodes = addCodeWords(newFrozenSupersets, rcode.maxWidth+self.extraBits+i, rcode.freeCodes)
            supersetMap = {frozenset(superset.variables) : superset for superset in newSupersets}
            for k,v in codewordMap.items():
                supersetMap[k].codeword = v
            rcode.supersets.extend(newSupersets)
            rcode.codeBuilt = True

        print([rcode.widthUsed() for rcode in self.rcodes])
        return 1 if actualChanging else 0
            # expandIfNecessary


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
            if i != len(blankCodes) - 1:
                rPadding = separator + rPadding


            for elem in subset:
                strings[elem] = [lPadding + substring + rPadding for substring in substrings[elem]]

        strings = {e:strings[self.shadowElements.get(e,e)] for e in inputElements}
        return strings

    def elemIsInTag(self, elem, tag, queryDict):
        """ Slow. Essentially simulates a TCAM table.
        """
        for query in queryDict[elem]:
            if ternary_compare(query, tag):
                return True
        return False

    def setFromTag(self, tag, queryDict):
        """ Brute-force inverts an encoded set. Slow, should only be used for debugging.
        """
        results = set([elem for elem in self.elements if self.elemIsInTag(elem, tag, queryDict)]).union(
            [elem for elem in self.shadowElements if self.elemIsInTag(elem, tag, queryDict)])
        return results

    def verifyCompression(self):
        queryDict = self.matchStrings()
        for originalSet in self.originalSets:
            tag = self.tagString(originalSet)
            recovered = set(self.setFromTag(tag, queryDict))
            if recovered != set(originalSet):
                print("True row:", set(originalSet))
                print("Recovered row:", recovered)
                for col in set(originalSet):
                    print(col, self.matchStrings(elements=[col], decorated=True))
                print("Tag:", self.tagString(originalSet, decorated=True))
                raise Exception("A row recovered from the compression module was not equal to the original row!")
        print("Encoding verified successfully.")


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

