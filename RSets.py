try:
    from .optimize import removeSubsets, minimizeVariableWidthGreedy, minimizeRulesGreedy, mergeIntersectingSets, generateCodeWords
    from .analyze import getSupersetIndex, bitsRequiredVariableID, ternary_compare
    from .biclustering import AbsNode
except:
    from optimize import removeSubsets, minimizeVariableWidthGreedy, minimizeRulesGreedy, mergeIntersectingSets, generateCodeWords
    from analyze import getSupersetIndex, bitsRequiredVariableID, ternary_compare
    from biclustering import AbsNode

import math
from bidict import bidict
from collections import deque as queue

from typing import List,Set,Dict


class SuperSet(object):
    codeword : str = None
    rowIDs : List[int] = None # the IDs of original matrix rows that were merged to produce this superset

    ordered : bool = False
    ordering  : Dict[int,int] = None # Maps elements to their ordering positions
    elements  : Set[int] = None # All elements in this superset
    absolutes : Set[int]  = None # elements that appeared in every row that was merged to produce this superset
    variables : Set[int]  = None # items that are not absolutes
    # TODO: for the unordered case, absolutes should only be assigned a position once converted to a variable

    def __init__(self, elements, firstRowID=None, ordering=None, absoluteGiven = False):
        if not absoluteGiven:
            self.elements = set(elements)
        else:
            self.elements = set(elements[0]).union(elements[1])

        self.ordered = (ordering != None)
        if self.ordered:
            self.ordering = {elem:ordering[elem] for elem in elements}
            # in the ordered superset case, we will see every subset of the elements, so nothing is absolute
            self.absolutes = set()
            self.variables = self.elements.copy()
        else:
            # order them arbitrarily
            # there is no ordering on absolutes (they arent in the mask), so initially the ordering is blank
            self.ordering = {}
            if not absoluteGiven:
                self.absolutes = self.elements.copy()
                self.variables = set()
            else:
                self.variables = set(elements[0])
                self.absolutes = set(elements[1])
                self.ordering = {elem:i for i,elem in enumerate(self.variables)}

        if firstRowID != None:
            self.rowIDs = [firstRowID]
        else:
            self.rowIDs = []


    def setAbsolutes(self, absolutes):
        self.absolutes = set(absolutes)
        self.variables = self.elements.difference(self.absolutes)
        if not self.ordered:
            self.ordering = {elem:i for i,elem in enumerate(self.variables)}


    def update(self, other):
        self.elements.update(other)

        if type(other) == type(self):
            if self.ordered != other.ordered:
                raise Exception("Attempting to merge an ordered superset with an unordered superset!")
            self.absolutes.intersection_update(other.absolutes)
            self.rowIDs.extend(other.rowIDs)
            if self.ordered:
                self.ordering.update(other.ordering)
        else:
            self.absolutes.intersection_update(other)

        oldVariables= self.variables.copy()
        self.variables = set(self.elements).difference(self.absolutes)

        if not self.ordered:
            # assign mask positions (after existing variables) to all new variables
            newVariables = self.variables.difference(oldVariables)
            firstNewPos = 0 if (len(self.ordering)==0) else max(self.ordering.values())
            self.ordering.update({newVar:firstNewPos+i for i,newVar in enumerate(newVariables)})

    def updateWouldGrowMask(self, other):
        """ If we were to merge the 'other' superset with this superset, would this delta mask expand?
        """
        if type(self) == type(other):
            newElements = self.elements.union(other.elements)
            newAbsolutes = self.absolutes.intersection(other.absolutes)
        else:
            newElements = self.elements.union(other)
            newAbsolutes = self.absolutes.intersection(other)
        newVariables = newElements.difference(newAbsolutes)
        return len(newVariables) > len(self.variables)


    def union(self, other):
        ss = self.copy()
        ss.update(other)
        return ss


    def __contains__(self, elem):
        return self.elements.__contains__(elem)

    def __iter__(self):
        return self.elements.__iter__()

    def __len__(self):
        return self.elements.__len__()

    def issuperset(self, other):
        return self.elements.issuperset(other)

    def issubset(self, other):
        return self.elements.issubset(other)

    def containsTagFor(self, subset):
        """ Determines if a tag for the given subset can be produced from this superset.
        """
        subset = set(subset)
        # All the absolutes must appear in the given subset
        if not self.absolutes.issubset(subset): return False
        varSubset = set(subset).difference(self.absolutes)
        # every non-absolute in the given subset must appear in our variables
        if not varSubset.issubset(self.variables): return False
        return True

    def mask(self, subset):
        """
        """
        subset = set(subset)
        sortedVars = list(self.variables)
        sortedVars.sort(key=lambda x:self.ordering[x])
        bits = ['1' if elem in subset else '0' for elem in sortedVars]
        # The mask is reversed because it grows from the end of tag towards the front
        return ''.join(reversed(bits))

    def queryMask(self, elem):
        """ What should the query mask be for this item
        """
        bits = ['*']*len(self.variables)
        if elem not in self.absolutes:
            sortedVars = list(self.variables)
            sortedVars.sort(key=lambda x:self.ordering[x])
            elemPos = sortedVars.index(elem)
            bits[elemPos] = '1'
            # in the ordered case, a query should only succeed if all preceding elements are absent
            if self.ordered:
                for i in range(elemPos):
                    bits[i] = '0'
        # The mask is reversed because it grows from the end of tag towards the front
        return ''.join(reversed(bits))

    def maskLen(self):
        return len(self.variables)


    def discard(self, elem):
        if elem in self.elements:
            del self.ordering[elem]
            self.elements.discard(elem)
            self.absolutes.discard(elem)
            self.variables.discard(item)

    def difference_update(self, elements):
        for elem in elements:
            self.ordering.pop(elem, None)
        self.elements.difference_update(elements)
        self.absolutes.difference_update(elements)
        self.variables.difference_update(elements)

    def copy(self):
        ss = SuperSet(self.elements)
        ss.codeword = self.codeword
        ss.rowIDs = self.rowIDs.copy()
        ss.ordered = self.ordered
        ss.ordering = self.ordering.copy()
        ss.absolutes = self.absolutes.copy()
        ss.variables = self.variables.copy()
        return ss





class RCode:
    """ Supersets is a list of list of elements. Example: [['A','B'], ['B','C']].
        elementOrdering is a dictionary mapping elements to indices in some global ordering.
            Example: {'A':0, 'B':1, 'C':2}. Only used if isOrdered is True
        codes is a list of prefix-free superset identifiers. codes[i] is the identifier of supersets[i]
        maxWidth is the maximum width our tags can use.
        elementWeights is a mapping of each element to how expensive it is. More expensive
            elements should appear in less sets.
        codeBuild is a boolean stating whether we have built the codes array or not.
        isOrdered is a boolean stating if we are encoding an ordered or unordered universe of elements.
        freeCodes is a list of binary strings.
    """
    originalSets    : List["frozenset"] = None
    elements        : Set = None    # all elements in the matrix
    supersets       : List["SuperSet"] = None
    elementOrdering : Dict = None
    maxWidth        : int  = 0
    elementWeights  : Dict = None
    codeBuilt       : bool = False
    isOrdered       : bool = False
    freeCodes       : List[str] = None
    extractions     : Set = None # elements that were removed from the input matrix
    logging         : bool = False

    absHierarchy : List = None # list of encoding
    absCodes : Dict = None # prefix coding for absolute elements
    emptyCode = ''

    # pep8 can go suck it


    def logger(self, *args):
        """ Placeholder logger.
        """
        if self.logging:
            print(' '.join(str(arg) for arg in args))


    def __init__(self, elementSets, maxWidth=None,
                    elementOrdering=None, elementWeights=None, logging=False, absHierarchy=None):
        """ elementSets preferably will have no duplicate sets.
            Duplicate sets do not impact correctness, only efficiency.
        """
        self.logging=logging

        self.absHierarchy = absHierarchy
        self.absCodes = {}
        self.emptyCode = ''

        # what elements appear across all the supersets?
        self.elements = set([])

        if not absHierarchy:
            for es in elementSets:
                self.elements.update(es)

            self.originalSets = [frozenset(es) for es in elementSets]
            # self.originalSets = bidict({i:s for i,s in enumerate(deduplicatedSets)})

            self.supersets = [SuperSet(elementSet, i) for (i,elementSet) in enumerate(self.originalSets)]
            self.removeSubsets()

        else:
            # don't know the purpose of originalSets for hierarchy case, SHOULD BE USELESS, maybe BUGGY
            self.originalSets = [frozenset(set(vares).union(abses)) for vares, abses in elementSets]
            for es in self.originalSets:
                self.elements.update(es)

            self.supersets = [SuperSet(elementSet, i, absoluteGiven = True) for (i, elementSet) in enumerate(elementSets)]


        if maxWidth is None:
            # assume the smallest possible max width if none given
            self.maxWidth = self.minimumWidth()
        else:
            self.maxWidth = maxWidth

        # ordering
        if elementOrdering is not None:
            self.isOrdered = True
            if len(self.elements.difference(elementOrdering.keys())) > 0:
                raise Exception("Ordered RCode was given an incomplete ordering!")
            self.elementOrdering = dict(elementOrdering)

        # weighting
        if elementWeights is not None:
            self.elementWeights = elementWeights.copy()
            if len(self.elements.difference(elementWeights.keys())) > 0:
                raise Exception("Weighted RCode was given an incomplete weighting!")
        else:
            self.elementWeights = {elem:1 for elem in self.elements}

    def groupingStrategy(self, groups):
        self.codeBuilt = False
        self.supersets = [SuperSet(g) for g in groups]
        originals = list(self.originalSets)
        while len(originals) > 0:
            subset = originals.pop()
            for superset in self.supersets:
                if superset.issuperset(subset):
                    superset.update(subset)
                    break
            else:
                raise Exception("The given grouping does not cover every input set!")

        self.expandIfNecessary()


    def elementOccurrences(self):
        """ Returns a dictionary mapping every element to how many supersets it occurs in.
        """
        occurrences = dict({elem : 0 for elem in self.elements})
        for superset in self.supersets:
            for elem in superset:
                occurrences[elem] += 1
        return occurrences



    def extract(self, elements):
        """ Remove the given elements'columns from the matrix, and return
            the removed columns as a submatrix
        """
        self.codeBuilt = False
        elements = set(elements)
        submatrix = [elements.intersection(row) for row in self.originalSets]
        for superset in self.supersets:
            superset.difference_update(elements)
        self.elements.difference_update(elements)
        return submatrix


    def removeSubsets(self):
        """ Removes all subsets from self.supersets
        """
        unsieved_supersets = self.supersets
        unsieved_supersets.sort(key=len, reverse=True)
        self.supersets = []
        i = 0
        while len(unsieved_supersets) > 0:
            sieve = unsieved_supersets[0] # the longest set is certainly not a subset
            self.supersets.append(sieve) # add the longest set to the final answer
            sieved_supersets = []

            for unsieved in unsieved_supersets[1:]:
                if not unsieved.issubset(sieve):
                    sieved_supersets.append(unsieved)
                else:
                    sieve.update(unsieved)
                    # reclaim the codeword if it exists
                    if self.codeBuilt:
                        if unsieved.codeword != None:
                            self.freeCodes.append(unsieved.codeword)

            unsieved_supersets = sieved_supersets


    def allMatchStrings(self, elements=None, decorated=False):
        """ Returns every match string in the encoding for the given elements. If the elements are [1,2,3],
            the return value will something look like {1:["0*1*"], 2:["0**1", "1*1*"], 3:["1**1"]}.
        """
        if not self.codeBuilt:
            self.buildCode()
        if elements == None:
            elements = self.elements
        return {element : self.matchStrings(element, decorated) for element in elements}


    def optimizeWidth(self):
        """ Attempts to minimize the number of bits this encoding will take.
            WARNING: wipes out an existing code!
        """
        self.codeBuilt = False
        startingWidth = self.minimumWidth()
        self.supersets = minimizeVariableWidthGreedy(self.supersets)
        endingWidth = self.minimumWidth()
        if startingWidth < endingWidth:
            raise Exception("Optimize width did not optimize width??")


    def removePadding(self):
        """ Set the maximum tag width to be the minimum possible.
        """
        if self.codeBuilt:
            self.maxWidth = self.widthUsed()
        else:
            self.maxWidth = self.minimumWidth()


    def expandIfNecessary(self):
        self.maxWidth = max(self.maxWidth, self.minimumWidth())


    def mergeOverlaps(self):
        """ If overlapping sets are unacceptable for an application, this will merge
            any supersets that have a nonempty intersection.
            WARNING: wipes out an existing code!
        """
        self.codeBuilt = False
        self.supersets = mergeIntersectingSets(self.supersets)
        self.expandIfNecessary()


    def optimizeMemory(self, padding = 0):
        """ Attempts to minimize the amount of dataplane memory this encoding will take.
            WARNING: wipes out an existing code!
        """
        self.codeBuilt = False
        self.supersets = minimizeRulesGreedy(self.supersets, self.elementWeights, self.maxWidth - padding)


    def memoryRequired(self):
        """ How many match strings will there be in the dataplane total?
        """
        if not self.codeBuilt:
            self.buildCode()
        return sum(sum(self.elementWeights[element] for element in superset) for superset in self.supersets)


    def memoryPerElement(self):
        """ Returns a dictionary where the keys are elements and the values are how many decoding strings are
            associated with each element.
        """
        if not self.codeBuilt:
            self.buildCode()
        memCounts = {element:0 for element in self.elementWeights.keys()}
        for superset in self.supersets:
            for element in superset:
                memCounts[element] += self.elementWeights[element]
        return memCounts


    def minimumWidth(self):
        if self.absHierarchy != None:
            return bitsRequiredVariableID(self.absHierarchy)
        return bitsRequiredVariableID([ss.variables for ss in self.supersets])


    def width(self):
        """ What is the actual width of the tags currently, including padding?
        """
        return self.maxWidth


    def widthUsed(self):
        """ Returns the maximum width of any tag (if there was no padding)
        """
        if not self.codeBuilt:
            self.buildCode()
        return max(len(superset.codeword) + superset.maskLen() for superset in self.supersets)


    def matchStrings(self, element, decorated=False, printD=False):
        """ Given an element, return a wildcard string for every superset that element appears in. This
            set of wildcard strings, when matched against a packet's tag, determines if that element is present.
        """
        if not self.codeBuilt:
            self.buildCode()


        separator, padChar = '', '*'
        if decorated:
            separator, padChar = '-', '~'

        if self.absHierarchy != None and element in self.absCodes:
            identifier = self.absCodes[element][0]
            paddingLen = self.maxWidth - (len(identifier))
            padding = padChar * paddingLen
            matchString = identifier + separator + padding
            return [matchString]

        strings = []
        for superset in self.supersets:
            if element not in superset:
                continue

            identifier = superset.codeword
            mask = superset.queryMask(element)

            paddingLen = self.maxWidth - (len(identifier) + len(mask))
            padding = padChar * paddingLen

            matchString = identifier + separator + padding + mask

            strings.append(matchString)
        if len(strings) == 0:
            self.logger("No match strings generated for element:", element)
        return strings


    def getSupersetIndex(self, elements):
        """ Return the index of the first superset that contains all the given elements.
            Returns -1 if no superset exists.
        """
        for i, superset in enumerate(self.supersets):
            if superset.containsTagFor(elements):
                return i
        return -1


    def tagString(self, elements, decorated=False):
        """ Given an element set, returns a binary string to be used as a packet tag.
            If decorated, padding bits are replaced with ~, and the identifier and mask are
            separated by a - character.
        """
        if not self.codeBuilt:
            self.buildCode()

        subset = set(elements)

        if self.absHierarchy != None and len(subset) == 0:
            identifier = self.emptyCode
            mask = ''
        elif self.absHierarchy != None and subset.issubset(self.absCodes.keys()):
            # print(self.absHierarchy)
            # print([(superset.variables, superset.absolutes) for superset in self.supersets])

            identifier = max([self.absCodes[element] for element in subset], key=lambda x : len(x[0]))[1]
            mask = ''
        else:
            ssIndex = self.getSupersetIndex(subset)
            if ssIndex == -1:
                self.logger("Asking for a tag for an unseen element set: ", elements)
                return ""

            superset = self.supersets[ssIndex]

            identifier = superset.codeword
            mask = superset.mask(subset)

        paddingLen = self.maxWidth - ((len(identifier) + len(mask)))
        padding = ('~' if decorated else '0') * paddingLen

        separator = '-' if decorated else ''

        return identifier + separator + padding + mask


    def buildCode(self):
        """ Rebuilds all identifiers. Sets codeBuilt to True.
        """
        self.logger("Building codewords... ")

        if self.absHierarchy != None:
            codewordMap, self.freeCodes, self.absCodes = generateCodeWords(self.absHierarchy, absHierarchy=True)
            supersetMap = {frozenset(superset.variables) : superset for superset in self.supersets}
            for k,v in codewordMap.items():
                if k == frozenset(["E"]):
                    self.emptyCode = v
                else:
                    supersetMap[k].codeword = v
        else:
            maskSets = [superset.variables for superset in self.supersets]
            codewords, self.freeCodes = generateCodeWords(maskSets)
            for codeword, superset in zip(codewords, self.supersets):
                superset.codeword = codeword

        self.codeBuilt = True
        self.logger("Done building codewords.")


    def _bestCodewordToSplit(self, newSet):
        """ You call this cause you need a new prefix-free set identifier. Goes through
            all the existing supersets and returns the index of the the identifier that,
            after splitting into two codewords, minimizes the decrease in padding size
            available for all tags. Returns -1 if no codeword can be split.
        """
        splitCosts = [min(self.maxWidth - (len(superset) + len(superset.codeword)+1),
                          self.maxWidth - (len(newSet)   + len(superset.codeword)+1))
                      for superset in self.supersets]

        bestSplitCost = max(splitCosts)
        if bestSplitCost < 0:
            return -1
        return splitCosts.index(bestSplitCost)



    def _bestSetToExpandUnordered(self, newSet):
        """ In the unordered case, given a new set we need to reproduce, returns the index
            of the superset whos expansion will have minimal impact on memory usage.
        """
        minIncrease = float("inf")
        bestIndex = -1
        for (i, superset) in enumerate(self.supersets):
            union = set(superset).union(newSet)
            if len(union) + len(superset.codeword) > self.maxWidth:
                continue
            newElements = union.difference(superset)
            increase = sum(self.elementWeights[element] for element in newElements)
            if increase < minIncrease:
                minIncrease = increase
                bestIndex = i

        return bestIndex


    def _bestSetToExpandOrdered(self, newSet):
        """ In the ordered case, given a new set we need to reproduce, returns the index
            of the superset whos expansion will have minimal impact on memory usage.
        """
        minIncrease = float("inf")
        bestIndex = -1
        for (i, superset) in enumerate(self.supersets):
            union = set(superset).union(newSet)
            if len(union) + len(self.codes[i]) > self.maxWidth:
                continue

            orderedUnion = self._orderSuperset(union)
            orderedMergeSet = self._orderSuperset(superset)
            if orderedUnion[-len(superset):] != orderedMergeSet:
                continue

            newElements = union.difference(superset)
            increase = sum(self.elementWeights[element] for element in newElements)
            if increase < minIncrease:
                minIncrease = increase
                bestIndex = i

        return bestIndex



    def addSet(self, newSet):
        """ Adds a new set to the encoding. Returns a list of changes that must be made to the dataplane,
            in the form of a list of new wildcard rules to install and old wildcard rules to remove.
            Example: code.addSet([1,2]) could return
                {"add":{1:['111*'], 2:['11*1'], 3:['101*']}, "remove":{3:[1*1*]}}
            The dictionary has two keys: "add" and "remove". Each key maps to a dictionary where the keys
            are elements and the values are associated wildcard strings which test for that element.
            Note that you may need to add or remove rules for elements which weren't in the new set!
            If there is a failure to add the set, returns None.
        """
        newSet = SuperSet(newSet)
        for element in newSet:
            if element not in self.elementWeights:
                self.elementWeights[element] = 1

        changes = {"add":{}, "remove":{}}
        if self.getSupersetIndex(newSet) != -1:
            self.logger("Adding a set which already exists in some superset:", newSet)
            return changes

        if not self.codeBuilt:
            self.supersets.append(newSet)
            self.codes.append("")
            return changes


        if not self.isOrdered:
            expandIndex = self._bestSetToExpandUnordered(newSet)
        else:
            expandIndex = self._bestSetToExpandOrdered(newSet)


        if expandIndex == -1:
            ## Code block for allocating a new superset with its own codeword
            splitIndex = self._bestCodewordToSplit(newSet)
            if splitIndex == -1:
                self.logger("Not enough tag space for adding set:", newSet)
                return None
            # split logic goes here
            changes["remove"] = self.allMatchStrings(self.supersets[splitIndex])
            splitCode = self.supersets[splitIndex].codeword
            self.supersets[splitIndex].codeword = splitCode + '0'

            newSet.codeword = splitCode + '1'
            self.supersets.append(newSet)

            changes["add"] = self.allMatchStrings(self.supersets[splitIndex])
            moreAddChanges = self.allMatchStrings(newSet)
            for element, strings in moreAddChanges.items():
                if element not in changes["add"]:
                    changes["add"][element] = []
                changes["add"][element].extend(strings)

        else:
            ## Code block for merging the new set into an existing superset
            # TODO: remove the need for changes["remove"]
            oldStrings = {elem:set(strings) for (elem,strings) in self.allMatchStrings(self.supersets[expandIndex]).items()}

            self.supersets[expandIndex].update(newSet)
            newStrings = self.allMatchStrings(self.supersets[expandIndex])

            deletedStrings = {}
            # remove strings that arent in the new string set
            for elem in oldStrings:
                deletedStrings[elem] = oldStrings[elem].difference(newStrings[elem])

            addedStrings = {}
            for elem in newStrings:
                if elem in oldStrings:
                    addedStrings[elem] = newStrings[elem].difference(oldStrings[elem])
                else:
                    addedStrings[elem] = newStrings[elem]

            changes["remove"] = deletedStrings
            changes["add"] = addedStrings

        return changes

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
        return set([elem for elem in self.elements if self.elemIsInTag(elem, tag, queryDict)])

    def verifyCompression(self):
        self.buildCode()

        queryDict = self.allMatchStrings()

        for originalSet in self.originalSets:
            tag = self.tagString(originalSet)
            recovered = set(self.setFromTag(tag, queryDict))
            if recovered != set(originalSet):
                print("True row:", set(originalSet))
                print("Recovered row:", recovered)
                print("Tag:", tag)
                raise Exception("A row recovered from the compression module was not equal to the original row!")
        print("Encoding verified successfully.")




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
    rcode = RCode(matrix)
    rcode.removePadding()
    rcode.validate()
    print("pre-width-optimization")
    for row in matrix:
        print(row, "has tag", rcode.tagString(row, True))
    print(rcode.allMatchStrings(rcode.elements))
    print("Post-width-optimization")
    rcode.optimizeWidth()
    rcode.removePadding()
    rcode.validate()
    print(rcode.supersets)
    for row in matrix:
        print(row, "has tag", rcode.tagString(row, True))

    rcode = RCode(matrix)
    rcode.validate()
    rcode.groupingStrategy([[1,2,3,4,5], [6,7,8]])
    for row in matrix:
        print(row, "has tag", rcode.tagString(row, True))
    rcode.validate()



if __name__=="__main__":
    unit_test()
