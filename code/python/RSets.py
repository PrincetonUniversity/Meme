from optimize import removeSubsets, minimizeVariableWidthGreedy
from analyze import getSupersetIndex, bitsRequiredVariableID
import math


LOGGING = "Sure why not"


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
    """
    supersets = []
    elementOrdering = {}
    codes = []
    maxWidth = 0
    elementWeights = {}
    codeBuilt = False
    isOrdered = False


    # pep8 can go suck it


    def logger(self, *args):
        """ Placeholder logger.
        """
        if LOGGING:
            print ' '.join(str(arg) for arg in args)


    def __init__(self, supersets, maxWidth, isOrdered = False, elementOrdering={}, elementWeights={}):
        """ Constructor. Ensures the input adheres to some assumptions, and corrects it if it does not.
        """
        self.supersets = [list(superset) for superset in removeSubsets(supersets)]
        self.maxWidth = maxWidth
        self.isOrdered = isOrdered

        # what elements appear across all the supersets?
        allElements = set([])
        [ allElements.update(superset) for superset in self.supersets ]

        if isOrdered:
            # what elements were we given an ordering for?
            orderedElements = set(elementOrdering.keys())
            maxIndex = 0 if len(elementOrdering) > 0 else max(index for index in elementOrdering.keys())

            # generate an ordering for any elements we weren't given an order for (could be none)
            unorderedElements = allElements.difference(orderedElements)
            newOrdering = {element : maxIndex + 1 + i for (i,element) in enumerate(unorderedElements)}

            # add the ordering we generated to the ordering we were given
            self.elementOrdering = elementOrdering.copy()
            self.elementOrdering.update(newOrdering)

        # what elements were we given weights for?
        weightedElements = set(elementWeights.keys())

        # generate default weights for all unweighted elements
        unweightedElements = allElements.difference(weightedElements)
        newWeights = {element : 1 for element in unweightedElements}

        # add the weights we generated to the weights we were given
        self.elementWeights = elementWeights.copy()
        self.elementWeights.update(newWeights)

    def allMatchStrings(self):
        """ Returns every match string in the encoding. If the elements are [1,2,3],
            the return value will look like {1:["0*1*"], 2:["0**1", "1*1*"], 3:["1**1"]}.
        """
        allElements = self.elementWeights.keys()
        allStrings = {}
        for element in allElements:
            allStrings[element] = self.matchStrings(element)

        return allStrings

    def allSupersets(self):
        """ Returns a dictionary where keys are codewords and values are supersets.
            Cool for debugging.
        """
        all = {}
        for (i,superset) in enumerate(self.supersets):
            if self.isOrdered:
                all[self.codes[i]] = self._orderSuperset(superset)
            else:
                all[self.codes[i]] = superset

        return all


    def optimizeWidth(self):
        """ Attempts to minimize the number of bits this encoding will take.
            WARNING: wipes out an existing code!
        """
        self.codeBuilt = False
        self.supersets = minimizeVariableWidthGreedy(self.supersets)


    def optimizeMemory(self):
        """ Attempts to minimize the amount of dataplane memory this encoding will take.
            WARNING: wipes out an existing code!
        """
        self.codeBuilt = False
        pass # TODO: this

    def memoryRequired(self):
        return sum(sum(self.elementWeights[element] for element in superset) for superset in self.supersets)


    def _orderSuperset(self, superset):
        """ Private method. Applies elementOrdering to the given superset and returns it in the correct order.
        """
        orderedSS = list(superset)
        if self.isOrdered:
            orderedSS.sort(key = lambda x : self.elementOrdering[x])
        return orderedSS


    def matchStrings(self, element):
        """ Given an element, return a wildcard string for every superset that element appears in. This
            set of wildcard strings, when matched against a packet's tag, determines if that element is present.
        """
        if not self.codeBuilt:
            self.buildCode()

        strings = []
        for (i, superset) in enumerate(self.supersets):
            if element not in superset:
                continue

            identifier = self.codes[i]
            mask = ['*'] * len(superset)

            orderedSS = self._orderSuperset(superset)

            mask[orderedSS.index(element)] = '1'
            mask = ''.join(mask)

            paddingLen = self.maxWidth - (len(identifier) + len(mask))
            padding = '*' * paddingLen

            matchString = identifier + padding + mask

            strings.append(matchString)
        if len(strings) == 0:
            self.logger("No match strings generated for element:", element)
        return strings


    def allSupersetStrings(self, superset):
        """ Given a superset, return every element match string associated with that superset.
            Example: for a superset [1,2] with codeword "01", this function will return:
            {1:["011*"], 2:["01*1"]}
        """
        if not self.codeBuilt:
            self.buildCode()

        if self.isOrdered:
            superset = set(superset)
        else:
            superset = list(superset)
        index = self.supersets.index(superset)
        if index == -1:
            self.logger("Cannot generate relevant match strings for a superset cause we can't find it:", superset)
            return {}
        if self.isOrdered:
            supereset = self._orderSuperset(superset)

        strings = {}

        identifier = self.codes[index]
        for (i, element) in enumerate(superset):
            mask = ['*'] * len(superset)
            mask[i] = '1'
            mask = ''.join(mask)

            paddingLen = self.maxWidth - (len(identifier) + len(mask))
            padding = '*' * paddingLen

            matchString = identifier + padding + mask
            strings.update({element:[matchString]})

        return strings



    def bitString(self, elements):
        """ Given an element set, returns a binary string to be used as a packet tag.
        """
        ssIndex = getSupersetIndex(elements, self.supersets)
        if ssIndex == -1:
            self.logger("Asking for a tag for an unseen element set: ", elements)
            return ""
        if not self.codeBuilt:
            self.buildCode()

        subset = set(elements)
        superset = self.supersets[ssIndex]

        identifier = self.codes[ssIndex]

        mask = ''.join("1" if element in subset else "0" for element in superset)

        paddingLen = self.maxWidth - ((len(identifier) + len(mask)))
        padding = '0' * paddingLen

        return identifier + padding + mask


    def kraftsInequality(self, base, depths):
        return sum(2**(base - depth) for depth in depths)


    def buildCode(self):
        """ Rebuilds all identifiers. Sets codeBuilt to True.
        """
        self.logger("Building encoding... ")
        self.supersets = removeSubsets(self.supersets)
        self.codes = [""] * len(self.supersets)

        minWidth = bitsRequiredVariableID(self.supersets)
        # indices of supersets that have no codes yet
        uncodedIndices = range(len(self.supersets))
        # sort it in descending order of available code widths
        uncodedIndices.sort(key = lambda index: minWidth - len(self.supersets[index]), reverse = True)
        codeLens = [minWidth - len(self.supersets[i]) for i in uncodedIndices]

        freeCodes = ['']
        currDepth = 0

        while len(uncodedIndices) > 0:
            if codeLens[-1] == currDepth:
                newCode = freeCodes.pop()
                index = uncodedIndices.pop()
                self.codes[index] = newCode
            elif self.kraftsInequality(currDepth, codeLens) < len(freeCodes)/2.0:
                currDepth += 1
            else:
                nextFreeCodes = []
                for freeCode in freeCodes:
                    nextFreeCodes.extend([freeCode + '0', freeCode + '1'])
                freeCodes = nextFreeCodes
                currDepth += 1

        if self.isOrdered:
            pass  # If we have a total ordering, set orderings are already decided
        # Otherwise, we need to order the supersets. Easy method: making them lists
        else:
            for (i, superset) in enumerate(self.supersets):
                self.supersets[i] = list(superset)

        self.codeBuilt = True
        self.logger("Done building encoding.")


    def _bestCodewordToSplit(self, newSet):
        """ You call this cause you need a new prefix-free set identifier. Goes through
            all the existing supersets and returns the index of the the identifier that,
            after splitting into two codewords, minimizes the decrease in padding size
            available for all tags. Returns -1 if no codeword can be split.
        """
        splitCosts = [min(self.maxWidth - (len(superset) + len(self.codes[i])+1),
                          self.maxWidth - (len(newSet)   + len(self.codes[i])+1))
                      for (i, superset) in enumerate(self.supersets)]

        bestSplitCost = max(splitCosts)
        if bestSplitCost < 0:
            return -1
        return splitCosts.index(bestSplitCost)



    def _bestSetToExpandUnordered(self, newSet):
        minIncrease = float("inf")
        bestIndex = -1
        for (i, superset) in enumerate(self.supersets):
            union = set(superset).union(newSet)
            if len(union) + len(self.codes[i]) > self.maxWidth:
                continue
            newElements = union.difference(superset)
            increase = sum(self.elementWeights[element] for element in newElements)
            if increase < minIncrease:
                minIncrease = increase
                bestIndex = i

        return bestIndex


    def _bestSetToExpandOrdered(self, newSet):
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
        newSet = list(newSet)
        for element in newSet:
            if element not in self.elementWeights:
                self.elementWeights[element] = 1

        changes = {"add":{}, "remove":{}}
        if getSupersetIndex(newSet, self.supersets) != -1:
            self.logger("Adding a set which already exists in some superset:", newSet)
            return changes

        if self.isOrdered:
            newSetOrdered = self._orderSuperset(newSet)
            if newSet != newSetOrdered:
                self.logger("Attempting to add a sequence which does not match the original ordering:", newSet)
                return None

        if not self.codeBuilt:
            self.supersets.append(newSet)
            self.codes.append("")
            return changes


        if not self.isOrdered:
            expandIndex = self._bestSetToExpandUnordered(newSet)
        else:
            expandIndex = self._bestSetToExpandOrdered(newSet)


        ## Code block for allocating a new superset with its own codeword
        if expandIndex == -1:
            splitIndex = self._bestCodewordToSplit(newSet)
            if splitIndex == -1:
                self.logger("Not enough tag space for adding set:", newSet)
                return None
            # split logic goes here
            changes["remove"] = self.allSupersetStrings(self.supersets[splitIndex])
            splitCode = self.codes[splitIndex]
            self.codes[splitIndex] = splitCode + '0'

            self.codes.append(splitCode + '1')
            self.supersets.append(newSet)

            changes["add"] = self.allSupersetStrings(self.supersets[splitIndex])
            moreAddChanges = self.allSupersetStrings(newSet)
            for element, strings in moreAddChanges.iteritems():
                if element not in changes["add"]:
                    changes["add"][element] = []
                changes["add"][element].extend(strings)
            return changes

        ## Code block for merging the new set into an existing superset
        oldSet = self.supersets[expandIndex]
        newElements = set(newSet).difference(oldSet)
        self.supersets[expandIndex] = list(newElements) + oldSet
        newStrings = self.allSupersetStrings(self.supersets[expandIndex])

        changes["add"] = {element:newStrings[element] for element in newElements}
        return changes

