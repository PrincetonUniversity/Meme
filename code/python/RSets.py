from optimize import removeSubsets, minimizeVariableWidthGreedy
from analyze import getSupersetIndex
import math

LOGGING = "Sure why not"


class RCode:
    """ Supersets is a list of collections of elements. Example: [['A','B'], ['B','C']]
                                                         OR [set('A','B'), set('B','C').
        elementOrdering is a dictionary mapping elements to indices in some global ordering.
            Example: {'A':0, 'B':1, 'C':2}.
        codes is a list of prefix-free superset identifiers. codes[i] is the identifier of supersets[i]
        maxWidth is the maximum width our tags can use.
        elementWeights is a mapping of each element to how expensive it is. More expensive
            elements should appear in less sets.
        codeBuild is a boolean stating whether we have built the codes array or not.
        codeTree is a reference to the root of the prefix-free code tree.
        isOrdered is a boolean stating if we are encoding an ordered or unordered universe of elements.
    """
    supersets = []
    elementOrdering = {}
    codes = []
    maxWidth = 0
    elementWeights = {}
    codeBuilt = False
    codeTree = None  # type == _BTNode
    isOrdered = False


    class _BTNode:
        """ Private Binary Tree Node class for constructing prefix-free codes.
            Private!! Donut Use!!!!!!
        """
        left = None  # type == _BTNode
        right = None  # type == _BTNode
        code = ""  # binary string
        codeLen = 0  # literally just len(code). don't laugh
        assigned = False  # is any superset using this codeword?
        ssIndex = -1  # which superset is using this codeword?
        needsUpdate = False  # is this a grown code with no sibling codes?

        def __init__(self, code):
            self.code = code
            self.codeLen = len(code)

        def addKids(self):
            self.left = RCode._BTNode(self.code + '0')
            self.right = RCode._BTNode(self.code + '1')

        def assign(self, rcode, index):
            self.assigned = True
            rcode.codes[index] = self.code
            self.ssIndex = index

        def unassign(self, rcode):
            self.assigned = False
            rcode.codes[self.ssIndex] = ""
            self.ssIndex = -1

    # pep8 can go suck it


    def logger(self, *args):
        """ Placeholder logger.
        """
        if LOGGING:
            print ' '.join(str(arg) for arg in args)

    def __init__(self, supersets, maxWidth, isOrdered = False, elementOrdering={}, elementWeights={}):
        """ Constructor. Ensures the input adheres to some assumptions, and corrects it if it does not.
        """
        self.supersets = removeSubsets(supersets)
        self.codes = ["" for _ in self.supersets]
        self.maxWidth = maxWidth
        self.codeTree = self._BTNode("")
        self.codeTree.addKids()
        self.isOrdered = isOrdered

        # what elements appear across all the supersets?
        allElements = set([])
        for superset in self.supersets:
            allElements.update(superset)

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


    def optimizeWidth(self):
        """ Attempts to minimize the number of bits this encoding will take. WARNING: creates code from scratch!
        """
        self.codeBuilt = False
        self.supersets = minimizeVariableWidthGreedy(self.supersets)


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


    def bitsUsed(self):
        """ Whats our current bit usage?
        """
        if not self.codeBuilt:
            self.buildCode()

        return len(self.supersets[0]) + len(self.codes[0])


    def kraftsInequality(self, base, depths):
        return sum(2**(base - depth) for depth in depths)


    def buildCode(self):
        """ Rebuilds everything. Sets codeBuilt to True.
        """
        self.logger("Building encoding... ")
        self.supersets = removeSubsets(self.supersets)
        self.codes = [""] * len(self.supersets)

        # figure out what minimum width is needed for tags via solving Kraft's inequality
        kraftSum = sum(2**len(superset) for superset in self.supersets)
        minWidth = math.ceil(math.log(kraftSum, 2.0))
        # indices of supersets that have no codes yet
        uncodedIndices = range(len(self.supersets))
        # sort it in descending order of available code widths
        uncodedIndices.sort(key = lambda index: minWidth - len(self.supersets[index]), reverse = True)
        codeLens = [minWidth - len(self.supersets[i]) for i in uncodedIndices]

        # initialize a tree of codewords
        self.codeTree = self._BTNode("")
        self.codeTree.addKids()
        # add the first two codewords to a tree
        thisTreeLevel = [self.codeTree.left, self.codeTree.right]
        currDepth = 1

        # while there are supersets which have not been assigned codewords
        while len(uncodedIndices) > 0:
            # if the current codeword tree is deep enough to assign some codes
            if codeLens[-1] == currDepth:
                treeNode = thisTreeLevel.pop()
                treeNode.assign(self, uncodedIndices[-1])
                uncodedIndices.pop()
                codeLens.pop()
            elif self.kraftsInequality(currDepth, codeLens) <= len(thisTreeLevel)/2.0:
                currDepth += 1
            else:
                # Our tree needs to grow to fit the remaining codes, so we add children
                # to each unallocated codeword. Children are codeword + "0" and codeword + "1".
                nextTreeLevel = []
                for treeNode in thisTreeLevel:
                    treeNode.addKids()
                    nextTreeLevel.extend([treeNode.left, treeNode.right])
                thisTreeLevel = nextTreeLevel
                currDepth += 1

        if self.isOrdered:
            pass  # If we have a total ordering, set orderings are already decided
        # Otherwise, we need to order the supersets. Easy method: making them lists
        else:
            for (i, superset) in enumerate(self.supersets):
                self.supersets[i] = list(superset)

        self.codeBuilt = True
        self.logger("Done building encoding.")


    def _growTree(self):
        """ Private Method. Basically adds another bit to all the codewords.
            Does a tree traversal and adds children to any assigned codewords.
        """
        if not self.codeBuilt:
            self.logger("Trying to grow a non-existent code tree.")
            return
        thisTreeLevel = [self.codeTree.left, self.codeTree.right]
        while len(thisTreeLevel) > 1:
            nextTreeLevel = []
            for treeNode in thisTreeLevel:
                # We've found an assigned node. This node corresponds to a codeword,
                # say "101" as an example, which is the ID of some superset. What we want is to add
                # a "0" to the right side of that superset's ID. The way we do this is by
                # adding children to this assigned node, which will be codes "1010" and "1011".
                # We then unassign this node from being the codeword for its superset, and instead
                # assign its left child, which is the codeword "0101", to the superset.
                if treeNode.assigned:
                    ssIndex = treeNode.ssIndex
                    treeNode.unassign(self)
                    treeNode.addKids()
                    treeNode.left.assign(self, ssIndex)
                # if it is an interior node, explore its children
                elif treeNode.left != None:
                    nextTreeLevel.extend([treeNode.left, treeNode.right])
                # else we've hit an unassigned leaf, do nothing with it
                else: pass

    """
    def _addSuperset(self, superset):
        self.codeBuilt = False

        if self.isOrdered:
            self.supersets.append(list(superset))
        else:
            self.supersets.append(set(superset))
        self.codes.append("")
        newIndex = len(self.codes) - 1

        kraftSum = sum(2**len(superset) for superset in self.supersets)
        oldTagWidth = math.ceil(math.log(kraftSum, 2.0))
        kraftSum += 2**len(newSet)
        newTagWidth = math.ceil(math.log(kraftSum, 2.0))

        # if the current code tree doesn't have space for the new set, grow it bruh
        while newTagWidth > self.bitsUsed():
            self._growTree()

        goalCodeLen = newTagWidth - len(newSet)

        thisTreeLevel = [self.codeTree.left, self.codeTree.right]
        currCodeLen = 1

        while(len(thisTreeLevel) > 0):
            if currCodeLen == goalCodeLen:
                treeNode = thisTreeLevel.pop()
                treeNode.assign(self, newIndex)
                self.codeBuilt = True
                return True
            else:
                nextTreeLevel = []
                for treeNode in thisTreeLevel:
                    if treeNode.assigned:
                        continue
                    if treeNode.left == None:
                        treeNode.addKids()
                        nextTreeLevel.append(treeNode.left)
                    else:
                        nextTreeLevel.extend([treeNode.left, treeNode.right])
                thisTreeLevel = nextTreeLevel
                currCodeLen += 1

        self.logger("Couldn't find a free codeword but kraft said there was one. That's a bug.")
        return False

    def _addSetUnordered(self, superset):
        return False

    def _addSetOrdered(self, superset):
        return False
    """


    def addSet(self, newSet):
        """ Adds a new set to the encoding. Returns a list of changes that must be made to the dataplane,
            in the form of a list of new wildcard rules to install and old wildcard rules to remove.
            Example: code.addSet([1,2]) could return
                {"add":{1:'111*', 2:'11*1', 3:['101*']}, "remove":{3:[1*1*]}}
            The dictionary has two keys: "add" and "remove". Each key maps to a dictionary where the keys
            are elements and the values are associated wildcard strings which test for that element.
            Note that you may need to add or remove rules for elements which weren't in the new set!
            If there is a failure to add the set, returns None.
        """
        """
        changes = {"add":[], "remove":[]}
        if getSupersetIndex(newSet, self.supersets) != -1:
            self.logger("Adding a set which already exists in some superset!")
            return changes

        if not self.codeBuilt:
            self.supersets.append(newSet)
            self.codes.append("")
            return changes

        currWidth = self.bitsUsed()
        excessBits = self.maxWidth - currWidth
        if excessBits == 0:
            self.logger("Zero bits available for adding a set.")
            return None

        if not self.isOrdered:
            return self._addSetUnordered(newSet)
        else:
            return self._addSetOrdered(newSet)

        minIncrease = float("inf")
        bestIndex = -1
        for (i, superset) in enumerate(self.supersets):
            union = set(superset).union(newSet)
            if len(union) - len(superset) > excessBits:
                continue
            newElements = union.difference(superset)
            increase = sum(self.elementWeights[element] for element in newElements)
            if increase < minIncrease:
                minIncrease = increase
                bestIndex = i

        if bestIndex != -1:
            self.supersets[bestIndex].update(newSet)
            return True

        else:
            return _addSuperset(newSet)

        # if we hit here, no superset could be expanded, so we need a new one

        """
        return False




