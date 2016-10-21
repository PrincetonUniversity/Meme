from optimize import removeSubsets, minimizeVariableWidthGreedy
from analyze import getSupersetIndex
import Queue
import math

class RCode:
    """ Supersets is a list of collections of elements. Example: [['A','B'], ['B','C']].
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
            self.ssIndex = -1

    # pep8 can go suck it

    def __init__(self, supersets, maxWidth, elementOrdering = {}, elementWeights = {}):
        """ Constructor. Ensures the input adheres to some assumptions, and corrects it if it does not.
        """
        self.supersets = removeSubsets(supersets)
        self.codes = ["" for _ in self.supersets]
        self.maxWidth = maxWidth
        self.codeTree = self._BTNode("")
        self.codeTree.addKids()

        # what elements appear across all the supersets?
        allElements = set([])
        for superset in self.supersets:
            allElements.update(superset)

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
        orderedSS.sort(key = lambda x : self.elementOrdering[x])
        return orderedSS


    def matchStrings(self, element):
        """ Given an element, return a wildcard string for every superset that element appears in. This
            set of wildcard strings, when matched against a packet's tag, determines if that element is present.
        """
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
        return strings


    def bitString(self, elements):
        """ Given an element set, returns a binary string to be used as a packet tag.
        """
        ssIndex = getSupersetIndex(elements, self.supersets)
        if ssIndex == -1:
            raise Exception("ASKING FOR TAG FOR UNSEEN SET YOU SHIRT. DON'T DO THAT")
        if not self.codeBuilt:
            self._buildCode()

        subset = set(elements)
        superset = self.supersets[ssIndex]

        mask = ''.join("1" if element in subset else "0" for element in superset)

        return self.codes[ssIndex] + mask

    def bitsUsed(self):
        """ Whats our current bit usage?
        """
        if not self.codeBuilt:
            self._buildCode()

        return len(self.supersets[0]) + len(self.codes[0])


    def _buildCode(self):
        """ Private method. Rebuilds everything. Sets codeBuilt to True.
        """
        self.supersets = removeSubsets(self.supersets)
        self.codes = [""] * len(self.supersets)

        # figure out what minimum width is needed for tags via solving Kraft's inequality
        kraftSum = sum(2**len(superset) for superset in self.supersets)
        neccessaryWidth = math.ceil(math.log(kraftSum, 2.0))

        # Make a list of pairs.
        # the first element in a pair is the index of a superset in the self.supersets list
        # the second element is how many bits are available for the superset's ID
        INDEX = 0
        IDWIDTH = 1
        unCodedSupersets = [(i, neccessaryWidth - len(self.supersets[i])) for i in xrange(len(self.supersets))]
        # sort it in descending order of available ID widths
        unCodedSupersets.sort(key = lambda x : x[IDWIDTH], reverse = True)

        # initialize a tree of codewords
        self.codeTree = self._BTNode("")
        self.codeTree.addKids()
        # add the first two codewords to a tree
        thisTreeLevel = [self.codeTree.left, self.codeTree.right]
        currCodeLen = 1

        # while there are supersets which have not been assigned codewords
        while len(unCodedSupersets) > 0:
            # if the current codeword tree is deep enough to assign some codes
            while unCodedSupersets[-1][IDWIDTH] == currCodeLen:
                treeNode = thisTreeLevel.pop()
                treeNode.assign(self, unCodedSupersets[-1][INDEX])
                unCodedSupersets.pop()

            # for every codeword which wasn't assigned to a superset, add children
            # children are codeword + "0" and codeword + "1"
            nextTreeLevel = []
            for treeNode in thisTreeLevel:
                treeNode.addKids()
                nextTreeLevel.extend([treeNode.left, treeNode.right])
            thisTreeLevel = nextTreeLevel
            currCodeLen += 1

        self.codeBuilt = True


    def _growTree(self):
        """ Private Method. Basically adds another bit to all the codewords.
            Does a tree traversal and adds children to any assigned codewords.
        """
        if not self.codeBuilt: raise Exception("Trying to grow a non-existent code tree?? Think again bozo")
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
                    treeNode.addKids()
                    treeNode.left.assign(self, treeNode.ssIndex)
                    treeNode.unassign(self)
                # if it is an interior node, explore its children
                elif treeNode.left != None:
                    nextTreeLevel.extend([treeNode.left, treeNode.right])
                # else we've hit an unassigned leaf, do nothing with it
                else: pass


    def addSet(self, newSet):
        """
            Returns True if the set was successfully added, and False otherwise.
        """
        if getSupersetIndex(newSet, self.supersets) != -1:
            return True

        if not self.codeBuilt:
            self.supersets.append(newSet)
            self.codes.append("")
            return True

        currWidth = self.tagWidth()
        excessBits = self.maxWidth - currWidth
        if excessBits == 0:
            return False


        minIncrease = float("inf")
        bestIndex = -1
        for (i, superset) in enumerate(self.supersets):
            union = superset.union(newSet)
            if len(union) - len(superset) > excessBits:
                continue
            newElements = union.difference(superset)
            increase = sum(self.ruleCounts[element] for element in newElements)
            if increase < minIncrease:
                minIncrease = increase
                bestIndex = i

        if bestIndex != -1:
            self.supersets[bestIndex].update(newSet)
            return True

        # if we hit here, no superset could be expanded, so we need a new one
        self.codeBuilt = False
        self.supersets.append(set(newSet))
        self.codes.append("")
        newIndex = len(self.codes) - 1

        kraftSum = sum(2**len(superset) for superset in self.supersets)
        kraftSum += 2**len(newSet)
        newTagWidth = math.ceil(math.log(kraftSum, 2.0))

        # if the current code tree doesn't have space for the new set, grow it bruh
        while newTagWidth > self.tagWidth():
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

        raise Exception("Couldn't find a free codeword but kraft said there was one. Wat")


