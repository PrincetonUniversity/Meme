from optimize import removeSubsets, minimizeVariableWidthGreedy
from analyze import getSupersetIndex
import Queue
import math

class RCode:
    supersets = []
    codes = []
    maxWidth = 0
    ruleCounts = {}
    codeBuilt = False
    codeTree = None # type == _BTNode

    class _BTNode:
        left = None # type == _BTNode
        right = None # type == _BTNode
        code = "" # binary string
        codeLen = 0
        assigned = False
        ssIndex = -1

        def __init__(self, code):
            self.code = code
            self.codeLen = len(code)

        def addKids(self):
            self.left = RCode._BTNode('0' + self.code)
            self.right = RCode._BTNode('1' + self.code)

        def assign(self, rcode, index):
            self.assigned = True
            rcode.codes[index] = self.code
            self.ssIndex = index

        def unassign(self, rcode):
            self.assigned = False
            self.ssIndex = -1


    def __init__(self, supersets, maxWidth, ruleCounts = {}):
        self.supersets = removeSubsets(supersets)
        self.codes = ["" for _ in self.supersets]
        self.maxWidth = maxWidth
        self.ruleCounts = ruleCounts
        self.codeTree = self._BTNode("")
        self.codeTree.addKids()


    def optimizeWidth(self):
        self.codeBuilt = False
        self.supersets = minimizeVariableWidthGreedy(self.supersets)


    def bitString(self, elements):
        ssIndex = getSupersetIndex(elements, self.supersets)
        if ssIndex == -1:
            raise Exception("ASKING FOR TAG FOR UNSEEN SET YOU SHIRT")
        if not self.codeBuilt:
            self._buildCode()

        subset = set(elements)
        superset = self.supersets[ssIndex]

        mask = ''.join("1" if element in subset else "0" for element in superset)

        return self.codes[ssIndex] + mask

    def tagWidth(self):
        if not self.codeBuilt:
            self._buildCode()

        return len(supersets[0]) + len(codes[0])


    def _buildCode(self):
        self.supersets = removeSubsets(self.supersets)
        self.codes = [""] * len(self.supersets)

        # figure out what minimum width is needed for tags via solving Kraft's inequality
        kraftSum = sum(2**len(superset) for superset in self.supersets)
        tagWidth = math.ceil(math.log(kraftSum, 2.0))

        # Make a list of pairs.
        # the first element in a pair is the index of a superset in the self.supersets list
        # the second element is how many bits are available for the superset's ID
        INDEX = 0
        IDWIDTH = 1
        unCodedSupersets = [(i, tagWidth - len(self.supersets[i])) for i in xrange(len(self.supersets))]
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
        if not self.codeBuilt: raise Exception("Trying to grow a non-existent code tree??")
        thisTreeLevel = [self.codeTree.left, self.codeTree.right]
        while len(thisTreeLevel) > 1:
            nextTreeLevel = []
            for treeNode in thisTreeLevel:
                # We've found an assigned node. This node corresponds to a codeword,
                # say "101" as an example, which is the ID of some superset. What we want is to add
                # a "0" to the left side of that superset's ID. The way we do this is by
                # adding children to this assigned node, which will be codes "0101" and "1101".
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


    # returns True if the set was successfully added, and False otherwise
    def addSet(self, newSet):
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


