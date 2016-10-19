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
            self.left = RCode._BTNode(self.code + '0')
            self.right = RCode._BTNode(self.code + '1')

        def assign(self, rcode, index):
            self.ssIndex = index
            self.assigned = True
            rcode.codes[index] = self.code


    def __init__(self, supersets, maxWidth, ruleCounts = {}):
        self.supersets = [set(superset) for superset in supersets]
        self.codes = ["" for _ in supersets]
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


    def _buildCode(self):
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

        self.codeBuilt = True

