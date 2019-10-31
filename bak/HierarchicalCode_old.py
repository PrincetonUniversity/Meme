from collections import deque as Queue
import copy
from typing import List, Dict, Set
from bidict import bidict

from common import *

from aho_corasick import AhoCorasick


from BaseCodes import ColumnID

from PathSets.analyze import groupOverlappingRows, transposeMatrix
#from bitsplit_dfa import BitSplit


def copyOnto(srcObj, dstObj):
    for key, val in srcObj.__dict__.items():
        dstObj.__dict__[key] = val


class HNode:
    columnID      : ColumnID = None     # implicit columnID. Can be none 
    childNodes : List[HNode]  = None    # List of HNodes children
    childSets  : List[Set[ColumnID]]    #
    subtreeHeight : int = -1            # height of the subtree rooted at this node. a singleton is height 1
    binary : bool = False               # is the subtree rooted at this node binary?
    producesTag : bool = False          # should this node produce a tag?
    producesPrefix : bool = False       # should this node produce a prefix?
    parent : HNode = None

    def __init__(self, parent=None, columnID=None, binary=False, isRoot=False):
        self.parent = parent
        self.columnID = columnID
        self.binary = binary
        self.isRoot = isRoot
        
        self.childNodes = []
        self.childSets = []


    def addChildSet(self, s : Set[ColumnID]):
        self.childSets.append(s)


    def getChildNode(self, columnID=None):
        """ Be wary that multiple children may match.
            In that case, the first match is returned.
        """
        for child in self.childNodes:
            if child.columnID == columnID:
                return child
        return None

    def addChildNode(self, columnID=None, **newChildKwargs):
        """ Unconditionally add a child with the given columnID.
        """
        child = HNode(parent=self, columnID=columnID, **newChildKwargs)
        self.children.append(child)
        return child

    def addOrGetChildNode(self, columnID, **newChildKwargs):
        """ Attempt to find and return a child that contains the given columnID.
            If no such child exists, a new one is created, adopted, and returned.
        """
        child = self.getChild(columnID)
        if child is None:
            child = self.addChild(columnID, **newChildKwargs)
        return child



    def condense(self) -> (List[HNode], List[Set[ColumnID]]):
        """ Make the children of this node only descendants that produce tags or prefixes, cutting out any middle nodes.
        """
        # only leaf nodes can have child sets
        # if a node has child nodes, it cannot have child sets post-condensation
        newChildNodes = []
        newChildSets = []
        for child in self.childNodes:
            # collect significant descendants
            grandchildNodes, grandchildSets = child.condense()
            newChildNodes.extend(grandchildNodes)
            newChildSets.extend(grandchildSets)

        if self.isRoot:
            self.childSets = newChildSets
            if len(newChildNodes) == 1:
                onlyChild = newChildNodes[0]
                self.childNodes = onlyChild.childNodes
                self.childSets.extend(onlyChild.childSets)
            else:
                self.childNodes = newChildNodes
                for childNode in self.childNodes:
                    childNode.parent = self


        elif self.columnID == None:
            # useless internal node is pruned, its child sets are passed up
            return (newChildNodes, newChildSets + self.childSets)

        else:
            # node lives on
            self.childNodes = newChildNodes
            for childNode in self.childNodes:
                childNode.parent = self
            self.childSets.extend(newChildSets)
            return ([self], [])

            


        elif len(self.childSets) != 0:
            if self.columnID == None:
                # child sets and nodes are passed up, this node dies
                pass
            else:
                # this node persists
                pass

        elif len(self.childSets) == 0:
            if self.columnID == None:
                pass
            else:
                pass


        if not (self.isRoot or self.producesPrefix):
            # non-prefixed internal nodes are pointless
            if self.producesTag:
                newChildren += [self]   # we live on, but as a leaf
                self.children = []
            return newChildren
        else:
            # our position is maintained in the condensed tree
            self.children = newChildren
            return [self]



    def makeBinary(self):
        """ If there are more than two children, they are pushed to the leaves
            of a new degree-2 subtree.
            Side-effect: sets self.subtreeHeight correctly
            Recursive.
        """
        self.binary = True

        if len(self.children) == 0:
            # leaf node edge case
            self.subtreeHeight = 1
            return
        if len(self.children) == 1:
            raise Exception("An internal tree node has only one child. This should not happen")

        for child in self.children:
            # first ensure that the child subtrees are binary
            # we also need the children to have the attribute subtreeHeight set
            child.makeBinary()

        # what is the shortest possible binary subtree we can construct?
        # determined via kraft's inequality
        heightRequired = sum(2**child.subtreeHeight for child in self.children)
        heightRequired = (heightRequired-1).bit_length()

        childDepths = [(child, heightRequired-child.subtreeHeight) for child in self.children]
        childDepths.sort(key=lambda tup:tup[1], reverse=True) # ordered from deep to shallow


        newChildren = [HNode(parent=self, columnID=None, binary=True), HNode(parent=self, columnID=None, binary=True)]
        freeLeaves = Queue([(child, 1) for child in newChildren])
        while len(childDepths) > 0:
            leaf, leafDepth = freeLeaves.pop()
            if (childDepths[-1][1] == leafDepth) or (len(childDepths) == len(freeLeaves)+1):
                oldChild, _ = childDepths.pop()
                copyOnto(oldChild, leaf)
            else:
                leaf.children = [HNode(None, binary=True), HNode(None, binary=True)]
                freeLeaves.extendleft([(leafChild, leafDepth+1) for leafChild in leaf.children])

        self.children = newChildren
        self.subtreeHeight = heightRequired


    def getTags(self, cumulative=True):
        if not self.binary:
            raise Exception("Attempting to retrieve binary tags from a non-binary subtree")
        ourTags = [('', self.outputIDs)] if self.producesTag else []

        if len(self.children) == 0:
            return ourTags

        leftChildTags = self.children[0].getTags(cumulative)
        rightChildTags = self.children[1].getTags(cumulative)

        extension = self.outputIDs if cumulative else []

        result = [('0' + tag, outputIDs+extension) for (tag,outputIDs) in  leftChildTags] \
               + [('1' + tag, outputIDs+extension) for (tag,outputIDs) in rightChildTags] \
               + ourTags

        return result



    def getMatchStrings(self, ourPrefix=''):
        if not self.binary:
            raise Exception("Attempting to retrieve match strings from a non-binary subtree")

        ourStrings = [(ourPrefix, out) for out in self.outputIDs] if self.producesPrefix else []

        if len(self.children) == 0:
            return ourStrings

        leftStrings  = self.children[0].getMatchStrings(ourPrefix + '0')
        rightStrings = self.children[1].getMatchStrings(ourPrefix + '1')

        return ourStrings + leftStrings + rightStrings

    def verifyInvariant(self, condensed=True, binary=False):
        if (not self.isRoot) and (self.parent == None):
            raise Exception("Non-root child has no assigned parent.")
        if len(self.children) == 0:
            if not self.producesTag:
                raise Exception("Leaf node that does not produce a tag found")
        elif condensed and (len(self.children) == 1):
            raise Exception("Internal node with 1 child found")
        elif binary and (len(self.children) not in [0,2]):
            raise Exception("Binary node with %d children found" % len(self.children))
        else:
            for child in self.children:
                child.verifyInvariant(condensed=condensed, binary=binary)





class HCode:
    root      : HNode = None        # Tree root
    made      : bool  = False       # Was the hierarchy built?
    cumulative    : bool = True     # If false, tree nodes will not contain the columnIDs of their ancestors
    cachedTags         : Dict[Set[ord], str]  = None    # Maps columnID sets to tags. Only valid if self.made
    cachedMatchStrings : Dict[ord, List[str]] = None    # Maps columnIDs to query string sets. Only valid if self.made


    def __init__(self, outputDict=None, charMask=0xff, matrix=None, ahocorasick=None, **kwargs) -> None:
        if outputDict != None:
            self.buildHierarchyFromDict(outputDict, charMask, **kwargs)
        elif matrix != None:
            self.buildHierarchyFromMatrix(matrix,             **kwargs)
        elif ahocorasick != None:
            self.buildHierarchyForCompactAC(ahocorasick,      **kwargs)


    def buildHierarchyFromMatrix(self, matrix : List[Set]):
        """ Build a tree hierarchy from a hierarchical collection of sets.
            Only really used for correctness and timing testing.
        """
        # remove duplicate rows
        matrix = [set(s) for s in set([frozenset(row) for row in matrix]) if len(s) > 0]

        self.root = HNode(columnID = None, isRoot=True)
        self.root.addChild(None).setOutputs([], True, True) # add the empty set node
        rowGroups = groupOverlappingRows(matrix)

        groupQueue = [(self.root, group) for group in groupOverlappingRows(matrix)]
        while len(groupQueue) > 0:
            parent, childMatrix = groupQueue.pop()
            absolutes = set.intersection(*childMatrix)

            for row in childMatrix: row.difference_update(absolutes)
            newChildMatrix = [row for row in childMatrix if len(row) > 0]

            isTerminal   = (len(newChildMatrix) < len(childMatrix))
            isInternal = (len(newChildMatrix) > 0)
            # Y internal, N terminal = N tag, Y prefix
            # Y internal, Y terminal = N tag, Y prefix
            # N internal, Y terminal = Y tag, Y prefix
            # N internal, N terminal = doesnt happen

            isTagged = isTerminal and not isInternal

            child = parent.addChild()
            child.setOutputs(absolutes, isTagged, True)

            if isTerminal and isInternal:
                nullGrandChild = child.addChild()
                nullGrandChild.setOutputs([], True, False)

            if len(newChildMatrix) > 0:
                grandChildMatrices = groupOverlappingRows(newChildMatrix)
                groupQueue.extend([(child, grandChildMatrix) for grandChildMatrix in grandChildMatrices])




    def buildHierarchyFromDict(self, outputDict : Dict[int, object], charMask : ord) -> None:
        """ Build a tree hierarchy for compression of partial match vectors of a bitsplit instance,
            via a reverse prefix tree of the words in the bitsplit instance.

            outputDict should be dfa.outputIDs
            charMask should be dfa.charMask
        """
        self.root = HNode(columnID=None, isRoot=True)

        branchingFactor = 2 ** len([c for c in bin(charMask) if c=='1'])

        # slice the words, find shadow columnIDs (ie equivalent slices)
        maskedOutputs = {}
        for outputID, output in outputDict.items():
            maskedOutput = bytes([char & charMask for char in output])
            if maskedOutput in maskedOutputs:
                maskedOutputs[maskedOutput].append(outputID)
            else:
                maskedOutputs[maskedOutput] = [outputID]


        # build the tree
        wordSlices = list(maskedOutputs.keys())
        # we add the longest words first, so internal nodes will be immediately identifiable
        wordSlices.sort(key=len, reverse=True)
        for wordSlice in wordSlices:
            outputIDs = maskedOutputs[wordSlice]
            reversedWordSlice = reversed(wordSlice)

            curr = self.root
            for char in reversedWordSlice:
                curr = curr.addOrGetChild(char)

            # Internal nodes should not produce tags
            # Their epsilon child will produce their tag instead
            if len(curr.children) == 0:
                curr.setOutputs(outputIDs, producesTag=True, producesPrefix=True)
            else:
                curr.setOutputs(outputIDs, producesTag=False, producesPrefix=True)
                child = curr.addChild(None)
                child.setOutputs([], producesTag=True, producesPrefix=False)

        # add the empty set node
        emptyNode = self.root.addChild(columnID=None)
        emptyNode.setOutputs([], producesTag=True, producesPrefix=True)



    def buildHierarchyForCompactAC(self, ac : AhoCorasick = None, words : List[str] = None, optDepth : int = -1) -> None:
        """ Given an AhoCorasick instance (or words for producing one), generate
            a state tagging and transition compressing encoding according to the CompactDFA paper.

            optDepth specifies the maximum depth for which states will have their transitions compressed.
            If optDepth is -1, the compression applies to the entire tree.
        """
        if ac == None:
            if words == None:
                raise Exception("buildHierarchyForCompactAC was given neither an AC nor a word list!")
            ac = AhoCorasick(words)

        self.cumulative = False

        self.root = HNode(columnID=None, isRoot=True)
        # the root corresponds to the empty label DFA state
        self.root.setOutputs([b""], producesTag=True, producesPrefix=True)

        # list of nodes that produce tags
        terminals = []
        terminals.append(self.root)

        # reverse prefix tree AKA suffix tree
        for word in ac.words:
            for end in range(1, len(word)+1):
                curr = self.root
                for char in reversed(word[:end]):
                    curr = curr.addOrGetChild(char)
                curr.setOutputs([word[:end]], producesTag=True, producesPrefix=True)
                terminals.append(curr)

        # nodes that are beyond the optimization depth do not get prefixes
        if optDepth >= 0:
            for terminal in terminals:
                label = terminal.outputIDs[0]
                if len(label) > optDepth:
                    terminal.producesPrefix = False

        # correct any internal tagged+prefixed nodes by having their tag delegated to an epsilon child
        for terminal in terminals:
            # all terminals are tagged, so we dont need to check terminal.producesTag
            if len(terminal.children) > 0 and terminal.producesPrefix:
                terminal.producesTag = False
                taggedChild = terminal.addChild(columnID=None)
                taggedChild.setOutputs(terminal.outputIDs, producesTag=True, producesPrefix=False)



    def printTraversal(self, title=""):
        printShellDivider("Traversal" + title)
        level = [(0, self.root)]
        i = 0
        while len(level) > 0:
            nextLevel = []
            levelStrings = []
            for nodeID, (parentID, node) in enumerate(level):
                chr1 = 'T' if node.producesTag else '.'
                chr2 = 'P' if node.producesPrefix else '.'
                levelStrings.append("%d->%d-%s%s%s-%dC" % (parentID, nodeID, str(node.outputIDs), chr1, chr2, len(node.children)))
                nextLevel.extend([(nodeID, child) for child in node.children])
            printAsColumns(levelStrings, "Level %d - %d Nodes" % (i, len(levelStrings)))
            i += 1
            level = nextLevel
        printShellDivider("End Traversal")


    def verifyInvariant(self, condensed : bool, binary : bool):
        """ For debugging. Checks that the tree adheres to an invariant.
            The invariant depends upon what operations have been performed on the tree.
            Set condensed = True if the tree has been condensed.
            Set binary = True if the tree has been made binary.
        """
        return self.root.verifyInvariant(condensed=condensed, binary=binary)

    def make(self, cache=True):
        if self.made: return
        self.verifyInvariant(condensed=False, binary=False)

        self.root.condense()
        self.verifyInvariant(condensed=True, binary=False)

        self.root.makeBinary()
        self.verifyInvariant(condensed=True, binary=True)

        self.made = True

        if cache:
            # calling these functions forces caching to execute
            self.tags()
            self.matchStrings()



    def tags(self, decorated=False, overrideCaching=False):
        """ Outputs a dictionary mapping rows (as frozensets) to hierarchial tags.
        """
        if not self.made: self.make()
        if self.cachedTags == None or overrideCaching:
            padding = '-' if decorated else '0'
            rawTags = self.root.getTags(self.cumulative)
            tagLen = self.root.subtreeHeight - 1
            self.cachedTags = {frozenset(outs):(tag + padding*(tagLen-len(tag))) for (tag,outs) in rawTags}
        return self.cachedTags



    def matchStrings(self, overrideCaching=False):
        """ Outputs a dictionary mapping column IDs to ternary match strings.
            For legacy reasons, the values are lists even though each columnID only has one match string
        """
        if not self.made: self.make()
        if self.cachedMatchStrings == None or overrideCaching:
            rawStrings = self.root.getMatchStrings()
            stringLen = self.root.subtreeHeight - 1
            self.cachedMatchStrings = {out:[prefix + '*'*(stringLen-len(prefix))] for prefix,out in rawStrings}
        return self.cachedMatchStrings



    def width(self):
        """ Returns the bits required for a codeword or prefix
        """
        if not self.made: self.make()

        return self.root.subtreeHeight - 1



def test0():
    words = bidict({1:b'cba', 2:b'a', 3:b'da', 4:b'ea'})

    hcode = HCode(words, 0xff)

    hcode.printTraversal()

    hcode.root.condense()

    hcode.printTraversal()

    hcode.make()

    hcode.printTraversal()

    print("========TAGS===========")
    print(hcode.tags())
    print("==========STRINGS=========")
    print(hcode.matchStrings())
    print("======================")




def test1():
    words = [bytes(s, 'utf8') for s in ["abc", "a", "ad", "ae"]]
    from aho_corasick import AhoCorasick
    from bitsplit_dfa import BitSplit

    bs = BitSplit(AhoCorasick(words))

    for dfaID, dfa in enumerate(bs.dfas):
        print("====== DFA %d ======" % dfaID)
        hcode = HCode(dfa.outputIDs, dfa.charMask)
        hcode.printTraversal()
        hcode.make()
        hcode.printTraversal()
        tags = hcode.tags()
        strings = hcode.matchStrings()
        print(tags)
        print(strings)
        for stateNode in dfa.states.values():
            if frozenset(stateNode.outputIDs) not in tags:
                print("Missing row is", stateNode.outputIDs)
                print("Present rows are", tags.keys())
                raise Exception("A row was not in the tag dictionary!")

        verifyCompression(tags, strings)


def test2():
    words = [bytes(s, 'utf8') for s in ["abc", "a", "ad", "ae"]]
    from aho_corasick import AhoCorasick
    from bitsplit_dfa import BitSplit

    bs = BitSplit(AhoCorasick(words))

    matrices = [dfaStates.values() for dfaStates,dfaTransitions in bs.allDfasAsTuples()]


    for matrixID, matrix in enumerate(matrices):
        print("====== DFA %d ======" % matrixID)

        hcode = HCode(matrix=matrix)
        #hcode.printTraversal()
        hcode.make()
        #hcode.printTraversal()
        tags = hcode.tags()
        strings = hcode.matchStrings()
        print(tags)
        print(strings)
        for row in matrix:
            if frozenset(row) not in tags:
                print("Missing row is", row)
                print("Present rows are", tags.keys())
                raise Exception("A row was not in the tag dictionary!")

        verifyCompression(tags, strings)


def test3():
    bs = getStandardBS()
    for dfa in bs.dfas:
        matrix = [set(stateNode.outputIDs) for stateNode in dfa.states.values()]

        hcode = HCode(matrix=matrix)
        hcode.make()
        tags = hcode.tags()
        strings = hcode.matchStrings()
        print("Code width is %d" % hcode.width())
        for stateNode in dfa.states.values():
            if frozenset(stateNode.outputIDs) not in tags:
                print("Missing row is", stateNode.outputIDs)
                print("Present rows are", tags.keys())
                raise Exception("A row was not in the tag dictionary!")

        verifyCompression(tags, strings)


def test4():
    bs = getStandardBS(overwrite=True)
    for dfa in bs.dfas:
        hcode = HCode(dfa.outputIDs, dfa.charMask)

        hcode.make()
        tags = hcode.tags()
        strings = hcode.matchStrings()
        print("Code width is %d" % hcode.width())
        for stateNode in dfa.states.values():
            if frozenset(stateNode.outputIDs) not in tags:
                print("Missing row is", stateNode.outputIDs)
                print("Present rows are", tags.keys())
                raise Exception("A row was not in the tag dictionary!")

        verifyCompression(tags, strings)



def testTiming():
    bs = getCustomBS(getCustomAC(getCustomWords()))
    print("%d words in BS instance" % len(bs.outputIDs))
    matrices = [[set(stateNode.outputIDs) for stateNode in dfa.states.values()] for dfa in bs.dfas]
    printTimer()
    print("Building HCodes from matrices")
    for matrix in matrices:
        hcode = HCode(matrix=matrix)
        tags = hcode.tags()
        strings = hcode.matchStrings()
    print("Done.")
    printTimer()
    print("Building HCodes from words")
    for dfa in bs.dfas:
        hcode = HCode(dfa.outputIDs, dfa.charMask)
        tags = hcode.tags()
        strings = hcode.matchStrings()
    print("Done.")
    printTimer()


def testCondense():
    words = [bytes(s, 'utf8') for s in ["abc", "a", "ad", "ae"]]
    from aho_corasick import AhoCorasick
    from bitsplit_dfa import BitSplit

    bs = BitSplit(AhoCorasick(words))
    for dfa in bs.dfas:
        matrix = [set(stateNode.outputIDs) for stateNode in dfa.states.values()]

        hcode = HCode(dfa.outputIDs, dfa.charMask)
        hcode.printTraversal()
        hcode.root.condense()
        hcode.printTraversal()
        break



def testCompactAC():
    from aho_corasick import AhoCorasick
    #words = [bytes(s, 'utf8') for s in ["abc", "a", "ad", "ae"]]
    words = [b'aaa', b'aba', b'bad', b'aaaaa', b'badad', b'dad', b'a', b'b', b'aa']

    ac = AhoCorasick(words)
    hcode = HCode(ahocorasick=ac, optDepth=2)
    hcode.printTraversal(" initial")
    hcode.verifyInvariant(condensed=False, binary=False)
    hcode.root.condense()
    hcode.printTraversal(" condensed")
    hcode.verifyInvariant(condensed=True, binary=False)
    #return
    #hcode.root.condense()
    #hcode.printTraversal()
    hcode.make()
    hcode.printTraversal(" binary")

    stateTags = {list(key)[0] : val for key,val in hcode.tags().items()}
    suffixPrefixes = {key:val[0] for key,val in hcode.matchStrings().items()}
    print(stateTags)
    print(suffixPrefixes)
    return

    charTables = {i:[] for i in range(256)}
    rules = {}

    label2acnode = {state.label : state for state in ac.states.values()}
    tag2label = {}
    rootTag = stateTags[b""]

    for label, tag in stateTags.items():
        priorStateString = suffixPrefixes[label[:-1]]
        if len(label) > 0:
            inChar = label[-1]
            charTables[inChar].append(priorStateString)
            rules[(inChar, priorStateString)] = tag
        tag2label[tag] = label

    inputText = b"abcacdeabcdef"

    curr1 = rootTag
    curr2 = ac.rootNode()
    for char in inputText:
        lpmTable = charTables[char]
        matchIndex = longest_prefix_match(lpmTable, curr1)
        if matchIndex == -1:
            curr1 = rootTag
        else:
            curr1 = rules[(char, lpmTable[matchIndex])]
        curr2 = ac.nextNode(curr2, char)

        if tag2label[curr1] != curr2.label:
            print("Current tag is", curr1, "Current label is", tag2label[curr1])
            print("Current AC label is", curr2.label)
            raise Exception("Didn't work")
    print("worked")




if __name__ == "__main__":
    #testTiming()
    #test4()
    testCompactAC()
