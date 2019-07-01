
try:
    from .analyze import bitsRequiredVariableID
except:
    from analyze import bitsRequiredVariableID
import copy

'''
    Node for the hierarchy of absolute columns and vairable columns
'''
class AbsNode:
    ownSupersets = None     # List of frozenset
    fullSupersets = None    # Set of frozenset (optional, can be None)
    absChildren = None      # List of AbsNodes
    absCol : int = 0
    height : int = 0
    ownPrefix : bool = False

    def __init__(self, absCol, supersets=None, posssibleChildren=None):
        self.absCol = absCol
        self.absChildren = []
        self.ownPrefix = False
        if supersets and posssibleChildren:
            self.fullSupersets = set(supersets)
            
            childSupersets = set([])
            for node in posssibleChildren:
                if node.fullSupersets.issubset(supersets):
                    childSupersets.update(node.fullSupersets)
                    self.absChildren.append(node)

            self.ownSupersets = list(self.fullSupersets.difference(childSupersets))
            self.height = bitsRequiredVariableID(self.getChildren())
        else:
            self.fullSupersets = None
            self.ownSupersets = []
            self.height = 0

    def addChild(self, child):
        self.absChildren.append(child)

    def addSuperset(self, superset):
        self.ownSupersets.append(frozenset(superset))

    def calculateHeight(self):
        self.height = bitsRequiredVariableID(self.getChildren())

    def getChildren(self):
        childrenList = self.ownSupersets + self.absChildren
        if self.ownPrefix:
            childrenList.append(frozenset([self.absCol]))
        return sorted(childrenList, key = lambda x: len(x))

    def __len__(self):
        return self.height

    def __str__(self):
        string = "absCol: " + str(self.absCol) + " "
        string +=  "absChildren: " + str([str(absNode) for absNode in self.absChildren]) + " "
        string +=  "ownSupersets: " + str(self.ownSupersets) + "    "
        string +=  "ownPrefix: " + str(self.ownPrefix) + "    "
        return string

    def checkPrefix(self, frozenMatrix, parentAbsCols = []):
        # If there is no child, the confusion is avoided due to the MASK
        if len(self.absChildren) == 0:
            self.calculateHeight()
            return []

        newparentAbsCols = copy.deepcopy(parentAbsCols)
        newparentAbsCols.append(self.absCol)

        if frozenset(newparentAbsCols) in frozenMatrix:
            self.ownPrefix = True
            result = [self.absCol] # for printing purpose
        else:
            result = [] # for printing purpose

        for node in self.absChildren:
            result.extend(node.checkPrefix(frozenMatrix, newparentAbsCols))

        self.calculateHeight()
        return result

    def getSupersetPairs(self, parentAbsCols = []):

        newparentAbsCols = copy.deepcopy(parentAbsCols)
        newparentAbsCols.append(self.absCol)
        result = [(superset, newparentAbsCols) for superset in self.ownSupersets]

        for node in self.absChildren:
            result.extend(node.getSupersetPairs(newparentAbsCols))
        return result

    def getAbsCount(self):
        return 1 + sum([node.getAbsCount() for node in self.absChildren])

    def getAbsCols(self):
        if not self.absChildren: return set([self.absCol])

        result = set.union(*[node.getAbsCols() for node in self.absChildren])
        result.add(self.absCol)
        return result

    def getAllSupersets(self):
        result = copy.deepcopy(self.ownSupersets)
        for node in self.absChildren:
            result.extend(node.getAllSupersets())
        return result
    
    def getAllCols(self):
        if len(self.ownSupersets) == 0:
            result = set([])
        else:
            result = set.union(*[set(superset) for superset in self.ownSupersets])
        result.add(self.absCol)

        for node in self.absChildren:
            result.update(node.getAllCols())
        
        return result
