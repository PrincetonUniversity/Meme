from BaseCodes import BaseCodeStatic
from common import generateIdentifiers, kraftsInequality
from optimize import removeSubsets, minimizeRulesGreedy, minimizeVariableWidthGreedy

# memory requirements are
#   sum( sum(tagWidth + overhead for overhead in ruleOverheads[elem])*occurrences[elem] for elem in elements)

# compare the minimum possible tag size that the schemes can achieve
# then, compare the minimum rule inflation the schemes achieve below different field widths: 16, 32, 48, 64 bits


class OriginalCodeStatic(BaseCodeStatic):
    supersets = []
    identifiers = []
    maxWidth = -1


    def make(self, maxWidth=-1):
        self.supersets = removeSubsets(self.originalRows)
        self.optimizeWidth()

        self.maxWidth = maxWidth
        if maxWidth != -1:
            self.optimizeMemory(maxWidth)

        self.buildIdentifiers()

        self.made = True


    def buildIdentifiers(self):
        self.identifiers, freeIDs = generateIdentifiers([len(ss) for ss in self.supersets])


    def width(self, includePadding : bool = True):
        if (not self.made) and includePadding and self.maxWidth != -1:
            return self.maxWidth
        return kraftsInequality([len(superset) for superset in self.supersets])


    def optimizeWidth(self):
        """ Attempts to minimize the number of bits this encoding will take.
            WARNING: wipes out an existing code!
        """
        self.codeBuilt = False
        startingWidth = self.width(includePadding=False)
        self.supersets = minimizeVariableWidthGreedy(self.supersets)
        endingWidth = self.width(includePadding=False)

        if startingWidth < endingWidth:
            raise Exception("Optimize width did not optimize width?? Went from %d to %d" % (startingWidth, endingWidth))


    def optimizeMemory(self, maxWidth):
        if maxWidth < self.width(includePadding=False):
            print("WARNING: Attempting to optimize memory with a tag width constraint smaller than the minimum possible width! Optimization skipped.")
            return
        self.maxWidth = maxWidth
        self.supersets = minimizeRulesGreedy(self.supersets, {colID:1 for colID in self.columnIDs}, maxWidth)


    def indexOfSuperset(self, subset):
        for i, superset in enumerate(self.supersets):
            if superset.issuperset(subset):
                return i
        return -1


    def buildString(self, ssIndex, mask=None, subset=None, padChar='0', maskNegChar='', partChar='', width=None):
        """ padChar can be '0' or '*'
            maskNegChar is the characters for columnIDs absent from a mask. Can be '0' or '*'
            partChar can be '' or '|', and goes between the identifier and padding, and the padding and mask
        """
        assert not (mask == None and subset == None)
        if maskNegChar == '':
            maskNegChar = padChar
        if width == None:
            width = self.width()


        identifier = self.identifiers[ssIndex]

        superset = self.supersets[ssIndex]

        if mask == None:
            mask = ''.join(['1' if colID in subset else maskNegChar for colID in superset])

        padding = padChar * (width - (len(identifier) + len(mask)))

        return identifier + partChar + padding + partChar + mask



    def tag(self, row, decorated=False):
        index = self.indexOfSuperset(row)

        partChar = '|' if decorated else ''
        return self.buildString(index, subset=row, padChar='0', partChar=partChar)


    def matchStrings(self, columnID, decorated=False):
        tagWidth = self.width()
        partChar = '|' if decorated else ''
        outStrings = []
        for i, superset in enumerate(self.supersets):
            if columnID in superset:
                outStrings.append(self.buildString(i, subset=[columnID], width=tagWidth, padChar='*', partChar=partChar))

        return outStrings


    def allMatchStrings(self, decorated=False):
        tagWidth = self.width()
        partChar = '|' if decorated else ''

        outStrings = {colID : [] for colID in self.columnIDs}
        for ssIndex, superset in enumerate(self.supersets):
            maskBits = ['*'] * len(superset)
            for i, columnID in enumerate(superset):
                maskBits[i] = '1'
                mask = ''.join(maskBits)
                outStrings[columnID].append(self.buildString(ssIndex, mask=mask, width=tagWidth, padChar='*', partChar=partChar))
                maskBits[i] = '*'

        return outStrings




def main():
    matrix = [[1,2,3], [3,4,5], [6,7]]
    code = OriginalCodeStatic(matrix=matrix)
    code.timeMake(maxWidth=2)
    code.verifyCompression()
    print("Memory of original code:", code.memoryRequired())



if __name__ == "__main__":
    main()
