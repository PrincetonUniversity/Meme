from abc import ABC, abstractmethod
from typing import Set, FrozenSet, List, Dict, Collection, Tuple, NewType
from common import recoverRow
import time
import logging



BinaryString   = NewType('BinaryString',  str)
TernaryString  = NewType('TernaryString', str)
TernaryStrings = NewType('TernaryStrings', List[TernaryString])

ColumnID       = NewType('ColumnID',      int)
Row      = NewType('Row',  Collection[ColumnID])
FixedRow = NewType('FRow', FrozenSet[ColumnID])
Matrix      = NewType('Matrix',  Collection[Row])
FixedMatrix = NewType('FMatrix', Collection[FixedRow])

# log levels in increasing order of verbosity
LOGGING_LEVELS = {0 : logging.CRITICAL,
                  1 : logging.ERROR,
                  2 : logging.WARNING,
                  3 : logging.INFO,
                  4 : logging.DEBUG}

CODE_LOG_LEVEL = LOGGING_LEVELS[3]


class BaseCode(ABC):
    originalRows : FixedMatrix = None       # The original rows of the input matrix, deduplicated
    columnIDs    : Set[ColumnID] = None     # All ColumnIDS present in the original matrix
    made         : bool = False             # Has the code been built?
    logger       : logging.Logger = None    # Module logger

    def __init__(self, matrix : Matrix = None, verbosity=3, **kwargs) -> None:
        CODE_LOG_LEVEL = LOGGING_LEVELS[verbosity]
        logging.basicConfig(level=CODE_LOG_LEVEL)
        self.logger = logging.getLogger(type(self).__name__)
       
        self.logger.debug("Base code class init called")
        self.originalRows = list(set([frozenset(row) for row in matrix]))
        self.columnIDs = list(frozenset.union(*self.originalRows))
        self.made = False



    @abstractmethod
    def make(self, *args, **kwargs) -> None:
        """ Build the code.
        """
        pass


    @abstractmethod
    def unmake(self, *args, **kwargs) -> None:
        """ Reset the code to unbuilt status.
        """
        pass


    @abstractmethod
    def tag(self, row : Row, decorated : bool = False) -> BinaryString:
        """ Returns a compressed tag for the given matrix row.
        """
        pass

    @abstractmethod
    def matchStrings(self, columnID : ColumnID, decorated : bool = False) -> TernaryStrings:
        """ Returns a list of ternary strings which, when compared to a row's tag,
            will compare True if the columnID is present in the tag.
        """
        pass


    def width(self, includePadding : bool = True) -> int:
        """ Number of bits necessary for this code.
        """
        self.logger.debug("Placeholder width method of base code class was called. Override recommended.")
        longestTag = max(len(tag) for tag in self.allTags().values())
        longestQuery = max(max(len(qStr) for qStr in qStrs) for qStrs in self.allMatchStrings().values())
        return max(longestTag, longestQuery)


    def allTags(self, decorated : bool = False) -> Dict[FixedRow, BinaryString]:
        """ Returns a dictionary mapping matrix rows to compressed tags.
        """
        self.logger.debug("allTags of base code class was called.")
        assert self.made
        return {row : self.tag(row=row, decorated=decorated) for row in self.originalRows}


    def allMatchStrings(self, decorated : bool = False) -> Dict[BinaryString, TernaryStrings]:
        """ Returns a dictionary mapping columnIDs to lists of ternary strings which,
            when compared to a row's tag, will compare True if the columnID is present in the tag.
        """
        self.logger.debug("allMatchStrings of base code class was called.")
        assert self.made
        return {colID : self.matchStrings(colID, decorated=decorated) for colID in self.columnIDs}


    def timeMake(self, *args, **kwargs) -> float:
        """ Run self.make(*args, **kwargs), and print how long it took.
        """
        className = type(self).__name__
        numRows = len(self.originalRows)
        numCols = len(self.columnIDs)
        print("Beginning code make for %4d rows and %4d columns using code class %s." % (numRows, numCols, className))
        startTime = time.time()
        self.make(*args, **kwargs)
        endTime = time.time()

        elapsed = endTime - startTime
        print("Make done. Took %4.2f seconds." % elapsed)
        assert self.made # make sure the make method properly sets self.made
        return elapsed


    def memoryRequired(self, columnRuleSizes : Dict[ColumnID, List[int]] = None) -> Tuple[int, int]:
        """ Returns memory required in the switch as a tuple (sramBits, tcamBits).
            columnRuleSizes is a dictionary which maps columnIDs to policy rule overheads,
            i.e. if column 1 appears in 4 MAT entries which have bit widths 8,16,16,32 before including the column TCAM check,
            then columnRuleSizes[1] == [8,16,16,32].
        """
        assert self.made
        if columnRuleSizes == None:
            columnRuleSizes = {colID : [0] for colID in self.columnIDs}
        columnOccurrences = {colID : len(ruleSizes) for colID, ruleSizes in columnRuleSizes.items()}

        queryStrings = self.allMatchStrings(decorated=False)
        tagWidth = self.width()

        sramBits = 0
        tcamBits = 0
        for colID in self.columnIDs:
            sramBits += sum(columnRuleSizes[colID])
            tcamBits += tagWidth * columnOccurrences[colID]

        return (sramBits, tcamBits)

    def tableEntriesRequired(self):
        assert self.made
        return sum(len(strings) for strings in self.allMatchStrings().values())

    def printInfo(self):
        assert self.made
        sramBits, tcamBits = self.memoryRequired()
        codeWidth = self.width()
        entries = self.tableEntriesRequired()

        print("Code width is %3d, number of entries is %5d" % (codeWidth, entries))
        print("Memory required is %4d SRAM bits, %4d TCAM bits" % (sramBits, tcamBits))



    def verifyCompression(self) -> bool:
        """ Verify that every matrix row can be recovered from the compressed tags and ternary match strings.
        """
        assert self.made # it is a programmer error to try to verify a code before it has been built
        self.logger.debug("Verifying compression...")

        queryDict = self.allMatchStrings()
        tagDict = self.allTags()

        for row in self.originalRows:
            if row not in tagDict:
                raise Exception("Row %s does not have an associated tag!" % str(row))

        for colID in self.columnIDs:
            colStrings = queryDict.get(colID, None)
            if colStrings == None or len(colStrings) == 0:
                raise Exception("A column does not have an associated set of ternary match strings!")

        for row, tag in tagDict.items():
            recovered = recoverRow(tag, queryDict)
            if set(row) != set(recovered):
                raise Exception("Row %s with tag %s incorrectly decompressed to %s!" % (str(row), str(tag), str(recovered)))
        self.logger.info("Compression verified successfully")
        return True




class BaseCodeStatic(BaseCode):
    pass


class BaseCodeStaticOrdered(BaseCodeStatic):
    pass


class BaseCodeDynamic(BaseCode):

    @abstractmethod
    def addRow(self, row : Row) -> Tuple[BinaryString, Dict[ColumnID,TernaryStrings], Dict[ColumnID,TernaryStrings]]:
        """ Add a row to the matrix. Returns the tag for the new row, a dictionary of table entries
            to be added, and a dictionary of table entries to be deleted.
        """
        pass



class NaiveCode(BaseCodeStatic):

    def make(self):
        self.made = True

    def unmake(self):
        self.made = False

    #def width(self, includePadding=False):
    #    return len(self.columnIDs)

    def tag(self, row, decorated=False):
        return ''.join(['1' if colID in row else '0' for colID in self.columnIDs])


    def matchStrings(self, columnID, decorated=False):
        bits = ['*'] * len(self.columnIDs)
        bits[self.columnIDs.index(columnID)] = '1'
        return [''.join(bits)]


def main():
    matrix = [[1,2,3], [3,4,5], [6,7]]
    code = NaiveCode(matrix=matrix)
    code.timeMake()
    code.verifyCompression()
    print("Memory of naive code:", code.memoryRequired())
    pass


if __name__ == "__main__":
    main()





