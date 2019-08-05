from abc import ABC, abstractmethod
from typing import Set, FrozenSet, List, Dict, Collection, Tuple
from common import recoverRow


class BaseCode(ABC):
    originalRows : List[FrozenSet[int]] = None
    columnIDs    : Collection[int] = None
    made         : bool = False

    @abstractmethod
    def __init__(self, matrix : List[Collection[int]] = None, **kwargs) -> None:
        self.originalRows = list(set([frozenset(row) for row in matrix]))
        self.columnIDs = list(frozenset.union(*self.originalRows))
        self.made = False

    @abstractmethod
    def make(self, **kwargs) -> None:
        """ Build the code.
        """
        pass

    @abstractmethod
    def width(self, includePadding : bool = True) -> int:
        """ Number of bits necessary for this code.
        """
        pass

    @abstractmethod
    def tag(self, row : Collection[int], decorated : bool = False) -> str:
        """ Returns a compressed tag for the given matrix row.
        """
        pass

    @abstractmethod
    def matchStrings(self, columnID : int, decorated : bool = False) -> List[str]:
        """ Returns a list of ternary strings which, when compared to a row's tag,
            will compare True if the columnID is present in the tag.
        """
        pass


    def allTags(self, decorated : bool = False) -> Dict[FrozenSet, set]:
        """ Returns a dictionary mapping matrix rows to compressed tags.
        """
        return {row : self.tag(row=row, decorated=decorated) for row in self.originalRows}


    def allMatchStrings(self, decorated : bool = False) -> Dict[int, List[str]]:
        """ Returns a dictionary mapping columnIDs to lists of ternary strings which,
            when compared to a row's tag, will compare True if the columnID is present in the tag.
        """
        return {colID : self.matchStrings(colID, decorated=decorated) for colID in self.columnIDs}


    def verifyCompression(self) -> bool:
        """ Verify that every matrix row can be recovered from the compressed tags and ternary match strings.
        """
        assert self.made

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
        print("Compression verified successfully")
        return True




class BaseCodeStatic(BaseCode):
    pass


class BaseCodeStaticOrdered(BaseCodeStatic):
    pass


class BaseCodeDynamic(BaseCode):

    @abstractmethod
    def addRow(self, row : Collection[int]) -> Tuple[str, Dict[int,List[str]], Dict[int,List[str]]]:
        """ Add a row to the matrix. Returns the tag for the new row, a dictionary of table entries
            to be added, and a dictionary of table entries to be deleted.
        """
        pass



class NaiveCode(BaseCodeStatic):
    matrix    : List[FrozenSet[int]] = None
    columnIDs : List[int] = None

    def __init__(self, matrix):
        self.matrix = [frozenset(row) for row in matrix]

        super().__init__(matrix)

    def make(self):
        self.made = True

    def width(self, includePadding=False):
        return len(self.columnIDs)

    def tag(self, row, decorated=False):
        return ''.join(['1' if colID in row else '0' for colID in self.columnIDs])


    def matchStrings(self, columnID, decorated=False):
        bits = ['*'] * len(self.columnIDs)
        bits[self.columnIDs.index(columnID)] = '1'
        return [''.join(bits)]


def main():
    matrix = [[1,2,3], [3,4,5], [6,7]]
    code = NaiveCode(matrix=matrix)
    code.make()
    code.verifyCompression()
    pass


if __name__ == "__main__":
    main()





