from BaseCodes import BaseCodeStatic, ColumnID, Row, FixedRow, Matrix, FixedMatrix, BinaryString, TernaryStrings


class MultiCode(BaseCodeStatic):

    subCodes      : List[BaseCodeStatic] = None
    subMatrices   : List[FixedMatrix]    = None
    colPartitions : List[Set[ColumnID]]  = None


    def make(self, *args, **kwargs) -> None:
        for subCode in self.subCodes:
            self.subCode.make(*args, **kwargs)


    def width(self):
        return sum([subCode.width() for subCode in self.subCodes])

    
    def tag(self, row, decorated=False) -> BinaryString:
        splitChar = '|' if decorated else ''
        outSub = ''
        subTags = [subCode.tag(row.intersection(subCode.columnIDs), decorated=decorated) for subCode in self.subCodes]
        return splitChar.join(subTags)


    def matchStrings(self, columnID, decorated=False) -> TernaryStrings:
        splitChar = '|' if decorated else ''
        codeIdx = 0
        while columnID not in self.subCodes[codeIdx].columnIDs:
            codeIdx += 1

        # one piece per subCode, initially all wildcards
        pieces = ['*'*subCode.width() for subCode in self.subCodes]
        outStrs = []
        for subMatchString in self.subCodes[codeIdx].matchStrings(columnID, decorated=decorated):
            pieces[codeIdx] = subMatchString
            outStrs.append(splitChar.join(pieces))

        return outStrs

