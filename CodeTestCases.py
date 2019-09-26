#!/usr/bin/env python3.7

from BaseCodes import BaseCode, BaseCodeStatic, NaiveCode
from MultiCodes import MultiCode
from ClusterCodes import OriginalCodeStatic
import random

ALL_CODES = [NaiveCode, MultiCode, OriginalCodeStatic]


def baseTest(CodeClass, matrix, **kwargs):
    """ Verify a code class on a given matrix.
    """
    code = CodeClass(matrix=matrix, **kwargs)
    code.timeMake()
    code.verifyCompression()
    print("Memory required:", code.memoryRequired())



def test1(CodeClass):
    """ Simple test. Small matrix.
    """
    matrix = [[1,2,3], [3,4,5], [6,7]]

    baseTest(CodeClass, matrix)



def test2(CodeClass, rows=100, columns=100, density=0.1):
    """ Simple test. Uniformly random sparse matrix.
    """

    matrixSize = rows * columns 

    matrix = [set() for _ in range(rows)]
    for _ in range(int(matrixSize * density)):
        row = random.randint(0, rows-1)
        col = random.randint(0, columns-1)

        matrix[row].add(col)

    baseTest(CodeClass, matrix)


ALL_TESTS = [test1, test2]


if __name__ == "__main__":
    for CodeClass in ALL_CODES:
        for test in ALL_TESTS:
            test(CodeClass)







