from .RSets import RCode
import json

print("Executing test cCases")


def printAllSupersets(supersets, rcode):
    for (i, superset) in enumerate(supersets):
        print("Set %d" %i, str(rcode.tagString(superset)))


print("----Running test case 1----")
sets1 = [[1,2,3,4,5]]
rcode = RCode(sets1, 5)
rcode.buildCode()
printAllSupersets(sets1, rcode)


print("----Running test case 2----")
sets2 = [[1, 2, 3, 4], [3, 4, 5], [3, 5, 6], [4, 5, 6]]
rcode = RCode(sets2, 8)
rcode.buildCode()
for (i, superset) in enumerate(sets2):
    print("Set %d" %i, str(rcode.tagString(superset)))
rcode.optimizeWidth()
rcode.buildCode()
print(">Optimized width")
printAllSupersets(sets2, rcode)

rcode.addSet([3,5])
rcode.addSet([3,4,5,6,7,8,9,10])

smallSet = [6,7]
changes = rcode.addSet(smallSet)
printAllSupersets(sets2, rcode)
print("small set: ", rcode.tagString(smallSet))
print("small set changes: ", json.dumps(changes))

mediumSet = [8,9,10,11]
changes = rcode.addSet(mediumSet)
printAllSupersets(sets2, rcode)
print("medium set: ", rcode.tagString(mediumSet))
print("medium set changes: ", json.dumps(changes))

print("Encoding in full:", rcode.allSupersets())



print("----Running test case 3----")
sets3 = [[1,2,3], [3,4,5], [5,6,7], [7,8,9]]
rcode = RCode(sets3, 9)
rcode.buildCode()
print("Before optimization: ", rcode.allSupersets())

rcode.optimizeMemory(padding = 3)
rcode.buildCode()
print("After memory optimization: ", rcode.allSupersets())

rcode.optimizeMemory()
rcode.buildCode()
print("After padding optimization: ", rcode.allSupersets())


print("----Running test case 4----")
sets = [[3,4,5,6], [6,7,8,9], [12,13,14]]
rcode = RCode(sets, 13, isOrdered=True, elementOrdering={i:i for i in range(18)})
rcode.buildCode()
print("Before additions: ", rcode.allSupersets())
rcode.addSet([1,2,3,4,5])
print("After addition 1: ", rcode.allSupersets())
rcode.addSet([10,11])
print("After addition 2: ", rcode.allSupersets())
rcode.addSet([15,16,17])
print("After addition 3: ", rcode.allSupersets())



print("----Running test case 5----")
sets = [
    [5, 8], [7, 8],
    [3, 5], [0, 7],
    [1, 6], [1, 2],
    [3, 7], [0, 6],
    [0, 2, 4, 5], [3, 4, 5, 7],
    [1, 4], [1, 5],
    [5], [7],
    [1, 4, 7], [1, 5, 6],
    [0, 4], [3, 5],
    [0], [3]
]
rcode = RCode(sets, 13, isOrdered=True, elementOrdering = {i:i for i in range(9)})
rcode.buildCode()
printAllSupersets(sets, rcode)


