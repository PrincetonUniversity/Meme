from RSets import RCode
import json

def printAllSupersets(supersets, rcode):
    for (i, superset) in enumerate(supersets):
        print "Set %d" %i, str(rcode.bitString(superset))


print "----Running test case 1----"
sets1 = [[1,2,3,4], [3,4,5], [3,5,6], [4,5,6]]
rcode = RCode(sets1, 8, isOrdered=False)
rcode.buildCode()
for (i, superset) in enumerate(sets1):
    print "Set %d" %i, str(rcode.bitString(superset))
rcode.optimizeWidth()
rcode.buildCode()
print ">Optimized width"
printAllSupersets(sets1, rcode)

rcode.addSet([3,5])
rcode.addSet([3,4,5,6,7,8,9,10])

smallSet = [6,7]
changes = rcode.addSet(smallSet)
printAllSupersets(sets1, rcode)
print "small set: ", rcode.bitString(smallSet)
print "small set changes: ", json.dumps(changes)

mediumSet = [8,9,10,11]
changes = rcode.addSet(mediumSet)
printAllSupersets(sets1, rcode)
print "medium set: ", rcode.bitString(mediumSet)
print "medium set changes: ", json.dumps(changes)


