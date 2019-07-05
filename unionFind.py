from typing import Dict, Hashable

class UnionFind:
    nodeOwners : Dict = None

    def __init__(self, count : int = 0):
        # initially, each node is it's own owner
        self.nodeOwners = {i:i for i in range(count)}

    def find(self, node:Hashable):
        prevNode = None
        while node != prevNode:
            prevNode = node
            node = self.nodeOwners.setdefault(node, node)
        return node

    def union(self, node1:Hashable, node2:Hashable):
        owner1 = self.find(node1)
        owner2 = self.find(node2)

        # make node1's root own node2
        self.nodeOwners[owner2] = owner1

    def components(self):
        ownerGroups = {}
        for node in self.nodeOwners:
            owner = self.find(node)
            if owner not in ownerGroups:
                ownerGroups[owner] = [node]
            else:
                ownerGroups[owner].append(node)
        return [group for group in ownerGroups.values() if len(group) > 0]

def testUnionFind():
    uf = UnionFind(10)
    for i in [1,2,3]:
        uf.union(1, i)
    for i in [3,4,5,6]:
        uf.union(7, i)

    for i in [6,8,9]:
        uf.union(i, 7)

    groups = uf.components()
    print(groups)

if __name__ == "__main__":
    testUnionFind()
