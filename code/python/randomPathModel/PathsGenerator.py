import json
import random


def weighted_choice(weights):
   total = sum(weights)
   r = random.uniform(0, total)
   upto = 0
   for i, w in enumerate(weights):
      if upto + w >= r:
         return i
      upto += w
   assert False, "Shouldn't get here"



def generatePaths(numElements, numPaths, maxFlipDist, probFlip):
    jumpWeights = [0.7**i for i in range(numElements)] # array of probabilities for jump positions
    startWeights = [0.9**i for i in range(numElements)]

    paths = []

    # First, build totally ordered paths
    for _ in range(numPaths):
        path = []
        curNode = weighted_choice(startWeights)
        finalNode = min(numElements, curNode + random.randint(numElements/2, numElements))
        while curNode < finalNode:
            path.append(str(curNode))
            curNode += 1 + weighted_choice(jumpWeights)

        paths.append(path)


    # Next, iterate through and randomly flip some elements
    for path in paths:
        for i in range(len(path)):
            r = random.random()
            if r < probFlip:
                flipPos = i + random.randint(0, maxFlipDist)
                flipPos = min(flipPos, len(path) - 1)

                path[i], path[flipPos] = path[flipPos], path[i]

    return paths



if __name__ == '__main__':
    filename = "../data/paths600.json"


    paths = generatePaths(20, 600, 3, 0.1)
    for path in paths:
        print path

    things = {"paths":paths}

    with open(filename,'w') as fp:
        json.dump(things,fp)