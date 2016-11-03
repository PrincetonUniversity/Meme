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
    jumpWeights = [0.8**i for i in range(numElements)] # array of probabilities for jump positions
    startWeights = [0.8**i for i in range(numElements)]

    paths = []

    # First, build totally ordered paths
    for _ in range(numPaths):
        path = []
        curNode = weighted_choice(startWeights)
        finalNode = min(numElements, curNode + random.randint(numElements/2, numElements))
        while curNode < finalNode:
            path.append(str(curNode))
            curNode += 1 + weighted_choice(jumpWeights)

        #if len(path) > maxLen:
        #    r = 1 + random.uniform(int(0.1*maxLen), maxLen-1)
        #    path = path[:r]

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
    directory = "../../data/"
    filename = "mboxpaths1.json"

    dump = {}
    countgap = 10
    repetitions = 10
    for mboxcount in range(countgap, 51, countgap):
        mboxkey = "Boxes:%03d"%mboxcount
        dump[mboxkey] = {}
        for pathcount in range(countgap, 401, countgap):
        #for pathcount in [1]:
            pathkey = "Paths:%03d"%pathcount
            dump[mboxkey][pathkey] = []
            for i in range(repetitions):
                dump[mboxkey][pathkey].append(generatePaths(mboxcount, pathcount, 1, 0.05))


        #filename = base_filename + "_num%3d"%count + "_err%2d"%int(err*100)

    #paths = generatePaths(20, 600, 1, 0.05)
    #for path in paths:
    #    print path

    with open(directory + filename,'w') as fp:
        json.dump(dump,fp)