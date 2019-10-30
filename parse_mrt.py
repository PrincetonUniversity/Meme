from collections import defaultdict, Counter
import pickle


def mrtCsvToMatrix(filename, outfile):
    pathIDtoAS = {}


    ASset = set()
    with open(filename, 'r') as fp:
        AS = ''
        pathID = ''
        oldPathID = ''
        prefix = ''
        session = ''
        flag = 0
        firstNLRI = 0
        for line in fp:
            line = line.rstrip()
            if "            Path Segment Value: " in line:
                AS = line.replace("            Path Segment Value: ", '').split(' ')[0]
                ASset.add(AS)
            if "        COMMUNITY: " in line:
                line = line.replace("        COMMUNITY: ", '')
                sessionMap = {pair.split(":")[0] : pair.split(":")[1] for pair in line.split(' ')}
                session = sessionMap["47065"]
            if "        Path Identifier: " in line:
                pathID = line.replace("        Path Identifier: ", '')
                if firstNLRI == 0:
                    firstNLRI = 1
                elif flag == 1:
                    assert oldPathID == pathID
                oldPathID = pathID
                if flag == 1:
                    if pathID not in pathIDtoAS:
                        pathIDtoAS[pathID] = {}
                    if session not in pathIDtoAS[pathID]:
                        pathIDtoAS[pathID][session] = defaultdict(set)

            if line == "    NLRI":
                flag = 1
            if line == "    Withdrawn Routes":
                flag = 2
            if line == "---------------------------------------------------------------":
                firstNLRI = 0
            if "        Prefix: " in line:
                prefix = line.replace("        Prefix: ", '')
                if flag == 1:
                    pathIDtoAS[pathID][session][AS].add(prefix)
                    #if session in rprefixtoAS:
                        #rprefixtoAS[session][prefix].add(AS)
                if flag == 2:
                    removedMap = defaultdict(set)
                    for iterSession in pathIDtoAS[pathID].keys():
                        for iterAS in pathIDtoAS[pathID][iterSession].keys():
                            if prefix in pathIDtoAS[pathID][iterSession][iterAS]:
                                removedMap[iterSession].add(iterAS)
                                pathIDtoAS[pathID][iterSession][iterAS].remove(prefix)
                    #if len(removedMap) > 1:
                    #    print("Multi session ", pathID, prefix, removedMap)
                    #if len(removedMap) > 0 and max([len(v) for v in removedMap.values()]) > 1:
                    #    print("Multi AS ", pathID, prefix, removedMap)                                        
                    

    prefixtoAS = defaultdict(set)
    for iterPath in pathIDtoAS.keys():
        for iterSession in pathIDtoAS[iterPath].keys():
            for iterAS in pathIDtoAS[iterPath][iterSession].keys():
                for iterPrefix in pathIDtoAS[iterPath][iterSession][iterAS]:
                    prefixtoAS[iterPrefix].add(iterAS)

    prefixes = set(prefixtoAS.keys())
    nextHops = set.union(*list(prefixtoAS.values()))

    #print("\n %d nexthops, %d prefixes" % (len(nextHops), len(prefixes)))
    #print("mean nexthops per prefix: ", (0.0 + sum([len(hopset) for hopset in prefixtoAS.values()]))/len(prefixes))

    counts = Counter([len(hopset) for hopset in prefixtoAS.values()])
    #print(counts)
    #print("\n count of in ASes: ", len(ASset))
    #print("\n count of intersecting ASes: ", len(ASset.intersection(nextHops)))

    matrixfull = set([frozenset(v) for v in prefixtoAS.values()])

    print("Writing matrix to file", outfile)
    with open(outfile, "wb") as f:
        pickle.dump(matrixfull, f)
    print("Done writing matrix.")

    return matrixfull


if __name__ == "__main__":
    infile = "PeeringClient/fulldumpers1.csv"
    outfile = "matrix.pickle"
    mrtToMatrix(infile, outfile)
