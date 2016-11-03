from randomPathsGenerator import generatePaths
import subprocess
import json
from RSets import RCode
import sys
from RSets import RCode


dataDir = "../data/"


def spew_files():
    baseFileName = "paths_"
    extension = "_u.json"

    experiment_results = {}

    filenames = []

    repetitions = 10
    # iterate over total middlebox counts
    for mboxcount in range(10, 41, 10):
        mboxkey = "Boxes%03d"%mboxcount
        # iterate over number of paths
        # for pathcount in range(10, 401, 10):
        for pathcount in range(100, 801, 100):
            pathkey = "Paths%03d"%pathcount
            # iterate over experiment counts
            print mboxkey, pathkey,
            for i in range(repetitions):
                print i,
                paths = generatePaths(mboxcount, pathcount, 1, 0.05)
                dump = {"paths":paths}
                filename = baseFileName + mboxkey + '_' + pathkey + "_Iter" + str(i) + extension
                with open(dataDir + filename,'w') as fp:
                    json.dump(dump,fp)
                filenames.append(filename)



                """
                java_output = subprocess.check_output(["java", "-cp", javadir,
                                        "TotalOrderBuilder", datadir + filename])

                supersets = json.loads(java_output)
                originalSets = [supersets[i*2] for i in range(len(supersets)/2)]
                orderedSets = [supersets[i*2 + 1] for i in range(len(supersets)/2)]
                maxElement = max(max(superset) for superset in orderedSets)
                ordering = {i:i for i in range(maxElement+1)}

                rcode = RCode(orderedSets, maxWidth=1000, elementOrdering=ordering)
                rcode.optimizeWidth()

                experiment_results.append(rcode.widthRequired())
                """
            print "."

    with open(dataDir + "path_filenames.txt", "w") as text_file:
        text_file.write('\n'.join(filenames))


def analyze_files(filenames):
    widthResults = {}
    memoryResults = {}
    for filename in filenames:

        print "Handling file", filename

        with open(filename, 'r') as fp:
            supersets = json.load(fp)

        originalSets = [supersets[i*2] for i in range(len(supersets)/2)]
        orderedSets = [supersets[i*2 + 1] for i in range(len(supersets)/2)]
        maxElement = max(max(superset) for superset in orderedSets)
        ordering = {i:i for i in range(maxElement+1)}

        rcode = RCode(orderedSets, maxWidth=60, elementOrdering=ordering)
        rcode.optimizeWidth()
        #rcode.optimizeMemory()




        namebits = filename.split('_')
        boxkey = namebits[1]  # number of middleboxes for this file
        pathkey = namebits[2]  # number of paths present in this file
        if boxkey not in widthResults:
            widthResults[boxkey] = {pathkey:[]}
            memoryResults[boxkey] = {pathkey:[]}
        if pathkey not in widthResults[boxkey]:
            widthResults[boxkey][pathkey] = []
            memoryResults[boxkey][pathkey] = []


        widthResults[boxkey][pathkey].append(rcode.widthRequired())
        memories = [mem for mem in rcode.memoryPerElement().values() if mem != 0]
        memoryResults[boxkey][pathkey].extend(memories)

    print "Writing experiment results"
    with open(dataDir + "ordered_widths_experiment.json", 'w') as fp:
        json.dump(widthResults, fp)
    with open(dataDir + "ordered_memories_experiment.json", 'w') as fp:
        json.dump(memoryResults, fp)
    print "Done."




if __name__ == '__main__':

    if len(sys.argv) == 1:
        spew_files()
    else:
        analyze_files(sys.argv[1:])



