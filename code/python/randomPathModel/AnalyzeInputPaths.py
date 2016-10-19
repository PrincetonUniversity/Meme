from prefix_groups_analysis import minimize_bits_greedy, bitsRequired


if __name__ == '__main__':
    paths = []
    with open("../data/bit_sequences.txt", "r") as fp:
        for line in fp:
            line.strip()
            line = line[1:-2]
            path = line.split(", ")
            paths.append(path)

    elements = set([])
    for path in paths:
        elements.update(path)
    print "Distinct elements: ", elements
    print len(elements), "distinct elements found."
    print "Minimizing bits required..."
    pathsMinimized = minimize_bits_greedy(paths)
    print "Bits required: ", bitsRequired(pathsMinimized)