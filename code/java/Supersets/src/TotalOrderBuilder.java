import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;

import java.io.FileReader;
import java.util.Iterator;


import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;


public class TotalOrderBuilder {
    private ST<String, Integer> item2node;
    private ST<Integer, String> node2item;
    private ST<String, Queue<Integer>> item2bits;

    private Digraph orderGraph;
    private boolean orderingBuilt;

    private ST<String, HashSet<String>> itemAdj;

    private int bitsWorstCase;


    public TotalOrderBuilder(String[] items)
    {
        item2node = new ST<>();
        node2item = new ST<>();
        itemAdj = new ST<>();

        // map each item to an integer ID for the digraph
        for (int i = 0; i < items.length; i++) {
            item2node.put(items[i], i);
            node2item.put(i, items[i]);
            itemAdj.put(items[i], new HashSet<>());
        }

        orderGraph = new Digraph(items.length);
        orderingBuilt = false;
    }


    /**
     * Add the item ordering imposed by the given path to the underlying ordering.
     * @param path The item path to add to the underlying ordering.
     */
    public void addList(String[] path) {
        // our previously built total ordering is no longer correct
        orderingBuilt = false;

        /* Every item node v must have an edge to every other item node
         * w which appears after it in the path
         */
        for (int i = 0; i < path.length; i++) {
            int v = item2node.get(path[i]);
            for (int j = i + 1; j < path.length; j++) {
                int w = item2node.get(path[j]);

                if (!itemAdj.get(path[i]).contains(path[j])) {
                    itemAdj.get(path[i]).add(path[j]);
                    orderGraph.addEdge(v, w);
                }
            }
        }
    }

    /**
     * Topologically orders a SCC such that the backward edges are minimized
     * @param sccNodes a queue of IDs of the nodes that belong to the SCC
     * @return
     */
    private Queue<Integer> getOrderForOneSCC(Queue<Integer> sccNodes) {
        Digraph fwdEdges = new Digraph(orderGraph.V());
        Digraph backEdges = new Digraph(orderGraph.V());

        // which vertices are in the SCC?
        boolean[] isInSCC = new boolean[orderGraph.V()];
        for (int node : sccNodes)
            isInSCC[node] = true;

        // assume for now that every edge is a forward edge
        for (int v : sccNodes) {
            for (int w : orderGraph.adj(v)) {
                if (isInSCC[w])
                    fwdEdges.addEdge(v, w);
            }
        }
        Digraph reversedFwdEdges = fwdEdges.reverse();



        // and now we will remove edges until there is no cycle
        for (int v : sccNodes) {
            // if indegree(V) is smaller, remove all edges directed at V
            if (fwdEdges.indegree(v) < fwdEdges.outdegree(v)) {
                Stack<Integer> edgesToRemove = new Stack<>();
                for (int w : reversedFwdEdges.adj(v))
                    edgesToRemove.push(w);

                for (int w : edgesToRemove) {
                    fwdEdges.removeEdge(w, v);
                    reversedFwdEdges.removeEdge(v, w);
                    backEdges.addEdge(w, v);
                }
            }
            // else, remove all edges from V
            else {
                Stack<Integer> edgesToRemove = new Stack<>();
                for (int w : fwdEdges.adj(v))
                    edgesToRemove.push(w);

                for (int w : edgesToRemove) {
                    fwdEdges.removeEdge(v, w);
                    reversedFwdEdges.removeEdge(w, v);
                    backEdges.addEdge(v, w);
                }
            }

            // stop removing edges if there is no longer a cycle
            DirectedCycle dc = new DirectedCycle(fwdEdges);
            if (!dc.hasCycle()) break;
        }

        Topological top = new Topological(fwdEdges);

        Queue<Integer> order = new Queue<>();
        for (int node : top.order()) {
            if (isInSCC[node])
                order.enqueue(node);
        }
        return order;
    }


    /**
     * Given a path in the form of a list, return the indices of the bits that will be 1 in a mask.
     * @param list The path elements, in order. Must be a path which has been implied by inputs.
     * @return The indices of the element bits.
     */
    public int[] getBits(String[] list) {
        if (!orderingBuilt)
            buildOrder();


        int[] bits = new int[list.length];
        int prevIndex = -1;
        for (int i = 0; i < list.length; i++) {
            String item = list[i];

            // pick the earliest bit position which is greater than the previous item's position
            Queue<Integer> validItemBits = item2bits.get(item);
            boolean found = false;
            for (int bitIndex : validItemBits) {
                if (bitIndex > prevIndex) {
                    prevIndex = bitIndex;
                    found = true;
                    break;
                }
            }
            if (!found)
                throw new RuntimeException("Cannot build list of bits! Failed on item " + item);
            bits[i] = prevIndex;
        }
        return bits;
    }


    /**
     * Returns the number of bits required if all paths through the ordering are explored.
     * @return bits required by a worst-case set of paths
     */
    public int bitsWorstCase(){
        if (!orderingBuilt)
            buildOrder();

        return bitsWorstCase;
    }


    /**
     * Constructs a total ordering across all elements seen so far. Assigns multiple positions in the
     * ordering to a single element if that element has an ambiguous position i.e. is incomparable with
     * some other set of elements.
     */
    public void buildOrder() {

        TarjanSCC scc = new TarjanSCC(orderGraph);
        int M = scc.count();

        // edgeSTs[i] is the adjacency list for SCC i
        ST<Integer, Integer>[] edgeSTs = (ST<Integer, Integer>[]) new ST[M];
        for (int i = 0; i < edgeSTs.length; i++)
            edgeSTs[i] = new ST<>();

        // the vertices which are members of each SCC
        Queue<Integer>[] sccNodes = (Queue<Integer>[]) new Queue[M];
        for (int i = 0; i < sccNodes.length; i++)
            sccNodes[i] = new Queue<>();


        int[] sccSizes = new int[M];
        for (int v = 0; v < orderGraph.V(); v++) {
            sccNodes[scc.id(v)].enqueue(v); // add V to the scc's list of members
            for (int w : orderGraph.adj(v)) {
                // add the edge from the SCC of v to the SCC of w
                if (scc.id(v) != scc.id(w))
                    edgeSTs[scc.id(v)].put(scc.id(w), 1);
            }
        }

        // build the SCC DAG from the edges we've found between SCCs
        Digraph sccGraph = new Digraph(M);
        for (int v = 0; v < M; v++) {
            for (int w : edgeSTs[v].keys()) {
                sccGraph.addEdge(v, w);
            }
        }

        /*
        StdOut.println("Partial order graph: ");
        StdOut.println(orderGraph.toString());

        StdOut.println("SCC Graph: ");
        StdOut.println(sccGraph.toString());
        */

        Topological sccTopo = new Topological(sccGraph);
        if (!sccTopo.hasOrder())
            StdOut.println("No Order found! wat");


        item2bits = new ST<>();

/*
        for (int i : sccTopo.order()) {
            StdOut.print("Nodes in SCC " + i + ": ");
            for (int j : sccNodes[i])
                StdOut.print(node2item.get(j) + ", ");
            StdOut.println();
        }
        */




        int bitIndex = 0;
        for (int i : sccTopo.order()) {
            Queue<Integer> nodeOrder = getOrderForOneSCC(sccNodes[i]);
            for (int j : nodeOrder) {
                for (int k : nodeOrder) {
                    String item = node2item.get(k);
                    if (!item2bits.contains(item))
                        item2bits.put(item, new Queue<>());
                    item2bits.get(item).enqueue(bitIndex++);
                }
            }
        }
        bitsWorstCase = bitIndex;

        orderingBuilt = true;
    }


    /**
     * Takes the name of a json file containing a list of paths, and returns them as an array of arrays.
     * @param filename Name of the json file to read.
     * @return An array of arrays, where each array is an ordered path of elements.
     */
    public static String[][] readListsFromJSON(String filename) {
        //System.out.print("Reading paths from JSON... ");

        JSONParser parser = new JSONParser();
        ArrayList<String[]> paths = new ArrayList<>();

        try {
            JSONObject jsonObject = (JSONObject) parser.parse(new FileReader(filename));

            // Read in the paths from the JSON
            JSONArray jsonPaths = (JSONArray) jsonObject.get("paths");

            for (Object jsonPathObj : jsonPaths) {
                JSONArray jsonPath = (JSONArray) jsonPathObj;
                ArrayList<String> path = new ArrayList<>();

                for (int i = 0; i < jsonPath.size(); i++)
                {
                    String node = (String) jsonPath.get(i);
                    path.add(node);
                }

                paths.add(path.toArray(new String[path.size()]));

            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        //System.out.println("Done.");
        return paths.toArray(new String[paths.size()][]);
    }


    /**
     * Takes a list of paths, finds the ordering implied by those paths, and returns the ratio
     * of the number of bits needed to represent the implied ordering compared to the number of
     * input elements.
     * Also writes the resulting bit sequences to bit_sequences.txt.
     * @param paths A list of ordered paths through elements.
     * @return The inflation factor for the implied ordering.
     */
    public static double inflationFactorForChunk(String[][] paths) {
        HashSet<String> itemSet = new HashSet<>();
        for (String[] path : paths) {
            for (String item : path) {
                itemSet.add(item);
            }
        }

        String[] itemArray = new String[itemSet.size()];
        int i = 0;
        for (String item : itemSet)
            itemArray[i++] = item;



        TotalOrderBuilder tob = new TotalOrderBuilder(itemArray);

        // build the ordering graph
        for (String[] path : paths) {
            tob.addList(path);
        }


        tob.buildOrder();

        HashSet<Integer> bits = new HashSet<>();

        for (String[] path : paths) {
            for (int bit : tob.getBits(path)) {
                bits.add(bit);
            }
        }

        try {
            PrintWriter writer = new PrintWriter("data/bit_sequences.txt", "UTF-8");
            for (String[] path : paths) {
                writer.println(arrayToString(tob.getBits(path)));
            }
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
        return (double) bits.size() / (double) itemSet.size();

    }


    /* Helper for converting array of strings to a string. */
    public static String arrayToString(String[] array) {
        boolean first = true;
        StringBuilder sb = new StringBuilder("[");
        for (String item : array) {
            if (!first)
                sb.append(", ");
            else
                first = false;
            sb.append(item);
        }
        sb.append("]");

        return sb.toString();
    }

    /* Helper for converting array of integers to a string. */
    public static String arrayToString(int[] array) {
        boolean first = true;
        StringBuilder sb = new StringBuilder("[");
        for (int item : array) {
            if (!first)
                sb.append(", ");
            else
                first = false;
            sb.append(item);
        }
        sb.append("]");

        return sb.toString();
    }



    public static void main(String[] args) {

        String[][] paths = readListsFromJSON("data/paths600.json");

        int chunkSize = 60;

        String[][] pathsChunk = new String[chunkSize][];
        for (int i = 0; i < paths.length/chunkSize; i++) {
            for (int j = 0; j < chunkSize; j++) {
                pathsChunk[j] = paths[i*chunkSize + j];
            }

            double infFactor = inflationFactorForChunk(pathsChunk);
            //System.out.println("Inflation factor: " + infFactor);
            break;
        }

    /*

        HashSet<Integer> bits = new HashSet<>();

        for (String[] list : lists) {
            for (int bit : tob.getBits(list)) {
                bits.add(bit);
            }
        }
        System.out.println("Original number of elements: " + items.size());
        System.out.println("Number of bits required for total ordering: " + bits.size());
        System.out.println("Number of bits needed worst case: " + tob.bitsWorstCase());
        */
    }
}
