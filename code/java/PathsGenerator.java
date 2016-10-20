import java.io.PrintWriter;

/**
 * Created by robertmacdavid on 6/5/16.
 */
public class PathsGenerator {
    /**
     *
     * @param n The number of elements total
     * @param m The number of paths through the elements
     * @param k the maximum distance at which elements are flipped
     * @param p The probability of flipping any element with another
     * @return
     */
    public static String[][] generate(int n, int m, int k, double p) {
        return new String[5][5];

    }


    public static void main(String[] args) {

        String[][] paths = {{"A", "B", "C", "D"}, {"A", "C", "B", "D",}, {"D", "A"}};

        try {
            PrintWriter writer = new PrintWriter("data/generated_1.json", "UTF-8");
            writer.print("{\n\t\"paths\" :\n\t[");

            boolean firstpath = true;

            for (String[] path : paths) {
                if (!firstpath)
                    writer.print(", ");
                else
                    firstpath = false;
                writer.print("\n");

                writer.print("\t\t[ ");
                boolean first = true;
                for (String node : path) {
                    if (!first) {
                        writer.print(", ");
                    }
                    else
                        first = false;

                    writer.print("\"" + node + "\"");
                }
                writer.print(" ]");



            }
            writer.print("\n\t]\n}");

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
