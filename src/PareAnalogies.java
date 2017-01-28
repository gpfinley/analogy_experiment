import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by greg on 1/28/17.
 */
public class PareAnalogies {


    /**
     * Inner class used to pare down a large bunch of analogies on a separate run, according to some criteria
     * Must have:
     *      input file
     *      output file
     *      WordNet path
     *      embeddings file
     * Currently makes some assumptions
     */
        public static void main(String[] args) throws Exception {
            new PareAnalogies(args);
        }

        private Embeddings emb;
        private WordNet wordNet;

        @Option(name="-emin")
        private Double minEntropy = null;

        @Option(name="-emax")
        private Double maxEntropy = null;

        @Option(name="-anyemin")
        private Double minAnyEntropy = null;

        @Option(name="-anyemax")
        private Double maxAnyEntropy = null;

        @Option(name="-emb")
        private String embeddingsFile = null;

        @Option(name="-in")
        private String inputFile;

        @Option(name="-rel")
        private String forceRelation;

        // min similarity between terms 3 and 4
        @Option(name="-min34")
        private double min34 = 0;

        // min similarity between terms 2 and 4 (e.g., domain similarity between both sides of the analogy)
        @Option(name="-min24")
        private double min24 = 0;

        // min similarity between terms 1 and 2
        @Option(name="-min12")
        private double min12 = 0;

        @Option(name="-max34")
        private double max34 = 1;
        @Option(name="-max24")
        private double max24 = 1;
        @Option(name="-max12")
        private double max12 = 1;

        @Option(name="-pros1min")
        private double pros1min = 0;
        @Option(name="-pros2min")
        private double pros2min = 0;
        @Option(name="-pros1max")
        private double pros1max = 1;
        @Option(name="-pros2max")
        private double pros2max = 1;

        @Option(name="-scores")
        private String scoresFile;
        @Option(name="-wn")
        private String wordNetPath = null;

        @Option(name="-out")
        private String outputFile;

        @Argument
        private List<String> arguments = new ArrayList<>();

        public PareAnalogies(String[] args) throws Exception {
            CmdLineParser parser = new CmdLineParser(this);
            parser.parseArgument(args);

            if(forceRelation != null) {
                // todo: make this work for all relations!!!
                if (forceRelation.equals("hyper")) forceRelation = "@";
                else if (forceRelation.equals("inst_hyper")) forceRelation = "@i";
                else if (forceRelation.equals("memb_holo")) forceRelation = "#m";
                else if (forceRelation.equals("part_holo")) forceRelation = "#p";
                else if (forceRelation.equals("anto")) forceRelation = "!";
            }

            if(embeddingsFile != null) {
                emb = Word2vecReader.readBinFile(embeddingsFile);
            }
            if(wordNetPath != null) {
                wordNet = new WordNet(wordNetPath);
            }

            BufferedReader reader = new BufferedReader(new FileReader(inputFile));
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
            String line;
            int numRead = 0;
            while((line = reader.readLine()) != null) {
                if(line.startsWith(":")) {
                    writer.write(line + "\n");
                    continue;
                }
                String[] words = line.split("\\s+");

                if(fitsCriteria(words)) {
                    writer.write(line + "\n");
                }

                numRead++;
                if(numRead % 100000 == 0) {
                    System.out.println(numRead + " analogies read");
                }
            }
            writer.flush();
            writer.close();
        }

        private boolean fitsCriteria(String[] words) {
            if(emb != null) {
                if (!emb.contains(words[0]) || !emb.contains(words[1]) || !emb.contains(words[2]) || !emb.contains(words[3]))
                    return false;
                double sim24 = emb.get(words[1]).cosSim(emb.get(words[3]));
                if (sim24 < min24 || sim24 > max24)
                    return false;
                double sim34 = emb.get(words[2]).cosSim(emb.get(words[3]));
                if (sim34 < min34 || sim34 > max34)
                    return false;
                double sim12 = emb.get(words[0]).cosSim(emb.get(words[1]));
                if (sim12 < min12 || sim12 > max12)
                    return false;
            }

            if(wordNetPath !=  null) {
                double pros1 = wordNet.probRelationOverSenses(words[0], words[1], forceRelation);
                double pros2 = wordNet.probRelationOverSenses(words[2], words[3], forceRelation);
                if(pros1 < pros1min || pros1 > pros1max || pros2 < pros2min || pros2 > pros2max)
                    return false;
                double e1 = wordNet.getEntropyOverLemmas(words[0]);
                double e2 = wordNet.getEntropyOverLemmas(words[1]);
                double e3 = wordNet.getEntropyOverLemmas(words[2]);
                double e4 = wordNet.getEntropyOverLemmas(words[3]);
                double entropy = e1 + e2 + e3 + e4;
                if(minAnyEntropy != null && (e1 < minAnyEntropy || e2 < minAnyEntropy || e3 < minAnyEntropy || e4 < minAnyEntropy))
                    return false;
                if(maxAnyEntropy != null && (e1 > maxAnyEntropy || e2 > maxAnyEntropy || e3 > maxAnyEntropy || e4 > maxAnyEntropy))
                    return false;
                if(minEntropy != null && entropy < minEntropy)
                    return false;
                if(maxEntropy != null && entropy > maxEntropy)
                    return false;
            }

            return true;
        }



}
