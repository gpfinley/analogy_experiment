import java.io.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Use the analogy corpus provided with word2vec and score based on mean correlation with term 4
 *
 * Created by gpfinley on 4/26/16.
 */
public class EmnlpAnalogyExperiment {

    public final static String EXPERIMENTS_PROPERTIES = "experiments.properties";

    private static String header;

    public static void main(String[] args) throws IOException {

        // Allow use of custom properties file as a single command-line argument
        Reader reader;
        if(args.length > 0) {
            reader = new FileReader(args[0]);
        } else {
            reader = new FileReader(EXPERIMENTS_PROPERTIES);
        }

        Properties props = new Properties();
        props.load(reader);

        String analogiesFile = props.getProperty("emnlpAnalogyAnalogiesPath");
        boolean caseSensitive = Boolean.parseBoolean(props.getProperty("emnlpAnalogyCaseSensitive"));
        String embeddingsFile = props.getProperty("emnlpAnalogyVectorsPath");
        String wordNetPath = props.getProperty("wordNetPath");
        boolean useCosine = Boolean.parseBoolean(props.getProperty("emnlpAnalogyUseCosine"));
        boolean useW3W4 = Boolean.parseBoolean(props.getProperty("emnlpAnalogyUseW3W4"));
        boolean useW2W4 = Boolean.parseBoolean(props.getProperty("emnlpAnalogyUseW2W4"));
        boolean useW1W2 = Boolean.parseBoolean(props.getProperty("emnlpAnalogyUseW1W2"));
        boolean useW1W4 = Boolean.parseBoolean(props.getProperty("emnlpAnalogyUseW1W4"));
        boolean useAllRanks = Boolean.parseBoolean(props.getProperty("emnlpAnalogyUseRanks"));
        String vocabFilePath = props.getProperty("emnlpAnalogyVocabFile");
        Integer cutoff = null;
        try {
            cutoff = Integer.parseInt(props.getProperty("emnlpAnalogyVectorCutoff"));
        } catch(NumberFormatException e) {}


        AnalogyExperiment.Builder builder = new AnalogyExperiment.Builder()
                .analogiesFile(analogiesFile)
                .embeddingsFile(embeddingsFile)
                .wordNetPath(wordNetPath)
                .caseSensitive(caseSensitive);
        if(cutoff != null)
            builder = builder.filterOn(cutoff);

        if(vocabFilePath != null)
            builder = builder.useGlove(vocabFilePath);

        AnalogyExperiment exp = builder.createExperiment();

        header = "analogy";
        List<Map> parametersTested = new ArrayList<>();

        if(useCosine) {
            header += ",cos";
            parametersTested.add(exp.additiveSimilarity());
        }
        if(useAllRanks) {
            header += ",baserank,addrank,mulrank";
            exp.scoreRanks(null);
            parametersTested.add(exp.getBaselineRanks());
            parametersTested.add(exp.getAddRanks());
            parametersTested.add(exp.getMulRanks());
        }
        if(useW3W4) {
            Map<Analogy, Double> analogyBaselineScores = exp.baselineSimilarity();

            header += ",w3w4";
            parametersTested.add(analogyBaselineScores);
        }
        if(useW2W4) {
            Map<Analogy, Double> analogyDomainSimilarity = new HashMap<>();
            for(Analogy analogy : exp.getAnalogies()) {
                analogyDomainSimilarity.put(analogy, exp.getEmbeddings().get(analogy.w2).cosSim(exp.getEmbeddings().get(analogy.w4)));
            }
            header += ",w2w4";
            parametersTested.add(analogyDomainSimilarity);
        }
        if(useW1W2) {
            Map<Analogy, Double> analogyFirstPairSimilarity = new HashMap<>();
            for(Analogy analogy : exp.getAnalogies()) {
                analogyFirstPairSimilarity.put(analogy, exp.getEmbeddings().get(analogy.w1).cosSim(exp.getEmbeddings().get(analogy.w2)));
            }
            header += ",w1w2";
            parametersTested.add(analogyFirstPairSimilarity);
        }
        if(useW1W4) {
            Map<Analogy, Double> analogyW1W4 = new HashMap<>();
            for(Analogy analogy : exp.getAnalogies()) {
                analogyW1W4.put(analogy, exp.getEmbeddings().get(analogy.w1).cosSim(exp.getEmbeddings().get(analogy.w4)));
            }
            header += ",w1w4";
            parametersTested.add(analogyW1W4);
        }

        assert(parametersTested.size() == header.split(",").length);

        Map<Analogy, String> analogyCategories = new HashMap<>();
        for(Map.Entry<String, List<Analogy>> e : exp.getAnalogiesByCategory().entrySet()) {
            // build category lookup for each analogy
            for (Analogy analogy : e.getValue()) {
                analogyCategories.put(analogy, e.getKey());
            }

//            System.out.println("Category: " + e.getKey());
//            System.out.println(header);
//            printCorrelation(e.getValue(), parametersTested);
        }

//        System.out.println("OVERALL RESULTS");
//        printCorrelation(null, parametersTested);

        // Write out performance stats and predictors to CSV (can use in R later)

        DateFormat dateFormat = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
        Date date = new Date();
        String datetime = dateFormat.format(date);
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("analogy_experiment_stats_" + datetime + ".csv")));
        // todo: change writer to write to args[0]-derived filename
        writer.write(header + ",category\n");
        for(Analogy analogy : (Set<Analogy>) parametersTested.get(0).keySet()) {
            writer.write(analogy.toString());
            for(Map map : parametersTested) {
                writer.write("," + map.get(analogy));
            }
            writer.write("," + analogyCategories.get(analogy));
            writer.write("\n");
        }
        writer.flush();
        writer.close();
    }

}
