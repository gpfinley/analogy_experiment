import com.sun.istack.internal.Nullable;

import java.io.*;
import java.util.*;

/**
 * Use the analogy corpus provided with word2vec and score based on mean correlation with term 4
 *
 * Created by gpfinley on 4/26/16.
 */
public class AnalogyExperiment {

    public static class Builder {

        private String embeddingsFile;
        private String analogiesFile;
        private String wordNetPath = null;
        private boolean caseSensitive = false;
        private int filterOn = 1000000000;
        private String vocabFile = null;
        private Embeddings emb = null;
        private List<Analogy> analogies = null;
        private Map<String, List<Analogy>> analogiesByCategory = null;

        public Builder embeddingsFile(String embeddingsFile) {
            this.embeddingsFile = embeddingsFile;
            return this;
        }
        public Builder useEmbeddings(Embeddings emb) {
            this.emb = emb;
            return this;
        }
        public Builder useAnalogies(List<Analogy> analogies) {
            this.analogies = analogies;
            return this;
        }
        public Builder useCategorizedAnalogies(Map<String, List<Analogy>> analogies) {
            this.analogiesByCategory = analogies;
            return this;
        }
        public Builder analogiesFile(String analogiesFile) {
            this.analogiesFile = analogiesFile;
            return this;
        }
        public Builder wordNetPath(String wordNetPath) {
            this.wordNetPath = wordNetPath;
            return this;
        }
        public Builder caseSensitive(boolean caseSensitive) {
            this.caseSensitive = caseSensitive;
            return this;
        }
        public Builder filterOn(int filterOn) {
            this.filterOn = filterOn;
            return this;
        }

        public Builder useGlove(String vocabFile) {
            this.vocabFile = vocabFile;
            return this;
        }

        public AnalogyExperiment createExperiment() throws IOException {

            if(emb == null) {
                if(vocabFile == null) {
                    emb = Word2vecReader.readBinFile(embeddingsFile);
                }
                else {
                    emb = GloVeReader.readBinFile(embeddingsFile, vocabFile);
                }
                emb.normalizeAll();
                if(filterOn < emb.size()) {
                    emb.filterOn(new HashSet<>(emb.getLexicon().subList(0, filterOn)));
                }
            }

            if(analogies == null && analogiesByCategory == null) {
                analogies = new ArrayList<>();
                analogiesByCategory = new HashMap<>();
                List categoryActiveList = null;
                InputStream inputStream = new FileInputStream(analogiesFile);
                BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith(":")) {
                        categoryActiveList = new ArrayList<>();
                        analogiesByCategory.put(line, categoryActiveList);
                    } else {
                        if(categoryActiveList == null) {
                            categoryActiveList = new ArrayList<>();
                            analogiesByCategory.put("first category", categoryActiveList);
                        }
                        if (!this.caseSensitive) line = line.toLowerCase();
                        String[] fields = line.split(" ");
                        if(emb.contains(fields[0]) && emb.contains(fields[1]) && emb.contains(fields[2]) && emb.contains(fields[3])) {
                            Analogy thisAnalogy = new Analogy(fields);
                            analogies.add(thisAnalogy);
                            if (!emb.contains(thisAnalogy.w1) || !emb.contains(thisAnalogy.w2) || !emb.contains(thisAnalogy.w3) || !emb.contains(thisAnalogy.w4)) {
                                System.out.println("bad analogy: " + thisAnalogy);
                            }
                            categoryActiveList.add(thisAnalogy);
                        } else {
                            System.out.println("Throwing out analogy " + line);
                        }
                    }
                }
            } else {
                if(analogiesByCategory == null) {
                    analogiesByCategory = new HashMap<>();
                    analogiesByCategory.put("default", analogies);
                }
                if(analogies == null) {
                    analogies = new ArrayList<>();
                    for(List<Analogy> oneList : analogiesByCategory.values()) {
                        analogies.addAll(oneList);
                    }
                }
            }

            AnalogyExperiment experiment = new AnalogyExperiment();
            experiment.emb = emb;
            experiment.analogies = analogies;
            experiment.analogiesByCategory = analogiesByCategory;
            if(wordNetPath != null) {
                experiment.wordNet = new WordNet(wordNetPath);
            }

            return experiment;
        }
    }

    private List<Analogy> analogies;
    private Map<String, List<Analogy>> analogiesByCategory;
    private Embeddings emb;
    private WordNet wordNet;

    private Map<Analogy, Integer> baselineRanks;
    private Map<Analogy, Integer> addRanks;
    private Map<Analogy, Integer> mulRanks;

    public Embeddings getEmbeddings() {
        return emb;
    }

    public WordNet getWordNet() {
        return wordNet;
    }

    public List<Analogy> getAnalogies() {
        return analogies;
    }

    public Map<String, List<Analogy>> getAnalogiesByCategory() {
        return analogiesByCategory;
    }

    private AnalogyExperiment() {}

    /**
     * Determine all words or phrases used among all analogies
     * @return
     */
    public Set<Phrase> wordsUsed() {
        Set<Phrase> used = new HashSet<>();
        for(Analogy analogy : analogies) {
            used.add(analogy.w1);
            used.add(analogy.w2);
            used.add(analogy.w3);
            used.add(analogy.w4);
        }
        return used;
    }

    public Map<Analogy, Integer> getBaselineRanks() {
        return baselineRanks;
    }
    public Map<Analogy, Integer> getAddRanks() {
        return addRanks;
    }
    public Map<Analogy, Integer> getMulRanks() {
        return mulRanks;
    }

    /**
     */
    public void scoreRanks(@Nullable String category) {
        baselineRanks = new HashMap<>();
        addRanks = new HashMap<>();
        mulRanks = new HashMap<>();
        List<Analogy> testList;
        if(category == null) {
            testList = analogies;
        } else {
            testList = analogiesByCategory.get(category);
        }
        int counter = 0;
        for (Analogy analogy : testList) {
            WordEmbedding w1 = emb.get(analogy.w1);
            WordEmbedding w2 = emb.get(analogy.w2);
            WordEmbedding w3 = emb.get(analogy.w3);
            WordEmbedding w4 = emb.get(analogy.w4);
            WordEmbedding w1p = new WordEmbedding(w1);
            WordEmbedding w2p = new WordEmbedding(w2);
            WordEmbedding w3p = new WordEmbedding(w3);
            WordEmbedding w4p = new WordEmbedding(w4);
            w1p.add(1);
            w2p.add(1);
            w3p.add(1);
            w4p.add(1);
            int addRank = 1;
            int mulRank = 1;
            int baselineRank = 1;
            if (emb.contains(analogy.w1) && emb.contains(analogy.w2) && emb.contains(analogy.w3) && emb.contains(analogy.w4)) {
                double addScore;
                double baselineScore;
                double mulScore;

                WordEmbedding calculated = analogyHypothesisEmbedding(analogy);
                addScore = calculated.dot(emb.get(analogy.w4));
                baselineScore = w3.dot(w4);

                mulScore = scoreLevyGoldberg(w1p, w2p, w3p, w4p);
                Iterator<WordEmbedding> iter = emb.embeddingIterator();
                while (iter.hasNext()) {
                    WordEmbedding hyp = iter.next();
                    if(hyp == w1 || hyp == w2 || hyp == w3) continue;
                    double addCompScore = calculated.dot(hyp);
                    if (addCompScore > addScore) addRank += 1;
                    double baselineCompScore = w3.dot(hyp);
                    if (baselineCompScore > baselineScore) baselineRank += 1;
                    hyp = new WordEmbedding(hyp);
                    hyp.add(1);
                    double mulCompScore = scoreLevyGoldberg(w1p, w2p, w3p, hyp);
                    if (mulCompScore > mulScore) mulRank += 1;
                }
            } else {
                baselineRank = emb.size();
                addRank = emb.size();
                mulRank = emb.size();
            }
            baselineRanks.put(analogy, baselineRank);
            addRanks.put(analogy, addRank);
            mulRanks.put(analogy, mulRank);
            counter++;
            if (counter % 10 == 0) {
                System.out.println(counter + " analogy ranks calculated");
            }
        }
    }

    /**
     * Return a mapping between all category names and all pairs (represented as strings w/ ':' in middle)
     *      represented in any analogy of that category
     */
    public Map<String, Set<String>> getPairsByCategory() {

        Map<String, Set<String>> pairsByCategory = new HashMap<>();

        for(String category : analogiesByCategory.keySet()) {
            pairsByCategory.put(category, new HashSet<String>());

            List<Analogy> analogies = analogiesByCategory.get(category);

            for (Analogy analogy : analogies) {
                String relation1 = analogy.w1 + ":" + analogy.w2;
                String relation2 = analogy.w3 + ":" + analogy.w4;
                pairsByCategory.get(category).add(relation1);
                pairsByCategory.get(category).add(relation2);
            }
        }

        return pairsByCategory;
    }

    public Map<Analogy, Double> baselineSimilarity() {
        Map<Analogy, Double> analogyBaselineScores = new HashMap<>();
        for(Analogy analogy : analogies) {
            if(emb.contains(analogy.w3) && emb.contains(analogy.w4)) {
                analogyBaselineScores.put(analogy, emb.get(analogy.w3).cosSim(emb.get(analogy.w4)));
            } else {
                analogyBaselineScores.put(analogy, 0.);
            }
        }
        return analogyBaselineScores;
    }

    public Map<Analogy, Double> additiveSimilarity() {
        Map<Analogy, Double> sumSimilarity = new HashMap<>();
        for(Analogy analogy : analogies) {
            if(emb.contains(analogy.w1) && emb.contains(analogy.w2) && emb.contains(analogy.w3) && emb.contains(analogy.w4)) {
            sumSimilarity.put(analogy, analogyHypothesisEmbedding(analogy).cosSim(emb.get(analogy.w4)));
            } else {
                sumSimilarity.put(analogy, 0.);
            }
        }
        return sumSimilarity;
    }

    public Map<String, Double> similaritiesToCategoryMean() {
        Map<String, WordEmbedding> meanVectors = new HashMap<>();
        Map<String, Double> sims = new HashMap<>();

        for(String category : analogiesByCategory.keySet()) {
            List<Analogy> analogies = analogiesByCategory.get(category);
            Map<String, Double> similarities = new HashMap<>();

            WordEmbedding meanVector = new WordEmbedding(emb.dimensionality());
            for (Analogy analogy : analogies) {
                String relation1 = analogy.w1 + ":" + analogy.w2;
                String relation2 = analogy.w3 + ":" + analogy.w4;
                if (emb.contains(analogy.w1) && emb.contains(analogy.w2)) {
                    if(!similarities.containsKey(relation1)) {
                        meanVector.add(emb.get(analogy.w1).difference(emb.get(analogy.w2)));
                        similarities.put(relation1, -1.);
                    }
                }
                if (emb.contains(analogy.w3) && emb.contains(analogy.w4)) {
                    if (!similarities.containsKey(relation2)) {
                        meanVector.add(emb.get(analogy.w3).difference(emb.get(analogy.w4)));
                        similarities.put(relation2, -1.);
                    }
                }
            }
            meanVector.scalarMultiply(1. / similarities.size());
            meanVectors.put(category, meanVector);

            for (Analogy analogy : analogies) {
                String relation1 = analogy.w1 + ":" + analogy.w2;
                String relation2 = analogy.w3 + ":" + analogy.w4;
                if(similarities.get(relation1) == -1.) {
                    double sim = emb.get(analogy.w1).difference(emb.get(analogy.w2)).cosSim(meanVector);
                    similarities.put(relation1, sim);
                }
                if(similarities.get(relation2) == -1.) {
                    double sim = emb.get(analogy.w3).difference(emb.get(analogy.w4)).cosSim(meanVector);
                    similarities.put(relation2, sim);
                }
            }

            System.out.println(similarities.size() + "\t" + category);
            sims.putAll(similarities);
        }
        return sims;
    }

    private WordEmbedding analogyHypothesisEmbedding(Analogy analogy) {
        WordEmbedding calculated = new WordEmbedding(emb.get(analogy.w3));
        calculated.add(emb.get(analogy.w2));
        calculated.subtract(emb.get(analogy.w1));
        return calculated;
    }

    /**
     * Compute Levy & Goldberg's 3CosMul score for an analogy
     * All vectors should be positive!
     * @param w1 word embedding for first word in analogy
     * @param w2 second
     * @param w3 third
     * @param w4 fourth (hypothesis or gold)
     * @return multiplicative score (does not normalize vectors)
     */
    private double scoreLevyGoldberg(WordEmbedding w1, WordEmbedding w2, WordEmbedding w3, WordEmbedding w4) {
        return w4.cosSim(w3) * w4.cosSim(w2) / (.001 + w4.cosSim(w1));
    }

}

