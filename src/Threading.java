import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.logging.Logger;

/**
 * Created by gpfinley on 8/25/16.
 */
public class Threading {

    private final static Logger LOGGER = Logger.getLogger(Threading.class.getName());

    private static int nThreads;
    static {
        String nThreadsStr = System.getProperties().getProperty("threads");
        if (nThreadsStr != null) {
            nThreads = Integer.parseInt(nThreadsStr);
        } else {
            nThreads = 20;
        }
    }

    /**
     * Get the number of threads that will be used for all operations.
     * @return the number of threads
     */
    public static int getnThreads() {
        return nThreads;
    }

    /**
     * Set the number of threads that will be used for all operations.
     * To set threads at application startup, use JVM argument -Dthreads=#
     * @return the number of threads to use
     */
    public static void setnThreads(int nThreads) {
        Threading.nThreads = nThreads;
    }


    /**
     * Perform threaded processing with any objects. Needs a class
     * @param max
     * @param threadClass
     * @param args
     */
    public static void chunkAndThread(int max, Class<? extends IntRangeThread> threadClass, Object... args) {
        int chunkSize = max/nThreads;
        List<Thread> threads = new ArrayList<>(nThreads);
        for(int i=0; i<nThreads; i++) {
            final int begin = i * chunkSize;
            final int end;
            if(i < nThreads - 1) {
                end = (i+1) * chunkSize;
            } else {
                end = max;
            }
            IntRangeThread thread;
            try {
                thread = threadClass.newInstance();
            } catch(Exception e) {
                e.printStackTrace();
                System.exit(1);
                thread = null;
            }
            thread.setBeginEnd(begin, end);
            thread.initializeParams(args);
            threads.add(thread);
            thread.start();
        }
        // wait for all the threads to finish before returning
        int nReady;
        do {
            nReady = 0;
            for(Thread thread : threads) {
                if(thread.getState().equals(Thread.State.TERMINATED)) nReady++;
            }
        } while(nReady < nThreads);
    }

    /**
     * Generic class for processing using this threading method.
     */
    public static abstract class IntRangeThread extends Thread {

        protected int begin;
        protected int end;

        private void setBeginEnd(int begin, int end) {
            this.begin = begin;
            this.end = end;
        }

        /**
         * Parse and cast the objects in this array to set up variables for processing.
         * Pass the same objects as the final arguments to chunkAndThread.
         * @param args a comprehensive list of objects for threaded processing
         */
        public abstract void initializeParams(Object[] args);

        /**
         * Override this method to include your threaded processing
         */
        @Override
        public abstract void run();
    }

    /**
     * Run multithreaded processing and fill an array with the results.
     * Pass a pre-created array and a function that calculates results based on the index of the array.
     * Array will be changed in place.
     * Function should not access indices of the answers array other than the integer passed to it (not threadsafe).
     * @param answers an array of any kind of object
     * @param func a function that takes integers and returns the same type as the array
     * @param <T> format of the results (likely Double, but other types possible)
     */
    public static <T> void fillArrayThreaded(T[] answers, Function<Integer, T> func) {
        chunkAndThread(answers.length, ArrayFillThread.class, answers, func);
    }

    private static class ArrayFillThread<T> extends IntRangeThread {
        protected T[] answers;
        protected Function<Integer, T> func;

        public ArrayFillThread() {}

        public void initializeParams(Object[] args) {
            answers = (T[]) args[0];
            func = (Function<Integer, T>) args[1];
        }

        public void run() {
            for (int i = begin; i < end; i++) {
                answers[i] = func.apply(i);
            }
            LOGGER.info(String.format("Finished from %d to %d", begin, end));
        }
    }
}
