package kow.kowal;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.model.*;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.Random;

import static java.lang.Math.toIntExact;

public class Dl4j {

    private Logger log = LoggerFactory.getLogger(getClass());

    public static void main(String[] args) throws Exception {
        new Dl4j().testDarknet();
    }

    public void testSimpleCNNNet() throws Exception {
        // needs reconfiguration
        SimpleCNN build = org.deeplearning4j.zoo.model.SimpleCNN.builder().seed(123)
                .inputShape(new int[] { 3, 128, 128 }).numClasses(2).build();
        MultiLayerConfiguration net = build.conf();

        runMultiLayerConfiguration(net, 100);
    }

    public void testNASNet() throws Exception {
        //invalid
        NASNet build = org.deeplearning4j.zoo.model.NASNet.builder().seed(123)
                .inputShape(new int[] { 3, 128, 128 }).numClasses(2).build();
        ComputationGraph init = build.init();
        runComputationGraph(init, 500);
    }

    public void testAlexNet() throws Exception {
        // invalid ..
//        AlexNet build = org.deeplearning4j.zoo.model.AlexNet.builder().seed(123).inputShape(new int[] { 3, 128, 128 }).numClasses(2).build();
        MultiLayerConfiguration net = new AnimalsClassification().alexnetModel2();
        runMultiLayerConfiguration(net, 100);
    }


    public void testXceptionNet() throws Exception {
        // 121 layer oO .. produces high load even on single batches
        Xception build = org.deeplearning4j.zoo.model.Xception.builder().seed(123)
                .inputShape(new int[] { 3, 128, 128 }).numClasses(2).build();
        ComputationGraph init = build.init();
        runComputationGraph(init, 500);
    }

    public void testUNet() throws Exception {
        // invalid
        UNet build = org.deeplearning4j.zoo.model.UNet.builder().seed(123)
                .inputShape(new int[] { 3, 128, 128 }).numClasses(2).build();
        ComputationGraph init = build.init();
        runComputationGraph(init, 500);
    }

    public void testSqueezeNet() throws Exception {
        SqueezeNet build = org.deeplearning4j.zoo.model.SqueezeNet.builder().seed(123)
                .inputShape(new int[] { 3, 128, 128 }).numClasses(2).build();
        ComputationGraph init = build.init();
        runComputationGraph(init, 500);
    }

    public void testLeNet() throws Exception {
        // converges pretty fast
        LeNet build = org.deeplearning4j.zoo.model.LeNet.builder().seed(123).inputShape(new int[] { 3, 128, 128 })
                .numClasses(2).build();
        MultiLayerConfiguration net = build.conf();
        runMultiLayerConfiguration(net, 100);
    }

    public void testResnet() throws Exception {
        // invalid ..
        ResNet50 build = org.deeplearning4j.zoo.model.ResNet50.builder().seed(123).inputShape(new int[] { 3, 128, 128 })
                .numClasses(2).build();
        ComputationGraph init = build.init();
        runComputationGraph(init, 100);
    }

    public void testDarknet() throws Exception {
        Darknet19 build = org.deeplearning4j.zoo.model.Darknet19.builder().seed(123)
                .inputShape(new int[] { 3, 128, 128 }).numClasses(2).build();
        ComputationGraph init = build.init();
        runComputationGraph(init, 100);
    }

    public void testVGG16() throws Exception {
        // runs into NAN
        VGG16 build = org.deeplearning4j.zoo.model.VGG16.builder().seed(123).inputShape(new int[] { 3, 128, 128 })
                .numClasses(2).build();
        ComputationGraph init = build.init();
        runComputationGraph(init, 100);
    }

    public void testVGG19() throws Exception {
        // runs into NAN
        VGG19 build = org.deeplearning4j.zoo.model.VGG19.builder().seed(123).inputShape(new int[] { 3, 128, 128 })
                .numClasses(2).build();
        ComputationGraph init = build.init();
        runComputationGraph(init, 100);

    }

    private void runComputationGraph(ComputationGraph graph, int batches) throws Exception {
        RunProvisioner p = new RunProvisioner("/home/wmi/lego-recognition-neural-java/src/main/resources/lego", 128, 128, 3, 5, 2)
                .withTerminateAfterBatches(batches);
        graph = p.setup(graph);
        graph.fit(p.getDataIterator(), 20);
        p.evaluateCg(graph);
    }

    private void runMultiLayerConfiguration(MultiLayerConfiguration model, int batches) throws Exception {

        RunProvisioner p = new RunProvisioner("/home/wmi/lego-recognition-neural-java/src/main/resources/lego", 128, 128, 3, 5, 2)
                .withTerminateAfterBatches(batches);
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(model);
        multiLayerNetwork = p.setup(multiLayerNetwork);
        multiLayerNetwork.fit(p.getDataIterator(), 20);
        p.evaluateMln(multiLayerNetwork);
    }

    private class RunProvisioner {

        private final Logger log = LoggerFactory.getLogger(RunProvisioner.class);
        private final String basepath;
        private final int height;
        private final int width;
        private final int channels;
        private final int batchSize;
        private final int classes;
        private Integer terminateAfter;

        private DataSetIterator dataIterator;
        private String trainingLabels;
        private InputSplit testFilesplit;
        private ImageRecordReader recordReader;
        private DataSetIterator testIter;

        private StatsStorage statsStorage;

        private void startUiServer() {
            // Initialize the user interface backend
            UIServer uiServer = UIServer.getInstance();
            // Configure where the network information (gradients, score vs. time etc) is to
            // be stored. Here: store in memory.
//            statsStorage = new FileStatsStorage(File.createTempFile("stats", "")); // Alternative: new
            statsStorage  = new InMemoryStatsStorage();
            // FileStatsStorage(File),
            // for
            // saving
            // and loading later
            // Attach the StatsStorage instance to the UI: this allows the contents of the
            // StatsStorage to be visualized
            uiServer.attach(statsStorage);

        }

        public RunProvisioner withTerminateAfterBatches(int batches) {
            this.terminateAfter = batches;
            return this;
        }

        public StatsStorage getStatsStorage() {
            return statsStorage;
        }

        DataSetIterator getDataIterator() {
            return dataIterator;
        }

        public String getTrainingLabels() {
            return trainingLabels;
        }

        public RunProvisioner(String basepath, int height, int width, int channels, int batchSize, int classes) {
            super();
            this.basepath = basepath;
            this.height = height;
            this.width = width;
            this.channels = channels;
            this.batchSize = batchSize;
            this.classes = classes;
        }

        private void initReaders() throws IOException {
//            File trainData = new File(basepath + "/train");
//            File testData = new File(basepath + "/test");


            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
            File mainPath = new File("./src/main/resources/lego");
            FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, new Random());
            int numExamples = toIntExact(fileSplit.length());
            int numLabels = Objects.requireNonNull(fileSplit.getRootDir().listFiles(File::isDirectory)).length; //This only works if your root is clean: only label subdirs.
            BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(), labelMaker, numExamples, numLabels, numExamples);

            double splitTrainTest = 0.8;
            InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
            InputSplit train = inputSplit[0];
            testFilesplit = inputSplit[1];

            // Define the FileSplit(PATH, ALLOWED FORMATS,random)
//            FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random());
//            testFilesplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random());

            // Extract the parent path as the image label
//            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

            recordReader = new ImageRecordReader(height, width, channels, labelMaker);

            // Initialize the record reader
            // add a listener, to extract the name
            recordReader.initialize(train);

            // The LogRecordListener will log the path of each image read
            // used here for information purposes,
            // If the whole dataset was ingested this would place 60,000
            // lines in our logs
            // It will show up in the output with this format
            // o.d.a.r.l.i.LogRecordListener - Reading /tmp/mnist_png/training/4/36384.png
            recordReader.setListeners(new LogRecordListener());

            // DataSet Iterator

            if (terminateAfter == null) {
                dataIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1,
                        recordReader.getLabels().size());
            } else {
                dataIterator = new EarlyTerminationDataSetIterator(
                        new RecordReaderDataSetIterator(recordReader, batchSize, 1, recordReader.getLabels().size()),
                        terminateAfter);
            }

            // Scale pixel values to 0-1
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.fit(dataIterator);
            dataIterator.setPreProcessor(scaler);

            trainingLabels = dataIterator.getLabels().toString();
            log.info("BUILD MODEL");
        }

        private CheckpointListener checkpointListener() {
            return new CheckpointListener.Builder(new File(basepath + "/models")).keepLast(3).saveEveryNIterations(1000)
                    .build();
        }

        private ComputationGraph setup(ComputationGraph graph) throws Exception {
            startUiServer();
            initReaders();
            prepareEval();
            graph.setListeners(new ScoreIterationListener(5), new StatsListener(getStatsStorage()), new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
            return graph;

        }

        public MultiLayerNetwork setup(MultiLayerNetwork model) throws Exception {
            startUiServer();
            initReaders();
            prepareEval();
            model.setListeners(new ScoreIterationListener(5), new StatsListener(getStatsStorage()),
                    new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
            return model;
        }

        private Evaluation prepareEval() throws IOException {
            log.info("EVALUATE MODEL");
            recordReader.reset();

            // The model trained on the training dataset split
            // now that it has trained we evaluate against the
            // test data of images the network has not seen

            recordReader.initialize(testFilesplit);

            RecordReaderDataSetIterator testDataReader = new RecordReaderDataSetIterator(recordReader, batchSize, 1,
                    classes);

            if (terminateAfter == null) {
                testIter = testDataReader;
            } else {
                testIter = new EarlyTerminationDataSetIterator(testDataReader, terminateAfter);
            }

            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

            scaler.fit(testIter);
            testIter.setPreProcessor(scaler);

            /*
             * log the order of the labels for later use In previous versions the label
             * order was consistent, but random In current verions label order is
             * lexicographic preserving the RecordReader Labels order is no longer needed
             * left in for demonstration purposes
             */
            String testlabels = recordReader.getLabels().toString();

            log.info("Test labels : {}", testlabels);

            // Create Eval object with 10 possible classes
            Evaluation eval = new Evaluation(classes);

            return eval;
        }

        private void evaluateCg(ComputationGraph model) throws IOException {
            Evaluation eval = prepareEval();

//			// Evaluate the network
            while (testIter.hasNext()) {
                DataSet next = testIter.next();
                INDArray testFeatures = next.getFeatures();

                for (INDArray arr : model.output(testFeatures)) {
                    eval.eval(next.getLabels(), arr);
                }

            }

            log.info("Eval Stats : \r\n{}", eval.stats());

            System.out.println(eval.confusionToString());

        }

        private void evaluateMln(MultiLayerNetwork model) throws IOException {
            Evaluation eval = prepareEval();

//			// Evaluate the network
//            while (testIter.hasNext()) {
//                DataSet next = testIter.next();
//                INDArray testFeatures = next.getFeatures();
//                INDArray testLabels = next.getLabels();
//
//                INDArray output = model.output(testFeatures);
//                eval.eval(testLabels, output);
//            }

            model.doEvaluation(testIter, eval);

            log.info("Eval Stats : \r\n{}", eval.stats());

            dataIterator.reset();
            DataSet testDataSet = dataIterator.next();
            List<String> allClassLabels = recordReader.getLabels();
            int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
            int[] predictedClasses = model.predict(testDataSet.getFeatures());
            String expectedResult = allClassLabels.get(labelIndex);
            String modelPrediction = allClassLabels.get(predictedClasses[0]);
            System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");

        }
    }

}

