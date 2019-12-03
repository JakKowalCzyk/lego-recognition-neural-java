package kow.kowal; /*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultipleEpochsIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.toIntExact;

/**
 * Animal Classification
 * <p>
 * Example classification of photos from 4 different animals (bear, duck, deer, turtle).
 * <p>
 * References:
 * - U.S. Fish and Wildlife Service (animal sample dataset): http://digitalmedia.fws.gov/cdm/
 * - Tiny ImageNet Classification with CNN: http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
 * <p>
 * CHALLENGE: Current setup gets low score results. Can you improve the scores? Some approaches:
 * - Add additional images to the dataset
 * - Apply more transforms to dataset
 * - Increase epochs
 * - Try different model configurations
 * - Tune by adjusting learning rate, updaters, activation & loss functions, regularization, ...
 */

public class LegoClassification {
    private static final Logger log = LoggerFactory.getLogger(LegoClassification.class);
    protected static int height=64;
    protected static int width=64;

    protected static int channels = 3;
    protected static int batchSize=150;// tested 50, 100, 200
    protected static long seed = 123;
    protected static Random rng = new Random(seed);
    protected static int iterations = 1;
    protected static int nEpochs = 50; // tested 50, 100, 200
    protected static double splitTrainTest = 0.8;
    protected static boolean save = true;
    private int numLabels;
    private static String OS = null;

    protected static String modelType = "AlexNet"; // LeNet, AlexNet or Custom but you need to fill it out


    public static void main(String[] args) throws Exception {
        new LegoClassification().run(args);
    }

    public static String getOsName()
    {
        if(OS == null) { OS = System.getProperty("os.name"); }
        return OS;
    }
    public static boolean isWindows()
    {
        return getOsName().startsWith("Windows");
    }

    public void run(String[] args) throws Exception {

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        String pathname;
        if (isWindows()) {
            pathname = "C:\\projects\\lego-recognition-neural-java\\src\\main\\resources\\lego";
        }else {
            pathname = "./src/main/resources/lego";
        }
        File mainPath = new File(pathname);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplit.length());
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

        /**
         * Split data: 80% training and 20% testing
         */
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        /**
         *  Create extra synthetic training data by flipping, rotating
         #  images on our data set.
         */
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));

        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, flipTransform2});
        /**
         * Normalization
         **/
        log.info("Fitting to dataset");
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);

        MultiLayerNetwork network;
        switch (modelType) {
            case "LeNet":
                network = lenetModel();
                break;
            case "AlexNet":
                network = alexnetModel();
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }

        network.init();
        // Visualizing Network Training
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        network.setListeners(new StatsListener( statsStorage),new ScoreIterationListener(iterations));

        /**
         * Load data
         */
        log.info("Load data....");
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;


        log.info("Train model....");
        // Train without transformations
        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);
        trainIter = new MultipleEpochsIterator(nEpochs, dataIter);
        network.fit(trainIter);

        // Train with transformations
//        for (ImageTransform transform : transforms) {
//            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
//            recordReader.initialize(trainData, transform);
//            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
//            preProcessor.fit(dataIter);
//            dataIter.setPreProcessor(preProcessor);
//            trainIter = new MultipleEpochsIterator(nEpochs, dataIter);
//            network.fit(trainIter);
//        }

        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));

        if (save) {
            log.info("Save model....");
            ModelSerializer.writeModel(network,  "bird.bin", true);
        }
        log.info("**************** Bird Classification finished ********************");
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5, 5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(PoolingType.AVG, kernel, new int[]{2, 2}, new int[]{0,0}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    public MultiLayerNetwork lenetModel() {
        /**
         * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())
                .list()
                .layer(0, convInit("cnn1", channels, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 1))
                .layer(1, maxPool("maxpool1", new int[]{2, 2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 1  ))
                .layer(3, maxPool("maxpool2", new int[]{2, 2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public MultiLayerNetwork alexnetModel() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new AdaDelta())
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .l2(5 * 1e-4)
                .list()
                .layer(convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(maxPool("maxpool1", new int[]{3, 3}))
                .layer(conv5x5("cnn2", 256, new int[]{1, 1}, new int[]{2, 2}, nonZeroBias))
                .layer(new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(maxPool("maxpool2", new int[]{3, 3}))
                .layer(conv3x3("cnn3", 384, 0))
                .layer(conv3x3("cnn4", 384, nonZeroBias))
                .layer(conv3x3("cnn5", 256, nonZeroBias))
                .layer(maxPool("maxpool3", new int[]{3, 3}))
                .layer(fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

}