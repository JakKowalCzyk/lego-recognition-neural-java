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
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultipleEpochsIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

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
    protected static int height = 64;
    protected static int width = 64;

    protected static int channels = 3;
    protected static int batchSize = 100;// tested 50, 100, 200
    protected static long seed = 1234;
    protected static Random rng = new Random(seed);
    protected static int iterations = 5;
    protected static int nEpochs = 1; // tested 50, 100, 200
    protected static double splitTrainTest = 0.8;
    protected static boolean save = true;
    private int numLabels;
    private static String OS = null;
    //     protected static long seed = 1234;
    private int[] inputShape = new int[]{3, 299, 299};
    private int numClasses = 2;
    private WeightInit weightInit = WeightInit.RELU;
    private IUpdater updater = new AdaDelta();
    private CacheMode cacheMode = CacheMode.NONE;
    private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    protected static String modelType = "x"; // LeNet, AlexNet or Custom but you need to fill it out


    public static void main(String[] args) throws Exception {
        new LegoClassification().run(args);
    }

    public static String getOsName() {
        if (OS == null) {
            OS = System.getProperty("os.name");
        }
        return OS;
    }

    public static boolean isWindows() {
        return getOsName().startsWith("Windows");
    }

    public void run(String[] args) throws Exception {

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        String pathname;
        if (isWindows()) {
            pathname = "C:\\projects\\lego-recognition-neural-java\\src\\main\\resources\\lego2";
        } else {
            pathname = "./src/main/resources/lego";
        }
        File mainPath = new File(pathname);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplit.length());
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, numExamples);

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

        NeuralNetwork network;
        switch (modelType) {
            case "LeNet":
                network = lenetModel();
                break;
            case "AlexNet":
                network = alexnetModel();
                break;
            case "x":
                network = init();
                break;
//            case "res":
//                network = new InceptionRestNet().init();
//                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }

        network.init();
        // Visualizing Network Training
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
//        network.setListeners(new StatsListener( statsStorage),new ScoreIterationListener(iterations), new PerformanceListener(1));
        if (network instanceof ComputationGraph) {
            ((ComputationGraph) network).setListeners(new StatsListener(statsStorage), new ScoreIterationListener(iterations), new PerformanceListener(1));
        } else {
            ((MultiLayerNetwork) network).setListeners(new StatsListener(statsStorage), new ScoreIterationListener(iterations), new PerformanceListener(1));
        }
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
        Evaluation eval = null;
        if (network instanceof ComputationGraph) {
            eval = ((ComputationGraph) network).evaluate(dataIter);
        } else {
            eval = ((MultiLayerNetwork) network).evaluate(dataIter);
        }
        log.info(eval.stats(true));

        if (save) {
            log.info("Save model....");
//            ModelSerializer.writeModel(network,  "bird.bin", true);
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
        return new SubsamplingLayer.Builder(PoolingType.AVG, kernel, new int[]{2, 2}, new int[]{0, 0}).name(name).build();
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
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 1))
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

    public ComputationGraph init() {
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder();

        graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]));

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }

    public ComputationGraphConfiguration.GraphBuilder graphBuilder() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .weightInit(weightInit)
                .l2(4e-5)
                .miniBatch(true)
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Truncate)
                .graphBuilder();

        graph
                // block1
                .addLayer("block1_conv1", new ConvolutionLayer.Builder(3, 3).stride(2, 2).nOut(32).hasBias(false)
                        .cudnnAlgoMode(cudnnAlgoMode).build(), "input")
                .addLayer("block1_conv1_bn", new BatchNormalization(), "block1_conv1")
                .addLayer("block1_conv1_act", new ActivationLayer(Activation.RELU), "block1_conv1_bn")
                .addLayer("block1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).hasBias(false)
                        .cudnnAlgoMode(cudnnAlgoMode).build(), "block1_conv1_act")
                .addLayer("block1_conv2_bn", new BatchNormalization(), "block1_conv2")
                .addLayer("block1_conv2_act", new ActivationLayer(Activation.RELU), "block1_conv2_bn")

                // residual1
                .addLayer("residual1_conv", new ConvolutionLayer.Builder(1, 1).stride(2, 2).nOut(128).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block1_conv2_act")
                .addLayer("residual1", new BatchNormalization(), "residual1_conv")

                // block2
                .addLayer("block2_sepconv1", new SeparableConvolution2D.Builder(3, 3).nOut(128).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block1_conv2_act")
                .addLayer("block2_sepconv1_bn", new BatchNormalization(), "block2_sepconv1")
                .addLayer("block2_sepconv1_act", new ActivationLayer(Activation.RELU), "block2_sepconv1_bn")
                .addLayer("block2_sepconv2", new SeparableConvolution2D.Builder(3, 3).nOut(128).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block2_sepconv1_act")
                .addLayer("block2_sepconv2_bn", new BatchNormalization(), "block2_sepconv2")
                .addLayer("block2_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3, 3).stride(2, 2)
                        .convolutionMode(ConvolutionMode.Same).build(), "block2_sepconv2_bn")
                .addVertex("add1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block2_pool", "residual1")

                // residual2
                .addLayer("residual2_conv", new ConvolutionLayer.Builder(1, 1).stride(2, 2).nOut(256).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "add1")
                .addLayer("residual2", new BatchNormalization(), "residual2_conv")

                // block3
                .addLayer("block3_sepconv1_act", new ActivationLayer(Activation.RELU), "add1")
                .addLayer("block3_sepconv1", new SeparableConvolution2D.Builder(3, 3).nOut(256).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block3_sepconv1_act")
                .addLayer("block3_sepconv1_bn", new BatchNormalization(), "block3_sepconv1")
                .addLayer("block3_sepconv2_act", new ActivationLayer(Activation.RELU), "block3_sepconv1_bn")
                .addLayer("block3_sepconv2", new SeparableConvolution2D.Builder(3, 3).nOut(256).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block3_sepconv2_act")
                .addLayer("block3_sepconv2_bn", new BatchNormalization(), "block3_sepconv2")
                .addLayer("block3_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3, 3).stride(2, 2)
                        .convolutionMode(ConvolutionMode.Same).build(), "block3_sepconv2_bn")
                .addVertex("add2", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block3_pool", "residual2")

                // residual3
                .addLayer("residual3_conv", new ConvolutionLayer.Builder(1, 1).stride(2, 2).nOut(728).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "add2")
                .addLayer("residual3", new BatchNormalization(), "residual3_conv")

                // block4
                .addLayer("block4_sepconv1_act", new ActivationLayer(Activation.RELU), "add2")
                .addLayer("block4_sepconv1", new SeparableConvolution2D.Builder(3, 3).nOut(728).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block4_sepconv1_act")
                .addLayer("block4_sepconv1_bn", new BatchNormalization(), "block4_sepconv1")
                .addLayer("block4_sepconv2_act", new ActivationLayer(Activation.RELU), "block4_sepconv1_bn")
                .addLayer("block4_sepconv2", new SeparableConvolution2D.Builder(3, 3).nOut(728).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block4_sepconv2_act")
                .addLayer("block4_sepconv2_bn", new BatchNormalization(), "block4_sepconv2")
                .addLayer("block4_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3, 3).stride(2, 2)
                        .convolutionMode(ConvolutionMode.Same).build(), "block4_sepconv2_bn")
                .addVertex("add3", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block4_pool", "residual3");

        // towers
        int residual = 3;
        int block = 5;
        for (int i = 0; i < 8; i++) {
            String previousInput = "add" + residual;
            String blockName = "block" + block;

            graph
                    .addLayer(blockName + "_sepconv1_act", new ActivationLayer(Activation.RELU), previousInput)
                    .addLayer(blockName + "_sepconv1", new SeparableConvolution2D.Builder(3, 3).nOut(728).hasBias(false)
                            .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), blockName + "_sepconv1_act")
                    .addLayer(blockName + "_sepconv1_bn", new BatchNormalization(), blockName + "_sepconv1")
                    .addLayer(blockName + "_sepconv2_act", new ActivationLayer(Activation.RELU), blockName + "_sepconv1_bn")
                    .addLayer(blockName + "_sepconv2", new SeparableConvolution2D.Builder(3, 3).nOut(728).hasBias(false)
                            .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), blockName + "_sepconv2_act")
                    .addLayer(blockName + "_sepconv2_bn", new BatchNormalization(), blockName + "_sepconv2")
                    .addLayer(blockName + "_sepconv3_act", new ActivationLayer(Activation.RELU), blockName + "_sepconv2_bn")
                    .addLayer(blockName + "_sepconv3", new SeparableConvolution2D.Builder(3, 3).nOut(728).hasBias(false)
                            .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), blockName + "_sepconv3_act")
                    .addLayer(blockName + "_sepconv3_bn", new BatchNormalization(), blockName + "_sepconv3")
                    .addVertex("add" + (residual + 1), new ElementWiseVertex(ElementWiseVertex.Op.Add), blockName + "_sepconv3_bn", previousInput);

            residual++;
            block++;
        }

        // residual12
        graph.addLayer("residual12_conv", new ConvolutionLayer.Builder(1, 1).stride(2, 2).nOut(1024).hasBias(false)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "add" + residual)
                .addLayer("residual12", new BatchNormalization(), "residual12_conv");

        // block13
        graph
                .addLayer("block13_sepconv1_act", new ActivationLayer(Activation.RELU), "add11")
                .addLayer("block13_sepconv1", new SeparableConvolution2D.Builder(3, 3).nOut(728).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block13_sepconv1_act")
                .addLayer("block13_sepconv1_bn", new BatchNormalization(), "block13_sepconv1")
                .addLayer("block13_sepconv2_act", new ActivationLayer(Activation.RELU), "block13_sepconv1_bn")
                .addLayer("block13_sepconv2", new SeparableConvolution2D.Builder(3, 3).nOut(1024).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block13_sepconv2_act")
                .addLayer("block13_sepconv2_bn", new BatchNormalization(), "block13_sepconv2")
                .addLayer("block13_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3, 3).stride(2, 2)
                        .convolutionMode(ConvolutionMode.Same).build(), "block13_sepconv2_bn")
                .addVertex("add12", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block13_pool", "residual12");

        // block14
        graph
                .addLayer("block14_sepconv1", new SeparableConvolution2D.Builder(3, 3).nOut(1536).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "add12")
                .addLayer("block14_sepconv1_bn", new BatchNormalization(), "block14_sepconv1")
                .addLayer("block14_sepconv1_act", new ActivationLayer(Activation.RELU), "block14_sepconv1_bn")
                .addLayer("block14_sepconv2", new SeparableConvolution2D.Builder(3, 3).nOut(2048).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block14_sepconv1_act")
                .addLayer("block14_sepconv2_bn", new BatchNormalization(), "block14_sepconv2")
                .addLayer("block14_sepconv2_act", new ActivationLayer(Activation.RELU), "block14_sepconv2_bn")

                .addLayer("avg_pool", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "block14_sepconv2_act")
                .addLayer("predictions", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX).build(), "avg_pool")

                .setOutputs("predictions")


        ;

        return graph;
    }

}