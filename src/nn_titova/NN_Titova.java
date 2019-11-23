/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn_titova;

import Activations.Sigmoid;
import Activations.Softmax;
import Optimizers.AdaGr;
import Outputs.MultiClassEntropy;
import java.util.List;
import layers.DenseLayer;
import layers.DropOut;
import layers.Layer;
import org.jblas.DoubleMatrix;

/**
 *
 * @author boyko_mihail
 */
public class NN_Titova {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        double[][] labels = MnistReader.readLabels("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/ML_HomeWork_NeuralNetwork/Data/train-labels.idx1-ubyte");
        double[][] images = MnistReader.readImages("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/ML_HomeWork_NeuralNetwork/Data/train-images.idx3-ubyte");

        int total_size = 60000;
        int test_size = 10000;

        double[][] data_train_X = new double[total_size - test_size][];
        double[][] data_test_X = new double[test_size][];
        double[][] data_train_Y = new double[total_size - test_size][10];
        double[][] data_test_Y = new double[test_size][10];

        int indexTrain = 0;
        int indexTest = 0;
        for (int j = 0; j < images.length; ++j) {
            if (j >= test_size) {
                data_train_X[indexTrain] = images[j];
                data_train_Y[indexTrain] = labels[j];
                ++indexTrain;
            } else {
                data_test_X[indexTest] = images[j];
                data_test_Y[indexTest] = labels[j];
                ++indexTest;
            }
        }

        DoubleMatrix data_train_X_matrix = new DoubleMatrix(data_train_X);
        DoubleMatrix data_test_X_matrix = new DoubleMatrix(data_test_X);
        DoubleMatrix data_train_Y_matrix = new DoubleMatrix(data_train_Y);
        DoubleMatrix data_test_Y_matrix = new DoubleMatrix(data_test_Y);

        Network net = new Network();

        Layer layer1 = new DenseLayer(784, 400, new Sigmoid());
        DropOut layer2 = new DropOut();
        layer2.set_dropout_ratio((float) 0.7);
        Layer layer3 = new DenseLayer(400, 120, new Sigmoid());
        
        Layer layer7 = new DenseLayer(400, 10, new Softmax());
        net.set_output(new MultiClassEntropy());
        net.add_layer(layer1);
        net.add_layer(layer2);
//        net.add_layer(layer3);
        net.add_layer(layer7);

        AdaGr opt = new AdaGr();
        opt.m_lrate = 0.03;

        net.init(0, 0.3);

        net.fit(opt, data_train_X_matrix.transpose(), data_train_Y_matrix.transpose(), 1000, 7);

        DoubleMatrix pred = net.predict(data_test_X_matrix.transpose());
        ConfMatrix confusion = new ConfMatrix(data_test_Y_matrix.transpose(), pred);
        confusion.print();

    }

}
