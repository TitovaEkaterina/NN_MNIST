/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn_titova;

import Optimizers.Optimization;
import Outputs.Output;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import layers.Layer;
import org.jblas.DoubleMatrix;

/**
 *
 * @author titova_ekaterina
 */
public class Network {

    List<Layer> m_layers;
    Output m_output;

    public Network() {
        m_layers = new ArrayList();
    }

    public void add_layer(Layer layer) {
        m_layers.add(layer);
    }

    public void set_output(Output output) {
        m_output = output;
    }

    public int num_layers() {
        return m_layers.size();
    }

    public void init(double mu, double sigma) {

        int nlayer = num_layers();

        for (int i = 0; i < nlayer; i++) {
            m_layers.get(i).init(mu, sigma);
        }
    }

    public Output get_output() {
        return m_output;
    }

    public boolean fit(Optimization opt, DoubleMatrix x, DoubleMatrix y,
            int batch_size, int epoch) {

        int nlayer = num_layers();

        if (nlayer <= 0) {
            return false;
        }

        List<Integer> listOfIndexes = new ArrayList<>(0);
        for (int i = 0; i < x.columns; ++i) {
            listOfIndexes.add(i);
        }

        int nbatch = x.columns / batch_size;

        for (int k = 0; k < epoch; k++) {

            Collections.shuffle(listOfIndexes, new Random());

            for (int i = 0; i < x.columns; i += batch_size) {

                double[][] data_bach_X = new double[(i + batch_size) < listOfIndexes.size() ? batch_size : listOfIndexes.size() - i][];
                double[][] data_bach_Y = new double[(i + batch_size) < listOfIndexes.size() ? batch_size : listOfIndexes.size() - i][10];

                int index = 0;

                for (int t = i; t < (i + batch_size) && t < listOfIndexes.size(); ++t) {
                    data_bach_X[index] = x.getColumn(listOfIndexes.get(t)).data;
                    data_bach_Y[index] = y.getColumn(listOfIndexes.get(t)).data;
                    index++;
                }

                DoubleMatrix bachX = new DoubleMatrix(data_bach_X);
                DoubleMatrix bachY = new DoubleMatrix(data_bach_Y);

                this.forward(bachX.transpose());
                this.backprop(bachX.transpose(), bachY.transpose());
                this.update(opt);
                double loss = this.m_output.loss();
                System.out.println("Epoch " + k + ", batch " + i / batch_size + " Loss = " + loss );

            }
        }

        return true;
    }

    public DoubleMatrix predict(DoubleMatrix x) {

        int nlayer = num_layers();

        if (nlayer <= 0) {
            return new DoubleMatrix();
        }
        this.forward(x);
        return m_layers.get(nlayer - 1).output();
    }

    private void forward(DoubleMatrix input) {

        int nlayer = num_layers();

        if (nlayer <= 0) {
            return;
        }

        m_layers.get(0).forward(input);

        for (int i = 1; i < nlayer; i++) {
            m_layers.get(i).forward(m_layers.get(i - 1).output());
        }

    }

    private void backprop(DoubleMatrix input, DoubleMatrix target) {

        int nlayer = num_layers();

        if (nlayer <= 0) {
            return;
        }

        Layer first_layer = m_layers.get(0);
        Layer last_layer = m_layers.get(nlayer - 1);
        m_output.evaluate(last_layer.output(), target);

        if (nlayer == 1) {
            first_layer.backprop(input, m_output.backprop_data());
            return;
        }
        last_layer.backprop(m_layers.get(nlayer - 2).output(), m_output.backprop_data());

        for (int i = nlayer - 2; i > 0; i--) {
            m_layers.get(i).backprop(m_layers.get(i - 1).output(),
                    m_layers.get(i + 1).backprop_data());
        }

        first_layer.backprop(input, m_layers.get(1).backprop_data());
    }

    private void update(Optimization opt) {

        int nlayer = num_layers();

        if (nlayer <= 0) {
            return;
        }

        for (int i = 0; i < nlayer; i++) {
            m_layers.get(i).update(opt);
        }
    }

}
