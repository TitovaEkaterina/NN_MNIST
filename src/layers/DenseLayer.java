/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package layers;

import Activations.ActivationFunc;
import Activations.ReLU;
import Optimizers.Optimization;
import org.jblas.DoubleMatrix;
import java.util.Random;

/**
 *
 * @author titova_ekaterina
 */
public class DenseLayer implements Layer {

    private DoubleMatrix m_weight; // W
    private DoubleMatrix m_bias; // Bias
    private DoubleMatrix m_dw; // Derivative of W
    private DoubleMatrix m_db; // Derivative of b
    private DoubleMatrix m_z; // z = W' * in + b
    private DoubleMatrix m_a; // a = act(z)
    private DoubleMatrix m_din; // Derivative of the input.

    private ActivationFunc activation;

    private int m_in_size;
    private int m_out_size;

    public DenseLayer(int in_size, int out_size, ActivationFunc activation) {
        this.activation = activation;
        this.m_in_size = in_size;
        this.m_out_size = out_size;
    }

    public DenseLayer(int in_size, int out_size) {
        this.activation = new ReLU();
        this.m_in_size = in_size;
        this.m_out_size = out_size;
    }

    @Override
    public int in_size() {
        return m_in_size;
    }

    @Override
    public int out_size() {
        return m_out_size;
    }

    @Override
    public void init(double mu, double sigma) {

        m_weight = new DoubleMatrix(m_in_size, m_out_size);
        m_bias = new DoubleMatrix(m_out_size, 1);
        m_dw = new DoubleMatrix(m_in_size, m_out_size);
        m_db = new DoubleMatrix(m_out_size, 1);
        Random fRandom = new Random();

        for (int i = 0; i < m_weight.rows; ++i) {
            for (int j = 0; j < m_weight.columns; ++j) {
                m_weight.put(i, j, fRandom.nextGaussian() * sigma + mu);
            }
        }
        for (int i = 0; i < m_bias.rows; ++i) {
            for (int j = 0; j < m_bias.columns; ++j) {
                m_bias.put(i, j, fRandom.nextGaussian() * sigma + mu);
            }
        }
    }

    @Override
    public void forward(DoubleMatrix prev_layer_data) {

        int nobj = prev_layer_data.columns;
        m_z = m_weight.transpose().mmul(prev_layer_data);
        m_z.addColumnVector(m_bias);
        m_a = new DoubleMatrix(m_out_size, nobj);
        activation.activate(m_z, m_a);

    }

    @Override
    public DoubleMatrix output() {
        return m_a;
    }

    @Override
    public void backprop(DoubleMatrix prev_layer_data, DoubleMatrix next_layer_data) {

        int nobj = prev_layer_data.columns;
        DoubleMatrix dLz = m_z;
        activation.calculate_jacobian(m_z, m_a, next_layer_data, dLz);
        m_dw = (prev_layer_data.mmul(dLz.transpose())).div(nobj);
        m_db = dLz.rowMeans();
        m_din = m_weight.mmul(dLz);
        
         
    }

    @Override
    public void update(Optimization opt) {
        DoubleMatrix dw = m_dw;
        DoubleMatrix db = m_db;
        DoubleMatrix w = m_weight;
        DoubleMatrix b = m_bias;
        opt.update(dw, w);
        opt.update(db, b);
    }

    @Override
    public String getNameOfLayer() {
        return "FullyConnected";
    }

    @Override
    public DoubleMatrix backprop_data() {
         return m_din;
    }

}
