/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package layers;

import Optimizers.Optimization;
import java.util.Random;
import org.jblas.DoubleMatrix;

/**
 *
 * @author boyko_mihail
 */
public class DropOut implements Layer {

    DoubleMatrix dropout_mask;
    float dropout_ratio = 1.0f;

    DoubleMatrix m_a;
    DoubleMatrix m_din;

    private int m_in_size;
    private int m_out_size;

    public DropOut() {
        m_in_size = 0;
        m_out_size = 0;
    }

    @Override
    public int in_size() {
        return m_in_size;
    }

    @Override
    public int out_size() {
        return m_out_size;
    }

    public void set_dropout_ratio(float dropout_ratio) {
        this.dropout_ratio = dropout_ratio;
    }

    @Override
    public void init(double mu, double sigma) {
    }

    @Override
    public void forward(DoubleMatrix prev_layer_data) {
        dropout_mask = getMask(prev_layer_data.rows, prev_layer_data.columns, dropout_ratio);
        m_a = prev_layer_data.mul(dropout_mask);
    }

    @Override
    public DoubleMatrix output() {
        return m_a;
    }

    @Override
    public void backprop(DoubleMatrix prev_layer_data, DoubleMatrix next_layer_data) {
        m_din = next_layer_data.mul(dropout_mask);
    }

    @Override
    public void update(Optimization opt) {
    }

    @Override
    public String getNameOfLayer() {
        return "DropOut";
    }

    private DoubleMatrix getMask(int rows, int cols, double ratio) {
        DoubleMatrix result = new DoubleMatrix(rows, cols);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                boolean b = Math.log(Math.random()) > Math.log(1.0 - ratio);
                result.put(r, c, b ? 1 : 0);
            }
        }

        return result;
    }

    @Override
    public DoubleMatrix backprop_data() {
        return m_din;
    }

}
