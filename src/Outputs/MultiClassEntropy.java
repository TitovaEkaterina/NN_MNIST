/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Outputs;

import org.jblas.DoubleMatrix;

/**
 *
 * @author titova_ekaterina
 */
public class MultiClassEntropy implements Output {

    private DoubleMatrix m_din;

    @Override
    public double loss() {
        double res = 0;
        int nelem = m_din.length;
        double[] din_data = m_din.data;

        for (int i = 0; i < nelem; i++) {
            if (din_data[i] < 0) {
                res += Math.log(-din_data[i]);
            }
        }

        return res / m_din.columns;
    }

    @Override
    public void evaluate(DoubleMatrix prev_layer_data, DoubleMatrix target) {

        int nobs = prev_layer_data.columns;
        int nclass = prev_layer_data.rows;
        m_din = new DoubleMatrix(nclass, nobs);
        m_din.data = target.div(prev_layer_data).mul(-1).data;
    }

    @Override
    public DoubleMatrix backprop_data() {
        return m_din;
    }

}
