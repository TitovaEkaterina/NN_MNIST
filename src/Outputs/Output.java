/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Outputs;

import org.jblas.DoubleMatrix;

/**
 *
 * @author boyko_mihail
 */
public interface Output {

    public void evaluate(DoubleMatrix prev_layer_data, DoubleMatrix target);

    public DoubleMatrix backprop_data();

    public double loss();

}
