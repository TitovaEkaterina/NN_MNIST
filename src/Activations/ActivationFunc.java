/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Activations;

import org.jblas.DoubleMatrix;

/**
 *
 * @author titova_ekaterina
 */
public interface ActivationFunc {

    public void activate(DoubleMatrix Z, DoubleMatrix A);

    public void calculate_jacobian(DoubleMatrix Z, DoubleMatrix A,
            DoubleMatrix F, DoubleMatrix G);

}
