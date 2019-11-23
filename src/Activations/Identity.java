/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Activations;

import org.jblas.DoubleMatrix;

/**
 *
 * @author boyko_mihail
 */
public class Identity implements ActivationFunc{

    @Override
    public void activate(DoubleMatrix Z, DoubleMatrix A) {
        A.data = Z.data;
    }

    @Override
    public void calculate_jacobian(DoubleMatrix Z, DoubleMatrix A, DoubleMatrix F, DoubleMatrix G) {
         G.data = F.data;
    }
    
}
