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
public class ReLU implements ActivationFunc{

    @Override
    public void activate(DoubleMatrix Z, DoubleMatrix A) {
        A.data = Z.max(0).data;
    }

    @Override
    public void calculate_jacobian(DoubleMatrix Z, DoubleMatrix A, DoubleMatrix F, DoubleMatrix G) {        
         G.data = F.select(A.max(0)).data;
    }
    
}
