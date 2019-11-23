/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Activations;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 *
 * @author boyko_mihail
 */
public class Sigmoid implements ActivationFunc {

    @Override
    public void activate(DoubleMatrix Z, DoubleMatrix A) {

        DoubleMatrix ones = new DoubleMatrix(Z.rows, Z.columns);
        ones.fill(1);
        A.data = ones.div(MatrixFunctions.exp(Z.mul(-1)).add(1)).data;
    }

    @Override
    public void calculate_jacobian(DoubleMatrix Z, DoubleMatrix A, DoubleMatrix F, DoubleMatrix G) {
        DoubleMatrix ones = new DoubleMatrix(A.rows, A.columns);
        ones.fill(1);
        G.data = A.mul(ones.sub(A)).mul(F).data;
    }

}
