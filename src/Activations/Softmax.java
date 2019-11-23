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
public class Softmax implements ActivationFunc{

    @Override
    public void activate(DoubleMatrix Z, DoubleMatrix A) {
        DoubleMatrix AA = MatrixFunctions.exp(Z.subRowVector(Z.columnMaxs()));
        DoubleMatrix colsums = AA.columnSums();
        A.data = AA.divRowVector(colsums).data;
    }

    @Override
    public void calculate_jacobian(DoubleMatrix Z, DoubleMatrix A, DoubleMatrix F, DoubleMatrix G) {
        DoubleMatrix a_dot_f = A.mul(F).columnSums();
        G.data = A.mul(F.subRowVector(a_dot_f)).data;
    }
    
}
