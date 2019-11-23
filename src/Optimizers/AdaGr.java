/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Optimizers;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 *
 * @author titova_ekaterina
 */
public class AdaGr implements Optimization {
    
    public double m_lrate;
    private double m_eps;   
    
    public AdaGr() {
        this.m_lrate = 0.01;
        this.m_eps = 1e-7;
    }
    
    public AdaGr(double m_lrate, double m_eps) {
        this.m_lrate = m_lrate;
        this.m_eps = m_eps;
    }

    @Override
    public void update(DoubleMatrix dvec, DoubleMatrix vec) {
        
        DoubleMatrix grad_square = MatrixFunctions.pow(dvec, 2);
        vec.data = vec.sub(dvec.mul(m_lrate).div(MatrixFunctions.sqrt(grad_square).add(m_eps))).data;
    }

}
