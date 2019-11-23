/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Optimizers;

import org.jblas.DoubleMatrix;

/**
 *
 * @author boyko_mihail
 */
public interface Optimization {
    public void update(DoubleMatrix dvec, DoubleMatrix vec);
}
