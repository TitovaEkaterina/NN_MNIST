/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package layers;

import Optimizers.Optimization;
import org.jblas.DoubleMatrix;

/**
 *
 * @author titova_ekaterina
 */
public interface Layer {
    
    public int in_size();

    public int out_size();

    public void init(double mu, double sigma);

    public void forward(DoubleMatrix prev_layer_data);

    public DoubleMatrix output();

    public void backprop(DoubleMatrix prev_layer_data, DoubleMatrix next_layer_data);
    
    public DoubleMatrix backprop_data();

    public void update(Optimization opt);
    
    public String getNameOfLayer();
    
}
