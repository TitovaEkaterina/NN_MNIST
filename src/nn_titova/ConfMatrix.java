/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn_titova;

import org.jblas.DoubleMatrix;

/**
 *
 * @author titova_ekaterina
 */
public class ConfMatrix {
    
    public int _classes;
    public int _samples;
    public DoubleMatrix _cm;
    
    public ConfMatrix(DoubleMatrix targets, DoubleMatrix outputs) {
        confusion(targets, outputs);
    }
    
    private void confusion(DoubleMatrix targets, DoubleMatrix outputs) {

        int numClasses = targets.rows;

        int numSamples = targets.columns;

        _classes = numClasses;
        _samples = numSamples;

        for (int col = 0; col < numSamples; col++) {
            double max = outputs.get(0, col);
            int ind = 0;

            for (int row = 1; row < numClasses; row++) {
                if (outputs.get(row, col) > max) {
                    max = outputs.get(row, col);
                    ind = row;
                }
                outputs.put(row, col, 0.0);
            }
            outputs.put(0, col, 0.0); 
            outputs.put(ind, col, 1.0); 
        }

        // Confusion matrix
        DoubleMatrix cm = new DoubleMatrix(numClasses, numClasses);
        cm.fill(0);

        DoubleMatrix i = new DoubleMatrix(numSamples, 1);
        DoubleMatrix j = new DoubleMatrix(numSamples,1);

        for (int col = 0; col < numSamples; col++) {
            for (int row = 0; row < numClasses; row++) {
                
                if (targets.get(row, col) == 1.0) {
                    i.put(col, 0, row);
                    break;
                }
            }
        }

        for (int col = 0; col < numSamples; col++) {
            for (int row = 0; row < numClasses; row++) {
                if (outputs.get(row, col) == 1.0) {
                    j.put(col, 0, row);
                    break;
                }
            }
        }

        for (int col = 0; col < numSamples; col++) {
            cm.put((int)i.get(col, 0), (int)j.get(col, 0), cm.get((int)i.get(col, 0), (int)j.get(col, 0)) + 1);
        }
        
        _cm = cm;
    }
    
    void print() {
        System.out.println("Confusion Results\n");
        for (int row = 0; row < _classes; row++) {
            System.out.print("\t\t");
            for (int col = 0; col < _classes; col++) {
                System.out.print((int)_cm.get(row, col) + " ");
            }
            System.out.println();
        }
    }
    
}
