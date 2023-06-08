#include "utils.h"
#include "matrix.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>


float train_data_gates_OR[][3] = {
  {0, 0, 0},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 1},
};

float train_data_gates_AND[][3] = {
  {0, 0, 0},
  {1, 0, 0},
  {0, 1, 0},
  {1, 1, 1},
};

float train_data_gates_NAND[][3] = {
  {0, 0, 1},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 0},
};

#define rows (sizeof(train_data_gates_OR)/sizeof(train_data_gates_OR[0]))
#define cols (sizeof(train_data_gates_OR[0])/sizeof(train_data_gates_OR[0][0]))

float result_gates[rows] = {};

int train_gates(){
  float learning_rate = 1e-1;
  float eps = 1e-1;
  float* matrix_ptr = &train_data_gates_NAND[0][0];
  float weigth = rand_float();
  float bias = rand_float();
  float weigth2 = rand_float();
  float bias2 = 0.0f;
//  print_matrix(rows,cols, matrix_ptr);
  
  for(size_t i =0 ; i < 240000 ; ++i){
      float default_cost = cost_function_gates(rows, cols, matrix_ptr, weigth, weigth2, bias);
      float dw = (cost_function_gates(rows, cols, matrix_ptr, (weigth + eps), weigth2, bias) - default_cost)/eps;
      float dw2 = (cost_function_gates(rows, cols, matrix_ptr, weigth, (weigth2+eps), bias) - default_cost)/eps;
      float db = (cost_function_gates(rows, cols, matrix_ptr, weigth, weigth2, (bias+ eps)) - default_cost)/eps;
    //  float db2 = (cost_function(rows, cols, matrix_ptr, weigth2, (bias2+ eps)) - default_cost2)/eps;
      weigth -= (dw*learning_rate);
      bias -= (db*learning_rate);
      weigth2 -= (dw2*learning_rate);
    //  bias2 -= (db2*learning_rate);

      printf("cost: %f w: %f, w2: %f, b: %f\n" , cost_function_gates(rows,cols,matrix_ptr,weigth, weigth2, bias), weigth, weigth2 , bias);
  }
  
  printf("Weigth: %f \n", weigth);
  printf("Weigth2: %f \n", weigth2);
  printf("Bias: %f \n", bias);
  dot_product_gates(rows,cols, matrix_ptr, weigth, weigth2,bias,  &result_gates[0]);
  printf(" Result: [ \n");
  for( int i=0; i< rows ; i++){
     printf("\t %f \n", result_gates[i]);
  }
  printf("] \n");
  
  return 0;
}
