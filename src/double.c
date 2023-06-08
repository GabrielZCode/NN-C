#include "utils.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train_data_double[][2] = {
  {0, 0},
  {1, 2},
  {2, 4},
  {3, 6},
  {4, 8},
  {5, 10},
  {6, 12},
  {7, 14},
  {8, 16},
  {9, 18},
  {10, 20}
};

#define rows (sizeof(train_data_double)/sizeof(train_data_double[0]))
#define cols (sizeof(train_data_double[0])/sizeof(train_data_double[0][0]))

float result[rows] = {};

int train_double(){
  //srand(69);
  float learning_rate = 1e-3;
  float weigth = rand_float() *10.0f;
  float bias = rand_float() *5.0f;
  float eps = 1e-3;
  float* matrix_ptr = &train_data_double[0][0];


  printf("rows %lu \n" , rows);
  printf("cols %lu \n",cols);
 
  for(size_t i =0 ; i < 12000 ; ++i){
      float default_cost = cost_function(rows, cols, matrix_ptr, weigth, bias);
      float dw = (cost_function(rows, cols, matrix_ptr, (weigth + eps), bias) - default_cost)/eps;
      float db = (cost_function(rows, cols, matrix_ptr, weigth, (bias+ eps)) - default_cost)/eps;
      weigth -= (dw*learning_rate);
      bias -= (db*learning_rate);
      printf("cost: %f w: %f, b: %f\n" , cost_function(rows,cols,matrix_ptr,weigth, bias), weigth, bias);
  }
  printf("Weigth: %f \n", weigth);
  printf("Bias: %f \n", bias);
  dot_product(rows,cols, matrix_ptr, weigth, bias,  &result[0]);
  printf(" Result: [ \n");
  for( int i=0; i< rows ; i++){
     printf("\t %f \n", result[i]);
  }
  printf("] \n");
  return 0;
}
