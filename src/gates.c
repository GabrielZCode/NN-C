#include "utils.h"
#include "matrix.h"
#include "network.h"
#include <stddef.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

float train_data_gates_XOR[] = {
  0, 0, 0,
  1, 0, 1,
  0, 1, 1,
  1, 1, 0,
};


float train_data_gates_OR[] = {
  0, 0, 0,
  1, 0, 1,
  0, 1, 1,
  1, 1, 1,
};

float train_data_gates_AND[] = {
  0, 0, 0,
  1, 0, 0,
  0, 1, 0,
  1, 1, 1,
};

float train_data_gates_NAND[] = {
  0, 0, 1,
  1, 0, 1,
  0, 1, 1,
  1, 1, 0,
};

#define td train_data_gates_NAND

int train_gates(){
  size_t samples = sizeof(td)/sizeof(td[0])/3;
  float eps =  1e-1;
  float rate =  1e-1;
  Mat input = {
    .rows = samples,
    .cols = 2,
    .stride = 3,
    .values = td
  };
  Mat output = { 
    .rows = samples,
    .cols = 1,
    .stride = 3,
    .values = &td[2]
  };

  size_t arch[] = {2, 2, 1};
  NN nn = NN_ALLOC_MAT(arch);
  NN g = NN_ALLOC_MAT(arch);
  nn_rand(nn, 0.f, 1.f);
  
//  printf("cost %f\n",nn_cost(nn,input, output));
  for(size_t i =0 ; i < 100000*10 ; i++){
    nn_finite_diff(nn, g, eps, input, output);
    learn(nn, g, rate);
  //  printf("cost %f\n",nn_cost(nn,input, output));
  }

  NN_PRINT(nn);
  for(size_t i=0 ; i < 2 ; ++i){
    for(size_t j =0 ; j< 2 ; ++j){
      MAT_AT(NN_INPUT(nn), 0, 0) = i;
      MAT_AT(NN_INPUT(nn), 0, 1) = j;
      nn_forward(nn);
      printf(" %zu * %zu == %f \n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
    }
  }
  return 0;
}
