#include "utils.h"
#include "matrix.h"
#include <network.h>
#include <stddef.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>



NN nn_alloc(size_t *arch,size_t arch_count){
  assert(arch_count > 0);
  NN nn;
  
  nn.layer_count = arch_count - 1;
  nn.ws = NN_ALLOC(sizeof(*nn.ws)* nn.layer_count);
  nn.bs = NN_ALLOC(sizeof(*nn.bs)* nn.layer_count);
  nn.as = NN_ALLOC(sizeof(*nn.as)*(nn.layer_count + 1));
  assert(nn.ws != NULL);
  assert(nn.bs != NULL);
  assert(nn.as != NULL);

  nn.as[0] = mat_alloc(1, arch[0]);
  for(size_t i=1 ;i < arch_count ; ++i){
    nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
    nn.bs[i-1] = mat_alloc(1, arch[i]);
    nn.as[i] = mat_alloc(1, arch[i]);
  }
  return nn;  
}

void nn_print(NN nn, const char *name){
  printf(" %s = [ \n", name);

  char buf[256];

  for(size_t i =0 ; i < nn.layer_count ; ++i){
   snprintf(buf, sizeof(buf), "ws%zu", i);
   print_matrix(nn.ws[i], buf, 2);
   snprintf(buf, sizeof(buf), "bs%zu", i);
   print_matrix(nn.bs[i], buf, 2);
   snprintf(buf, sizeof(buf), "as%zu", i);
   print_matrix(nn.as[i], buf, 2);
  }
  
  printf("] \n");
};

void nn_rand(NN nn, float min, float max){
  for(size_t i =0 ; i < nn.layer_count; ++i){
    mat_rand(nn.ws[i], min, max);
    mat_rand(nn.bs[i], min, max);
    mat_rand(nn.as[i], min, max);
  }
}

void nn_forward(NN nn){
  for(size_t i=0 ; i < nn.layer_count ; ++i){
    mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
    mat_sum_over(nn.as[i+1], nn.bs[i]);
    mat_apply_sigmoid(nn.as[i+1]);
  }
}


float nn_cost(NN nn, Mat input, Mat output){
  assert(input.rows == output.rows);
  assert(NN_OUTPUT(nn).cols == output.cols);
  float c = 0;
  
  for(size_t i=0; i < input.rows ; ++i){
    Mat x = mat_row(input, i);
    Mat y = mat_row(output, i);

    mat_copy(NN_INPUT(nn), x);
    nn_forward(nn);
    for(size_t j =0 ; j< output.cols; ++j){
      float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
      c += d*d;
    }
  }

  return c/input.rows;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat input, Mat output){
  float c = nn_cost(nn, input, output);
  float saved = 0;

  for(size_t i=0 ; i < nn.layer_count ; ++i){
    for(size_t j=0 ; j < nn.ws[i].rows ; ++j){
      for(size_t k=0; k < nn.ws[i].cols; ++k){
        saved = MAT_AT(nn.ws[i], j, k);
        MAT_AT(nn.ws[i], j, k) += eps;
        MAT_AT(g.ws[i], j,k) =  (nn_cost(nn, input, output) - c)/eps;
        MAT_AT(nn.ws[i], j, k) = saved;
      }
    }
    for(size_t j=0 ; j < nn.bs[i].rows ; ++j){
      for(size_t k=0; k < nn.bs[i].cols; ++k){
        saved = MAT_AT(nn.bs[i], j, k);
        MAT_AT(nn.bs[i], j, k) += eps;
        MAT_AT(g.bs[i], j,k) =  (nn_cost(nn, input, output) - c)/eps;
        MAT_AT(nn.bs[i], j, k) = saved;
      }
    }
  }
}

void learn(NN nn, NN g, float rate){
  for(size_t i=0 ; i < nn.layer_count ; ++i){
    for(size_t j=0 ; j < nn.ws[i].rows ; ++j){
      for(size_t k=0; k < nn.ws[i].cols; ++k){
        MAT_AT(nn.ws[i], j,k) -=  rate*(MAT_AT(g.ws[i], j, k));
      }
    }
    for(size_t j=0 ; j < nn.bs[i].rows ; ++j){
      for(size_t k=0; k < nn.bs[i].cols; ++k){
        MAT_AT(nn.bs[i], j,k) -=  rate*(MAT_AT(g.bs[i], j, k));
      }
    }
  }

}


