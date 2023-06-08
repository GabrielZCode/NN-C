#include "utils.h"
#include "matrix.h"
#include <stddef.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

float td[] = {
  0, 0, 0,
  1, 0, 1,
  0, 1, 1,
  1, 1, 0,
};

typedef struct {
  size_t layer_count;
  Mat input;
  Mat w ,b, a; 
  Mat w2 ,b2, a2;
} NN;


float forward_Xor(NN m){
  mat_dot(m.a, m.input, m.w);
  mat_sum_over(m.a, m.b);
  mat_apply_sigmoid(m.a);
  

  mat_dot(m.a2, m.a, m.w2);
  mat_sum_over(m.a2, m.b2);
  mat_apply_sigmoid(m.a2);
  return *m.a2.values;
}

float cost_Xor(NN m , Mat input, Mat output){
  assert(input.rows == output.rows);
  float cost = 0;
  for(size_t i=0 ; i < input.rows ; ++i){
    Mat x = mat_row(input, i);
    Mat y = mat_row(output, i);
    mat_copy(m.input,x);
    forward_Xor(m);
    for(size_t j =0 ; j< output.cols; ++j){
      float d = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
      cost += d*d;
    }
  }
  return cost/input.rows;
}

void finite_diff(NN m, NN g, float eps, Mat input, Mat output){
  float saved;

  float c = cost_Xor(m, input, output);
  for(size_t i=0 ; i < m.w.rows ; ++i){
    for(size_t j=0 ; j < m.w.cols ; ++j){
      saved = MAT_AT(m.w, i, j);
      MAT_AT(m.w, i, j) += eps;
      MAT_AT(g.w, i, j) = (cost_Xor(m, input, output) - c)/eps;
      MAT_AT(m.w, i, j) = saved;
    }

  }
  for(size_t i=0 ; i < m.b.rows ; ++i){
    for(size_t j=0 ; j < m.b.cols ; ++j){
      saved = MAT_AT(m.b, i, j);
      MAT_AT(m.b, i, j) += eps;
      MAT_AT(g.b, i, j) = (cost_Xor(m, input, output) - c)/eps;
      MAT_AT(m.b, i, j) = saved;
    }
  }
  for(size_t i=0 ; i < m.w2.rows ; ++i){
    for(size_t j=0 ; j < m.w2.cols ; ++j){
      saved = MAT_AT(m.w2, i, j);
      MAT_AT(m.w2, i, j) += eps;
      MAT_AT(g.w2, i, j) = (cost_Xor(m, input, output) - c)/eps;
      MAT_AT(m.w2, i, j) = saved;
    }
  }
  for(size_t i=0 ; i < m.b2.rows ; ++i){
    for(size_t j=0 ; j < m.b2.cols ; ++j){
      saved = MAT_AT(m.b2, i, j);
      MAT_AT(m.b2, i, j) += eps;
      MAT_AT(g.b2, i, j) = (cost_Xor(m, input, output) - c)/eps;
      MAT_AT(m.b2, i, j) = saved;
    }
  }
}

void learn(NN m, NN g, float rate){
  for(size_t i=0 ; i < m.w.rows ; ++i){
    for(size_t j=0 ; j < m.w.cols ; ++j){
      MAT_AT(m.w, i, j) -= rate*(MAT_AT(g.w, i, j));
    }

  }
  for(size_t i=0 ; i < m.b.rows ; ++i){
    for(size_t j=0 ; j < m.b.cols ; ++j){
      MAT_AT(m.b, i, j) -= rate*(MAT_AT(g.b, i, j));
    }
  }
  for(size_t i=0 ; i < m.w2.rows ; ++i){
    for(size_t j=0 ; j < m.w2.cols ; ++j){
      MAT_AT(m.w2, i, j) -= rate*(MAT_AT(g.w2, i, j));
    }
  }
  for(size_t i=0 ; i < m.b2.rows ; ++i){
    for(size_t j=0 ; j < m.b2.cols ; ++j){
      MAT_AT(m.b2, i, j) -= rate*(MAT_AT(g.b2, i, j));
    }
  }

}

NN xor_alloc(){
  return (NN) {
    .input = mat_alloc(1, 2),
    .w = mat_alloc(2, 2),
    .b = mat_alloc(1, 2),
    .a = mat_alloc(1, 2),
    .w2 = mat_alloc(2, 1),
    .b2 = mat_alloc(1, 1),
    .a2 = mat_alloc(1, 1),
  };  
}

int train_xor(){
  size_t samples = sizeof(td)/sizeof(td[0])/3;
  NN m = xor_alloc(); 
  NN g = xor_alloc();
  float eps = 1e-1;
  mat_rand(m.w, 0, 1);
  mat_rand(m.w2, 0, 1);
  mat_rand(m.b, 0, 1);
  mat_rand(m.b2, 0, 1);

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
  print_matrix_and_name(input);
  print_matrix_and_name(output);
 // printf("cost %f \n", cost_Xor(m,input, output));
  for(size_t i= 0 ; i < 1000000; ++i){
      finite_diff(m,g,eps,input, output);
      learn(m, g, eps);
   //   printf("cost %f \n", cost_Xor(m,input, output));
  }
  for(size_t i= 0 ; i < 2; ++i){
    for(size_t j =0; j < 2; ++j){
      MAT_AT(m.input , 0, 0) = i;
      MAT_AT(m.input , 0, 1) = j;
      forward_Xor(m);
      float y = *(m.a2.values);
      printf(" %zu * %zu = %f \n", i, j, y);
    }
  }
 // printf("cost %f \n", cost_Xor(m,input, output));
  print_matrix_and_name(m.w);
  print_matrix_and_name(m.w2);
  print_matrix_and_name(m.b);
  print_matrix_and_name(m.b2);
 // print_matrix_and_name(m.a2);
//  train_xor();
  return 0;
}
