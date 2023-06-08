#include "utils.h"
#include "matrix.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifndef NN_ALLOC
#define NN_ALLOC malloc
#endif /* !NN_ALLOC */



void print_matrix(Mat m, const char *name){
  size_t rows = m.rows;
  size_t cols = m.cols;
  float* matrix = m.values;
  printf(" %s = [ \n", name); 
  for(int i=0 ; i< rows; ++i){
      printf("\t[");
     for(int j =0; j< cols ; ++j){
      float elem = (MAT_AT(m, i, j));
      //float elem = *(matrix + (i*cols) + j);
      _Bool last_elem = (j+1) == cols;
      last_elem ? printf(" %f ", elem) : printf(" %f", elem); 
     }
      printf("]\n");
  }
  printf("] \n"); 
};

float cost_function(int rows, int cols, float* matrix, float w, float b){
  float cost = 0.0f;
  for(int i=0 ; i< rows; ++i){
      //printf("{ \n");
      float elem0 = *(matrix + (i*cols));
      //printf("\t x: %f \n", elem0);
      float y = sigmoid((elem0 * w) + b);
      //printf("\t y: %f \n", y);
      float elem1 = *(matrix + (i*cols) + 1);
      float diff = y - elem1;
      cost += diff*diff;
      //printf("\t elem: %f, expected: %f , result: %f, diff: %f \n", elem0, elem1, y, diff);
      //printf("} \n");
  }

  cost /= rows;
  return cost;
};
float cost_function_gates(int rows, int cols, float* matrix, float w, float w2, float bias){
  float cost = 0.0f;
  for(int i=0 ; i< rows; ++i){
      //printf("{ \n");
      float elem0 = *(matrix + (i*cols));
      float elem1 = *(matrix + (i*cols)+1);
      //printf("\t x: %f \n", elem0);
      float y = sigmoid((elem0 * w) + (elem1 * w2) + bias);
      //printf("\t y: %f \n", y);
      float elem2 = *(matrix + (i*cols) + 2);
      float diff = y - elem2;
      cost += diff*diff;
      //printf("\t elem: %f, expected: %f , result: %f, diff: %f \n", elem0, elem1, y, diff);
      //printf("} \n");
  }

  cost /= rows;
  return cost;
};

float* dot_product(int rows, int cols, float* matrix, float w, float b, float* result){
  for(int i=0; i< rows; ++i){
    float x = (*(matrix + (i*cols)));
    float y = (x* w) +b;
    (*(result + (i))) = y;
  }
  return result;
}

float* dot_product_gates(int rows, int cols, float* matrix, float w, float w2, float bias, float* result){

  for(int i=0; i< rows; ++i){
    float x = (*(matrix + (i*cols)));
    float x2 = (*(matrix + (i*cols)+1));
    float y = sigmoid((x * w) + (x2* w2)+ bias);
    printf("y: %f \n", y);
    (*(result + (i))) = y;
  }
  return result;
}


Mat mat_alloc(size_t rows, size_t cols){
  Mat m = (Mat) {
    .rows = rows,
    .cols = cols,
    .stride = cols
  };
  m.values = (float*)(NN_ALLOC(sizeof(*m.values)*rows*cols));
  assert(m.values != NULL);
  return m;  
}

void mat_fill(Mat m, float x){
  for(size_t i = 0; i < m.rows ; ++i){
    for(size_t j =0 ; j< m.cols ; ++j){
      MAT_AT(m, i, j) = x;
    }
  }
}

void mat_rand(Mat m, float low, float high){
  for(size_t i = 0; i < m.rows ; ++i){
    for(size_t j =0 ; j< m.cols ; ++j){
      MAT_AT(m, i, j) = rand_float()*(high - low) + low;
    }
  }
}

void mat_sum(Mat result_dst, Mat a, Mat b){
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  for(size_t i = 0; i < a.rows ; ++i){
    for(size_t j =0 ; j< a.cols ; ++j){
      MAT_AT(result_dst, i, j) =  MAT_AT(a, i, j) + MAT_AT(b, i , j);
    }
  }
}

void mat_sum_over(Mat result_dst, Mat a){
  assert(a.rows == result_dst.rows);
  assert(a.cols == result_dst.cols);
  for(size_t i = 0; i < a.rows ; ++i){
    for(size_t j =0 ; j< a.cols ; ++j){
      MAT_AT(result_dst, i, j) +=  MAT_AT(a, i, j);
    }
  }
}

void mat_dot(Mat result_dst, Mat a, Mat b){
  assert(a.cols == b.rows);
  size_t inner_size = a.cols;
  assert(result_dst.rows == a.rows);
  assert(result_dst.cols == b.cols);
  for(size_t i = 0; i < result_dst.rows ; ++i){
    for(size_t j =0 ; j< result_dst.cols ; ++j){
      MAT_AT(result_dst, i, j) = 0; 
      for(size_t k=0; k< inner_size ; k++){
          MAT_AT(result_dst, i, j) +=  MAT_AT(a, i, k) * MAT_AT(b, k , j);
      }
    }
  }
}


void mat_apply_sigmoid(Mat a){
  for(size_t i = 0; i < a.rows ; ++i){
    for(size_t j =0 ; j< a.cols ; ++j){
      MAT_AT(a, i, j) =  sigmoid(MAT_AT(a, i, j));
    }
  }
}

Mat mat_row(Mat m, size_t row){
  return (Mat){
   .rows = 1,
   .cols = m.cols,
   .stride = m.stride,
   .values = &MAT_AT(m, row, 0)
  }; 
}
void mat_copy(Mat dst, Mat src){
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);
  for(size_t i = 0; i < dst.rows ; ++i){
    for(size_t j = 0; j < dst.cols ; ++j){
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}
