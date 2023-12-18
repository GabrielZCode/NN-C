#include "gates.h"
#include <stdio.h>
#ifndef MATRIX_OP
#define MATRIX_OP

#ifndef NN_ALLOC
#define NN_ALLOC malloc
#endif /* !NN_ALLOC */


typedef struct{
  size_t rows;
  size_t cols;
  size_t stride;
  float* values;
}Mat;

#define MAT_AT(m, i ,j) (m).values[(i)*(m).stride + (j)] 

#define print_matrix_and_name(m) print_matrix(m, #m , 2)
void print_matrix(Mat m, const char *name, size_t padding);

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float x);
void mat_dot(Mat result_dst, Mat a, Mat b);
void mat_sum(Mat result_dst, Mat a, Mat b);
void mat_sum_over(Mat result_dst, Mat a);
void mat_apply_sigmoid(Mat a);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
#endif // !MATRIX_OP

