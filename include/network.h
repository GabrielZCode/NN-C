#include <matrix.h>
#include <stddef.h>

#ifndef NN_DEF
#define NN_DEF

typedef struct {
  size_t layer_count;
  Mat *ws;
  Mat *bs; 
  Mat *as;
} NN;


#define NN_ALLOC_MAT(arch) nn_alloc(arch, ARRAY_LEN(arch)) 
NN nn_alloc(size_t *architechture, size_t arch_count);
#define NN_PRINT(NN) nn_print(NN, #NN) 
void nn_print(NN nn, const char *name);

void nn_rand(NN nn, float min, float max);
void nn_forward(NN nn);
void nn_finite_diff(NN m, NN g, float eps, Mat input, Mat output);
float nn_cost(NN nn, Mat input, Mat output);
void learn(NN nn, NN g, float rate);


#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).layer_count]

#endif // !NN_DEF


