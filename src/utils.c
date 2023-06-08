#include <math.h>
#include <stdlib.h>
#include <time.h>


float rand_float(){
  return ((float)rand()  / (float)RAND_MAX);
}


float sigmoid(float x){
  return 1.f / (1.f + expf(-x));
}
