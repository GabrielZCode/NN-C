#include "utils.h"
#include "double.h"
#include "gates.h"
#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int main(){
  srand(time(0));
  train_xor();
  return 0;
}
