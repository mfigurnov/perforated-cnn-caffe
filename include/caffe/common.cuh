#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_

#include <cuda.h>

// File taken from https://github.com/cdmh/deeplab-public/blob/master/include/caffe/common.cuh

// CUDA: atomicAdd is not defined for doubles
static __inline__ __device__ double atomicAdd(double* address, double val) {
  // NOLINT_NEXT_LINE(runtime/int)
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  // NOLINT_NEXT_LINE(runtime/int)
  unsigned long long int old = *address_as_ull, assumed;
  if (val == 0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
      __double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif
