#pragma once
#include "cuda_runtime.h"
#include "Logging.h"
using namespace TenTrans;

#define CUDA_FLT_MAX 1.70141e+38
const int MAX_THREADS = 512;
const int MAX_BLOCKS = 65535;

#define CUDA_CHECK(expr) do {                                              \
	cudaError_t rc = (expr);                                                 \
	ABORT_IF(rc != cudaSuccess, "[TenTrans] CUDA error {} '{}' - {}:{}: {}", rc, cudaGetErrorString(rc),  __FILE__, __LINE__, #expr);                                                      \
} while(0)

template <typename T>
void CudaCopy(const T* start, const T* end, T* dest) {
  CUDA_CHECK(cudaMemcpy(dest, start, (end - start) * sizeof(T), cudaMemcpyDefault));
}
