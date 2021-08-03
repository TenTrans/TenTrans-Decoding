#include "HUTensor.h"
#include "HUCudaHelper.h"

namespace TenTrans{

namespace gpu{

template <typename T>
void copy(HUPtr<HUDevice> device, const T* begin, const T* end, T* dest) {
  CUDA_CHECK(cudaSetDevice(device->getDeviceId().no));
  CudaCopy(begin, end, dest);
  CUDA_CHECK(cudaStreamSynchronize(0));
}

template void copy<float>(HUPtr<HUDevice> device, const float* begin, const float* end, float* dest);

template void copy<int>(HUPtr<HUDevice> device, const int* begin, const int* end, int* des);

__global__ void gFill(float* d_in, int size, float val) {
  for(int bid = 0; bid < size; bid += blockDim.x * gridDim.x) {
    int index = bid + threadIdx.x + blockDim.x * blockIdx.x;
    if(index < size) {
      d_in[index] = val;
    }   
  }
}

void fill(HUPtr<HUDevice> device, float* begin, float* end, float value) {
  CUDA_CHECK(cudaSetDevice(device->getDeviceId().no));
  int size = end - begin;
  int threads = std::min(512, size);
  int blocks = (size / threads) + (size % threads != 0); 
  gFill<<<blocks, threads>>>(begin, size, value);
  CUDA_CHECK(cudaStreamSynchronize(0));
}

void alloc(float* data, size_t bufSize)
{
	cudaMalloc((void**)&data, sizeof(float)*bufSize);
}

void free(float* data)
{
	cudaFree(data);
}

}

}
