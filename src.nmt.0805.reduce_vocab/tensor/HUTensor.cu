#include "HUTensor.h"
#include "HUCudaHelper.h"

namespace TenTrans {

namespace gpu {

template <typename T>
void copy(HUPtr<HUDevice> device, const T* begin, const T* end, T* dest) 
{
  CUDA_CHECK(cudaSetDevice(device->getDeviceId().no));
  CudaCopy(begin, end, dest);
  CUDA_CHECK(cudaStreamSynchronize(0));
}
template void copy<float>(HUPtr<HUDevice> device, const float* begin, const float* end, float* dest);
template void copy<half>(HUPtr<HUDevice> device, const half* begin, const half* end, half* des);

template <typename T>
__global__ void gFill(T* d_in, int size, T val) 
{
  for(int bid = 0; bid < size; bid += blockDim.x * gridDim.x) 
  {
    int index = bid + threadIdx.x + blockDim.x * blockIdx.x;
    if(index < size) {
      d_in[index] = val;
    }   
  }
}
template <typename T>
void fill(HUPtr<HUDevice> device, T* begin, T* end, T value) 
{
  CUDA_CHECK(cudaSetDevice(device->getDeviceId().no));
  int size = end - begin;
  int threads = std::min(512, size);
  int blocks = (size / threads) + (size % threads != 0); 
  gFill<<<blocks, threads>>>(begin, size, value);
  CUDA_CHECK(cudaStreamSynchronize(0));
}
template void fill<float>(HUPtr<HUDevice> device, float* begin, float* end, float value);
template void fill<half>(HUPtr<HUDevice> device, half* begin, half* end, half value);


template <typename T>
void alloc(T* data, size_t bufSize)
{
  cudaMalloc((void**)&data, sizeof(T)*bufSize);
}
template void alloc<float>(float* data, size_t bufSize);
template void alloc<half>(half* data, size_t bufSize);


template <typename T>
void free(T* data)
{
  cudaFree(data);
}
template void free<float>(float* data);
template void free<half>(half* data);

}

}
