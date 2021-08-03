#pragma once
#include <memory>
#include <iostream>


#include <cuda_runtime.h>
#include <iostream>
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
#include <cublas_v2.h>
// #include <cublasLt.h>
#include <stdexcept>
#include <map>
#include "stdio.h"
#include <fstream>
/*#include <helper_cuda.h> 
#include <helper_functions.h> 
*/

// #define DEBUG_MOD
// #define DECODER_DEBUG

/*** old version ***/
// #define KERNEL_FUSION
// #define BIAS_LAYERNORM_FUSION

// #define DataType  float

// #define ENCODER_SELF_ATTENTION_FUSION

#define BASIC_KERNEL_FUSION
#define CROSS_ATTENTION_FUSION

#define SELF_ATTENTION_FUSION  // MAX_DECODER_STEPS=256
#define MAX_DECODER_STEPS 256

#define MAX_SOURCE_TOKENS 256

#define DECODER_PADDING_OPTIMIZE

// #define MATRIX_FUSION

// #define TIME_CALCULATION

#define FAST_GELU
#define FAST_LAYERNORM

namespace TenTrans{

static const char *_cudaGetErrorEnum(cudaError error)// cublasStatus_t error)
{
  switch (error)
  {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    std::cout << "result: " << result << std::endl;
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + 
            (_cudaGetErrorEnum(result)) + " " + file + ":" + std::to_string(line) + " \n");
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

/*
template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") +
                             (_cudaGetErrorEnum(result)) + " " + file +
                             ":" + std::to_string(line) + " \n");
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
*/

const size_t PAD_ID = 0;
const size_t UNK_ID = 1;
const size_t BOS_ID = 2;
const size_t EOS_ID = 2;

const std::string PAD_STR = "<pad>";
const std::string UNK_STR = "<unk>";
const std::string BOS_STR = "<s>";
const std::string EOS_STR = "<s>";

const int MAX_LENGTH_FACTOR = 3;

enum class DeviceType : size_t { gpu = 0, cpu = 1 };
enum class ActivationType { RELU, GELU, SWISH} ;

struct DeviceId {
	size_t no{0};
    DeviceType type{DeviceType::gpu};

    DeviceId() : no{0}, type{DeviceType::gpu} {}
    DeviceId(size_t no_, DeviceType type_) : no(no_), type(type_) {}

	friend std::ostream& operator<<(std::ostream& out, DeviceId deviceId) {
    	out << (deviceId.type == DeviceType::gpu ? "gpu" : "cpu") << deviceId.no;
    	return out;
  	}
};

template<class T>
using HUPtr = std::shared_ptr<T>;

template<class T, typename... Args>
HUPtr<T> HUNew(Args&&... args)
{
	return HUPtr<T>(new T(std::forward<Args>(args)...));
}

const size_t CHUNK = 128;
const size_t MBYTE = 1024 * 1024;
const size_t GROW = CHUNK * MBYTE;
const size_t ALIGN = 256;

#define Mb (1024*1024)
#define Kb (1024)
#define Gb (1024*1024*1024)
#define GETSTR(val) ({std::string(#val);})

#define MAX_TENSOR_DIM_NUM 4

#include <cuda.h>
#define __H__ __host__
#define __D__ __device__
#define __HI__ __host__ inline
#define __HD__ __host__ __device__
#define __HDI__ __host__ __device__ inline


}
