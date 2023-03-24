#include "HUTensorOP.h"
#include "HUCudaHelper.h"
#include "cub.cuh"
#include <cublas_v2.h>
#include <cfloat>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include <vector>
#include <type_traits>

namespace fastertransformer {

#define MAX_BLOCKS_PER_BEAM 8

  template <typename T>
  void topK_kernelLauncher(const T* log_probs,
                           int* topk_tmp_id_buf,
                           T* topk_tmp_val_buf,
                           int* topk_id_buf,
                           T* topk_val_buf,
                           const int batch_size,
                           const int beams_per_batch,
                           const int k,
                           const int vocab_size);
}  // fastertransformer 

/*
namespace fastertransformer_V2 {

  template <typename T>
  void topK_softMax(const T* log_probs, 
                    const T* bias, 
                    const uint8_t* finished, 
                    float* cum_log_probs,
                    int* ids,
                    const int K,
                    void* temp_storage,
                    const int temp_storage_size,
                    const int batch_size,
                    const int beam_width,
                    const int vocab_size,
                    const int end_id,
                    const T diversity_rate);

  template void topK_softMax<float>(const float* log_probs,
                                  const float* bias,
                                  const uint8_t* finished,
                                  float* cum_log_probs,
                                  int* ids,
                                  const int K,
                                  void* tmp_storage,
                                  const int temp_storage_size,
                                  const int batch_size,
                                  const int beam_width,
                                  const int vocab_size,
                                  const int end_id,
                                  const float diversity_rate); 
}   // astertransformer_V2    */

namespace TenTrans{

#define FINAL_MASK 0xffffffff
// static const int SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE = 256;

enum { FLOAT_DATATYPE=0, HALF_DATATYPE=1, INT8_DATATYPE=2 };

const int WARP_SIZE = 32;
const int ATTENTION_BLOCK_SIZE = 256;
const bool ATTENION_OPT = true;
// static const float HALF_FLT_MAX = 65504.F;


#define DO_SPLIT_SMALL_TOP_K_SOFTMAX
static const int SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE = 256;
static const int SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS = 128;
static const int MAX_K = 4;

static const float HALF_FLT_MAX = 65504.F;

template<typename T, int MAX_K>
struct TopK
{
    int p[MAX_K];
    T u[MAX_K];

    __device__ __forceinline__ void insert(T elem, int elem_id)
    {
        if (elem > u[MAX_K-1] || (p[MAX_K-1] == -1) || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        //if (elem > u[MAX_K-1] || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        {
            u[MAX_K-1] = elem;
            p[MAX_K-1] = elem_id;
        }

        for(int k = MAX_K - 2; k >= 0; --k)
        {
            if ((u[k+1] > u[k]) || (p[k] == -1) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            //if ((u[k+1] > u[k]) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            {
                T u2 = u[k];
                int p2 = p[k];
                u[k] = u[k+1];
                p[k] = p[k+1];
                u[k+1] = u2;
                p[k+1] = p2;
            }
        }
    }

    __device__ __forceinline__ void init()
    {
        const bool IS_FP16 = std::is_same<T, half>::value;
        const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;

        for(int i = 0; i < MAX_K; i++)
        {
            p[i] = -1;
            u[i] = -MAX_T_VAL;
        }
    }
};

template<typename T, int MAX_K>
__device__ __forceinline__ TopK<T, MAX_K> reduce_topk_op(const TopK<T, MAX_K>& a, const TopK<T, MAX_K>& b)
{
    TopK<T, MAX_K> res = a;
    for(int i = 0; i < MAX_K; ++i)
        res.insert(b.u[i], b.p[i]);
    return res;
}

template <int HALF_ELEMENTS_PER_WARP_LOAD>
using Copy_half_t =
    typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 32, half,
        typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 64, int,
            typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 128, int2, int4
            >::type
        >::type
    >::type;

template <typename T, int ELEMENTS_PER_WARP_LOAD>
using Copy_t = Copy_half_t<sizeof(T) / sizeof(half) * ELEMENTS_PER_WARP_LOAD>;

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

template <typename T>
__inline__ __device__ 
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

template <typename T>
__inline__ __device__ 
T warpReduceMax(T val)
{
  #pragma unroll
  for(int mask = 16; mask > 0; mask >>= 1) {
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  }
  return val;
}

template <typename T>
__inline__ __device__ 
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ 
T blockReduceMax(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) { // record in-warp maxx by warp Idx
    shared[wid] = val;
  }

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)-1e20f;
  val = warpReduceMax(val);
  return val;
}

template <typename T, int size_per_head, int block_sz>
__global__
void cross_attention_kernel_opt(
  T* __restrict query_buf, const T* __restrict Q_bias,
  T* __restrict key_cache, const T* __restrict K_bias,
  T* __restrict value_cache, const T* __restrict V_bias,
  T* length_per_sample, T* __restrict context_buf, 
  const uint8_t* finished, const int beam_size,  
  const int batch_size, const int head_num, const int step, const int seq_len, const float scalar)
{
#ifdef DECODER_PADDING_OPTIMIZE
  if(finished != nullptr && finished[blockIdx.x / beam_size / head_num]) return;
#endif

  typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;
  union Access_t
  {
    copy_t v;
    T x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Float_n_t
  {
    float x[elems_per_thread]; // supported size 1,2,4
  } float_n_t;

  __shared__ float_n_t sq[block_sz];
  extern __shared__ float logits[]; // use to store the logits from [0~step]

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int warp_num = block_sz / WARP_SIZE;

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;

  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x / head_num;
  const int head_id = blockIdx.x % head_num;

  int length = __ldg(&length_per_sample[bid]);

  const int lane_id = tid % WARP_SIZE;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head;
  int qkv_bias_id = head_id * size_per_head;

  int key_value_id = bid * (seq_len * head_num * size_per_head) + head_id * size_per_head;

  query_buf = &query_buf[qkv_id];
  K_bias = &K_bias[qkv_bias_id];
  key_cache = &key_cache[key_value_id];
  Q_bias = &Q_bias[qkv_bias_id];
  V_bias = &V_bias[qkv_bias_id];
  value_cache = &value_cache[key_value_id];
  context_buf = &context_buf[qkv_id];

  Access_t bias_r, key_val_r, query_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  bias_r.v = *((copy_t *)Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] = (float)query_buf_r.x[i] + (float)bias_r.x[i];
  }
  __syncthreads();

  //offset for each step
  int offset =  head_num * size_per_head;

  bias_r.v = *((copy_t *) K_bias + lane_id);
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if (step == 0)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++)
    {
      val = val + (float)key_val_r.x[i] * qb_r[i] * scalar;
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0)
    {
      logits[ite] = qk;
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;
  float local_i = -1e20f;
  for(int i = tid; i < length; i += blockDim.x)
    local_i = max(local_i, logits[i]);

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  float local_o = 0.0f;
  for(int i = tid; i < length; i += blockDim.x)
  {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < length; i += blockDim.x)
  {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads();

  // This optimization introduces discrepancy because of different order in FP32 summation
  float sum_r[elems_per_thread] = {0.f};
  bias_r.v = *((copy_t *) V_bias + lane_id);
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    if(step == 0)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&value_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
      sum_r[i] += (float)key_val_r.x[i] * logits[ite];
    }
  }
  for (int i = 0; i < elems_per_thread; i++)
  {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (threadIdx.x < WARP_SIZE)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i)
      {
        sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + threadIdx.x].x[i];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++)
  {
    key_val_r.x[i] = sum_r[i];
  }
  if (threadIdx.x < WARP_SIZE)
  {
    *((copy_t *)context_buf + lane_id) = key_val_r.v;
  }
}

template <int size_per_head, int block_sz>
__global__
void fusedQKV_masked_attention_kernel_opt(
  const float* __restrict qkv_buf, const float* __restrict qkv_bias,
  float* __restrict key_cache, float* __restrict value_cache,
  float* __restrict context_buf, int batch_size, const int head_num, const int step, const float scalar)
{
  typedef Copy_t<float, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;

  union Access_t
  {
    copy_t v;
    float x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Float_n_t
  {
    float x[elems_per_thread]; // supported size 1,2,4
  } float_n_t;

  __shared__ float_n_t sq[block_sz];

  extern __shared__ float logits[]; // use to store the logits from [0~step]

  const int tid = threadIdx.x;
  const int warp_num = block_sz / WARP_SIZE;
  const int bid = blockIdx.x;
  const int head_id = blockIdx.x % head_num;
  const int warp_id = tid / WARP_SIZE; // warp_id in block
  const int lane_id = tid % WARP_SIZE; // lane_id in warp
  const int batch_id = bid / head_num;
  const int hidden_units = head_num * size_per_head;

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;
  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  int qkv_id = batch_id * 3 * hidden_units + head_id * size_per_head;
  int qkv_bias_id = head_id * size_per_head;
  int cache_qkv_id = bid * size_per_head;

  const float* query_buf = qkv_buf + qkv_id;
  const float* key_buf = qkv_buf + hidden_units + qkv_id;
  const float* value_buf = qkv_buf + 2 * hidden_units + qkv_id;
  const float* self_Q_bias = qkv_bias + qkv_bias_id;
  const float* self_K_bias = qkv_bias + hidden_units + qkv_bias_id;
  const float* self_V_bias = qkv_bias + 2 * hidden_units + qkv_bias_id;
  value_cache = value_cache + cache_qkv_id;
  key_cache = key_cache + cache_qkv_id;
  context_buf = context_buf + cache_qkv_id;
  
  Access_t bias_r, query_buf_r;
  Access_t key_val_r, key_buf_r;
  Access_t value_val_r, value_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  key_buf_r.v = *((copy_t *)key_buf + lane_id);
  bias_r.v = *((copy_t *)self_Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i) {
    qb_r[i] =  (float)query_buf_r.x[i] + (float)bias_r.x[i];
  }

  //offset for each step
  int offset = batch_size * hidden_units;
  bias_r.v = *((copy_t *) self_K_bias + lane_id);
  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++) {
        key_val_r.x[i] = (float)key_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++) {
      val = val +  (float)key_val_r.x[i] * qb_r[i] * (float)scalar;
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0) {
      logits[ite] = qk;
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = -1e20f;
  for(int i = tid; i < step; i += blockDim.x) {
    local_i = max(local_i, logits[i]);
  }

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0) {
    s_max_val = max_val;
  }
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < step; i += blockDim.x) {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads();

  // This optimization introduces discrepancy because of different order in FP32 summation
  float sum_r[elems_per_thread] = {0.f};
  bias_r.v = *((copy_t *) self_V_bias + lane_id);
  value_buf_r.v = *((copy_t *)value_buf + lane_id);

  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    value_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        value_val_r.x[i] = (float)value_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&value_cache[ite * offset] + lane_id) = value_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i) {
      sum_r[i] += (float)value_val_r.x[i] * logits[ite];
    }
  }
  for (int i = 0; i < elems_per_thread; i++) {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (warp_id == 0)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i) {
        sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + tid].x[i];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++) {
    value_val_r.x[i] = sum_r[i];
  }
  if (warp_id == 0) {
    *((copy_t *)context_buf + lane_id) = value_val_r.v;
  }
}

template<typename T>
__global__ 
void cross_attention_kernel(
        T* query_buf, const T* Q_bias, 
        T* key_cache, const T* K_bias, 
        T* value_cache, const T* V_bias, 
        T* length_per_sample, T* context_buf, 
        const uint8_t* finished, const int beam_size, 
        const int batch_size, const int head_num, const int size_per_head, 
        const int step, const int seq_len, const T scalar)
{
#ifdef DECODER_PADDING_OPTIMIZE
  if(finished != nullptr && finished[blockIdx.x / beam_size / head_num]) return;
#endif

  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int length = __ldg(&length_per_sample[bid]);

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head) {
    sq[tid] = query_buf[qkv_id] + Q_bias[qkv_bias_id];
  }
  __syncthreads();

  for(int ite = 0; ite < length; ++ite)
  {
    int key_id = bid * (seq_len * head_num * size_per_head) + ite * (head_num * size_per_head)
      + head_id * size_per_head + tid;

    T key = tid < size_per_head ? key_cache[key_id] : (T)(0.0f);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if(step == 0 && tid < size_per_head)
    {
      key += K_bias[head_id * size_per_head + tid];
      key_cache[key_id] = key;
    }

    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0) {
      logits[ite] = qk;
    }
    __syncthreads(); //try to remove
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = tid < length ? (float)logits[tid] : -1e20f;
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0) {
    s_max_val = max_val;
  }
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < length ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0) {
    s_sum = val + 1e-6;
  }
  __syncthreads();
  if(tid < length) {
    logits[tid] = local_o / s_sum;
  }
  __syncthreads();

  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < length; ++ite)
    {
      int value_id = bid * seq_len * head_num * size_per_head + ite * head_num * size_per_head
        + head_id * size_per_head + tid;

      T value = value_cache[value_id];

      //for the first step, we should add bias to key memory cache
      if(step == 0)
      {
        value += V_bias[head_id * size_per_head + tid];
        value_cache[value_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[bid * head_num * size_per_head + head_id * size_per_head + tid] = sum;
  }
}

/*
template <int size_per_head, int block_sz>
__global__
void cross_attention_kernel_opt(
  float* __restrict query_buf, const float* __restrict Q_bias,
  float* __restrict key_cache, const float* __restrict K_bias,
  float* __restrict value_cache, const float* __restrict V_bias,
  const int* length_per_sample, float* __restrict context_buf,
  int batch_size, const int head_num, const int step, const int seq_len, const float scalar)
{
  typedef Copy_t<float, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;
  union Access_t
  {
    copy_t v;
    float x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Float_n_t
  {
    float x[elems_per_thread]; // supported size 1,2,4
  } float_n_t;

  __shared__ float_n_t sq[block_sz];
  extern __shared__ float logits[]; // use to store the logits from [0~step]

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int warp_num = block_sz / WARP_SIZE;

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;
  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x / head_num;
  const int head_id = blockIdx.x % head_num;

  int length = __ldg(&length_per_sample[bid]);
  const int lane_id = tid % WARP_SIZE;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head;
  int qkv_bias_id = head_id * size_per_head;

  int key_value_id = bid * (seq_len * head_num * size_per_head) + head_id * size_per_head;

  query_buf = &query_buf[qkv_id];
  K_bias = &K_bias[qkv_bias_id];
  key_cache = &key_cache[key_value_id];
  Q_bias = &Q_bias[qkv_bias_id];
  V_bias = &V_bias[qkv_bias_id];
  value_cache = &value_cache[key_value_id];
  context_buf = &context_buf[qkv_id];

  Access_t bias_r, key_val_r, query_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  bias_r.v = *((copy_t *)Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i) {
    qb_r[i] = (float)query_buf_r.x[i] + (float)bias_r.x[i];
  }

  //offset for each step
  int offset =  head_num * size_per_head;
  bias_r.v = *((copy_t *) K_bias + lane_id);
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if (step == 0)
    {
      for (int i = 0; i < elems_per_thread; i++) {
        key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++) {
      val = val + (float)key_val_r.x[i] * qb_r[i] * scalar;
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0) {
      logits[ite] = qk;
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;
  float local_i = -1e20f;
  for(int i = tid; i < length; i += blockDim.x) {
    local_i = max(local_i, logits[i]);
  }

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0) {
    s_max_val = max_val;
  }
  __syncthreads();

  float local_o = 0.0f;
  for(int i = tid; i < length; i += blockDim.x) {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0) {
    s_sum = val + 1e-6;
  }
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < length; i += blockDim.x) {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads();

  // This optimization introduces discrepancy because of different order in FP32 summation
  float sum_r[elems_per_thread] = {0.f};
  bias_r.v = *((copy_t *) V_bias + lane_id);
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    if(step == 0)
    {
      for (int i = 0; i < elems_per_thread; i++) {
        key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&value_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i) {
      sum_r[i] += (float)key_val_r.x[i] * logits[ite];
    }
  }
  for (int i = 0; i < elems_per_thread; i++) {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (threadIdx.x < WARP_SIZE)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i) {
        sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + threadIdx.x].x[i];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++) {
    key_val_r.x[i] = sum_r[i];
  }
  if (threadIdx.x  < WARP_SIZE) {
    *((copy_t *)context_buf + lane_id) = key_val_r.v;
  }
} */

template <typename T>
__global__ 
void gCopyRows(T* out,                      // 120 * 512
               const T* in,                 // 37000 * 512 
               size_t cols,                 // 512
               const size_t* sourceRowIdx,  // 120
               size_t rows)                 // 120
{
  for(int bid = 0; bid < rows; bid += gridDim.x) 
  {
    int j = bid + blockIdx.x;
    if(j < rows) {
      size_t dstId = j;
      size_t srcId = sourceRowIdx[j];

      T* rowOut = out + dstId * cols;
      const T* rowIn = in + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }
}

template <typename T>
__global__
void gCopyCols(T* out,                       // 512 * 120
               const T* in,                  // 512 * 37000
               size_t rows,                  // 512
               size_t cols,                  // 37000
               const size_t* sourceColIdx,   // 120
               size_t colsToCopy)            // 120
{
  for(int bid = 0; bid < colsToCopy; bid += gridDim.x)
  {
    int j = bid + blockIdx.x;
    if(j < colsToCopy) {
      size_t dstId = j;                 // 0, 1, .. -> 120
      size_t srcId = sourceColIdx[j];   // random: [0, 37000)

      T* rowOut = out + dstId;
      const T* rowIn = in + srcId;

      // rows: 0 -> 512
      for(int tid = 0; tid < rows; tid += blockDim.x) {
        // colsToCopy: 120,  cols: 37000
        int i = tid + threadIdx.x;
        // out[tid * colsToCopy + dstId] = in[tid * cols + srcId];
        if (i < rows)
            rowOut[i*colsToCopy] = rowIn[i*cols];
      }
    }
  }
}


/*
template <>
__global__
void gCopyRows(half* out, 
              const half* __restrict in, 
              size_t cols, 
              const size_t* sourceRowIdx, 
              size_t rows)
{
  half2* out_ptr = (half2*) out;
  const half2* in_ptr = (half2*) in;

  for(int bid = 0; bid < rows; bid += gridDim.x)
  {
    int j = bid + blockIdx.x;
    if(j < rows) 
    { 
      size_t dstId = j;
      size_t srcId = sourceRowIdx[j];

      half2* rowOut = out_ptr + dstId * cols;
      const half2* rowIn = in_ptr + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) 
      {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }
}
*/

void CopyRowsOP(HUPtr<HUTensor> out, const HUPtr<HUTensor> in, const std::vector<size_t>& indices)
{
	cudaSetDevice(out->getDeviceId().no);

  	size_t cols = in->shape().back();
  	size_t rowsToCopy = indices.size();

  	int threads = std::min(MAX_THREADS, (int)cols);
  	int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

  	size_t* d_indices;
  	CUDA_CHECK(cudaMalloc(&d_indices, rowsToCopy * sizeof(size_t)));
  	CUDA_CHECK(cudaMemcpy(d_indices,
                        indices.data(),
                        rowsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  	gCopyRows<TT_DATA_TYPE><<<blocks, threads>>>(
      	out->data(), in->data(), cols, d_indices, rowsToCopy);

  	CUDA_CHECK(cudaFree(d_indices));
}

void CopyColsOP(HUPtr<HUTensor> out, const HUPtr<HUTensor> in, const std::vector<size_t>& indices)
{
    cudaSetDevice(out->getDeviceId().no);

    size_t rows = in->shape()[0];
    size_t cols = in->shape().back();
    size_t colsToCopy = indices.size();

    int threads = std::min(MAX_THREADS, (int)rows);
    int blocks = std::min(MAX_BLOCKS, (int)colsToCopy);

    size_t* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, colsToCopy * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_indices,
                        indices.data(),
                        colsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));

    gCopyCols<TT_DATA_TYPE><<<blocks, threads>>>(
        out->data(), in->data(), rows, cols, d_indices, colsToCopy);

    CUDA_CHECK(cudaFree(d_indices));
}


void CopyRowsOP_V2(HUPtr<HUTensor> out, const HUPtr<HUTensor> in, size_t* indices, int rowsToCopy)
{
    cudaSetDevice(out->getDeviceId().no);

    size_t cols = in->shape().back();
    // size_t rowsToCopy = indices.size();

    int threads = std::min(MAX_THREADS, (int)cols);
    int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

    /*
    size_t* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, rowsToCopy * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_indices,
                        indices.data(),
                        rowsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));
    */

    gCopyRows<TT_DATA_TYPE><<<blocks, threads>>>(
        out->data(), in->data(), cols, indices, rowsToCopy);

    // CUDA_CHECK(cudaFree(d_indices));
}

HUPtr<HUTensor> ReshapeOP(HUPtr<HUTensor> in, HUShape shape)
{
	HUPtr<HUTensor> out = HUNew<HUTensor>(in->memory(), shape, in->getDevice());
	return out;
}

/*
extern "C" __global__ 
void KernelScaleAndShift(float * d, int size, float scale, float shift)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    bool isUnitScale = (scale == 1.0F);
    bool isZeroShift = (shift == 0.0F);
    if (i < size) 
    {
        if (isUnitScale && !isZeroShift) {
            d[i] = d[i] + shift;
        }
        else if (isUnitScale && isZeroShift) {
            d[i] = d[i];
        }
        else if (!isUnitScale && isZeroShift) {
            d[i] = d[i] * scale;
        }
        else {
            d[i] = d[i] * scale + shift;
        }
    }
} */
template <typename T>
__global__
void KernelScaleAndShift(T* d, int size, float scale, float shift)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    bool isUnitScale = (scale == 1.0F);
    bool isZeroShift = (shift == 0.0F);
    if (i < size) 
    {
        if (isUnitScale && !isZeroShift) {
            d[i] = d[i] + (T)shift;
        }
        else if (isUnitScale && isZeroShift) {
            d[i] = d[i];
        }
        else if (!isUnitScale && isZeroShift) {
            d[i] = d[i] * (T)scale;
        }
        else {
            d[i] = (T)((float)d[i] * scale + shift);
        }
    }
} 

void ScaleAndShiftOP(HUPtr<HUTensor> &a, float scale, float shift)
{
	cudaSetDevice(a->getDeviceId().no);
	int length = a->shape().elements();
	int threads = std::min(MAX_THREADS, length);
	int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

	KernelScaleAndShift<TT_DATA_TYPE><<<blocks, threads>>>(a->data(), length, scale, shift);
}

/*
extern "C" __global__ 
void KernelADD(float * a, float * b, float * c, int size, float beta)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    bool isUnitScale = (beta == 1.0F);
    if (i < size)
    {
        if (isUnitScale) {
            c[i] = a[i] + b[i];
        }
        else {
            c[i] = a[i] + b[i] * beta;
        }
    }
} */

template <typename T>
__global__
void KernelADD(T * a, 
               T * b, 
               T * c, 
               int size, 
               float beta)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    bool isUnitScale = (beta == 1.0F);
    if (i < size)
    {
        if (isUnitScale) {
            c[i] = a[i] + b[i];
        }
        else {
            c[i] = a[i] + b[i] * (T)beta;
        }
    }
} 

void Plus(HUPtr<HUTensor> &a, HUPtr<HUTensor> b, HUPtr<HUTensor> c, float scale)
{
	cudaSetDevice(a->getDeviceId().no);
	//bool broadcast = false;
	if(c == NULL)
		c = a;
	int length = a->shape().elements();
	int threads = std::min(MAX_THREADS, length);
	int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

	//if(a->shape() != b->shape())
	//	broadcast = true;

	KernelADD<TT_DATA_TYPE><<<blocks, threads>>>(a->data(), b->data(), c->data(), length, scale);
}

/*extern "C" __global__ void gElementPlus(float* a, float* b, float* c, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        c[i] = a[i] + b[i];
}*/

/*
extern "C" __global__ void gElementPlus(Functional::HUTensor<float> ga, Functional::HUTensor<float> gb, Functional::HUTensor<float> gc, bool broadcast)
{
	int length = ga.shape().elements();
	int indices[3];
	int dims[4];
	for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    	int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    	if(index < length) {
      		//indices.fill(index);
			for(int i=0; i<3; i++)
				indices[i] = index;

      		if(broadcast) {
        		ga.shape().dims(index, dims);
        		//for(int i = 1; i < 3; ++i)
          		//indices[i] = tensors[i].shape().bindex(dims);
				indices[1] = gb.shape().bindex(dims);
				indices[2] = gc.shape().bindex(dims);
      		}

      		ga[indices[0]] = gb[indices[1]] + gc[indices[2]];
    	}
  	}
} */

template <typename T>
__global__ 
void gElementPlus(Functional::HUTensor<T> ga, 
                  Functional::HUTensor<T> gb, 
                  Functional::HUTensor<T> gc, 
                  bool broadcast)
{
    int length = ga.shape().elements();
    int indices[3];
    int dims[4];
    for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
        int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
        if(index < length) {
            //indices.fill(index);
            for(int i=0; i<3; i++)
                indices[i] = index;

            if(broadcast) {
                ga.shape().dims(index, dims);
                //for(int i = 1; i < 3; ++i)
                //indices[i] = tensors[i].shape().bindex(dims);
                indices[1] = gb.shape().bindex(dims);
                indices[2] = gc.shape().bindex(dims);
            }

            ga[indices[0]] = gb[indices[1]] + gc[indices[2]];
        }
    }
} 

void PlusBroadcast(HUPtr<HUTensor> &a, HUPtr<HUTensor> b, HUPtr<HUTensor> c)
{
	Functional::HUTensor<TT_DATA_TYPE> ga(a);
	Functional::HUTensor<TT_DATA_TYPE> gb(b);
	Functional::HUTensor<TT_DATA_TYPE> gc(c);
	
	cudaSetDevice(a->getDeviceId().no);
	int length = a->size();
	int threads = std::min(MAX_THREADS, length);
	int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

	bool broadcast = false;
    if(a->shape() != b->shape() || a->shape() != c->shape())
		broadcast = true;

	gElementPlus<TT_DATA_TYPE><<<blocks, threads>>>(ga, gb, gc, broadcast);
#ifndef CUDA_DEBUG
	cudaStreamSynchronize(0);
#endif
}

/*
extern "C" __global__ void gElementGelu(Functional::HUTensor<float> out, Functional::HUTensor<float> in)
{
    int length = out.shape().elements();
    for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
        int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
        if(index < length) {
#ifdef FAST_GELU
            out[index] = 0.5f * in[index] * (1.0f + tanhf((0.7978845608028654f * (in[index] + 0.044715f * in[index] * in[index] * in[index]))));
#else
            out[index] = 0.5f * in[index] * (1.0f + erf(in[index] / sqrtf(2.0f)));
#endif
        }
    }
} */

template <typename T>
__global__ 
void gElementGelu(Functional::HUTensor<T> out, 
                  Functional::HUTensor<T> in)
{
    int length = out.shape().elements();
    for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
        int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
        if(index < length) {
            float tmp = in[index];
#ifdef FAST_GELU
            out[index] = (T)(0.5f * tmp * (1.0f + tanhf((0.7978845608028654f * (tmp + \
                                    0.044715f * tmp * tmp * tmp)))));
#else
            out[index] = (T)(0.5f * tmp * (1.0f + erf(tmp / sqrtf(2.0f))));
#endif
        }
    }
} 

/*
__inline__ __device__
T gelu(T x)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}
*/

/*
T gelu(T x)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
} */

template <typename T>
__inline__ __device__
T gelu(T x)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__inline__ __device__
half2 gelu(half2 val)
{
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp =  __half22float2(val);

  tmp.x = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

/*
extern "C" __global__ void gElementRelu(Functional::HUTensor<float> out, Functional::HUTensor<float> in)
{
    int length = out.shape().elements();
    for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
        int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
        if(index < length) {
            out[index] = in[index] > 0.f ? in[index] : 0.f;
        }
    }
} */

template <typename T>
__global__ 
void gElementRelu(Functional::HUTensor<T> out, 
                  Functional::HUTensor<T> in)
{
    int length = out.shape().elements();
    for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
        int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
        if(index < length) {
            out[index] = in[index] > (T)0.f ? in[index] : (T)0.f;
        }
    }
}

/*
extern "C" __global__ void gElementSwish(Functional::HUTensor<float> out, Functional::HUTensor<float> in)
{    int length = out.shape().elements();
    for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
        int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
        if(index < length) {
            //out[index] = in[index] > 0.f ? in[index] : 0.f;
			float sigmoid = in[index] > 0 ? (1.f / (1.f + expf(-in[index]))) : (expf(in[index]) / (1.f + expf(in[index])));
			out[index] = in[index] * sigmoid;
        }    
	}
} */

template <typename T>
__global__ void gElementSwish(Functional::HUTensor<T> out, 
                              Functional::HUTensor<T> in)
{    
    int length = out.shape().elements();
    for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
        int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
        if(index < length) {
            //out[index] = in[index] > 0.f ? in[index] : 0.f;

            float tmp = (float)in[index];
            float sigmoid = tmp > 0.f ? (1.f / (1.f + expf(-tmp))) : (expf(tmp) / (1.f + expf(tmp)));
            out[index] = (T)(tmp * sigmoid);
        }    
    }
} 

// gelu: a = 0.5b * (1.0 + erf(b / sqrt(2.0)))
void GeluOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in)
{
    Functional::HUTensor<TT_DATA_TYPE> gin(in);
    Functional::HUTensor<TT_DATA_TYPE> gout(out);

    cudaSetDevice(in->getDeviceId().no);
    int length = in->size();
    int threads = std::min(MAX_THREADS, length);
    int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
    gElementGelu<TT_DATA_TYPE><<<blocks, threads>>>(gout, gin);

#ifndef CUDA_DEBUG
    cudaStreamSynchronize(0);
#endif
}

void ReluOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in)
{
	Functional::HUTensor<TT_DATA_TYPE> gin(in);
	Functional::HUTensor<TT_DATA_TYPE> gout(out);

	cudaSetDevice(in->getDeviceId().no);
	int length = in->size();
	int threads = std::min(MAX_THREADS, length);
	int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
	gElementRelu<TT_DATA_TYPE><<<blocks, threads>>>(gout, gin);
#ifndef CUDA_DEBUG
	cudaStreamSynchronize(0);
#endif
}

void SwishOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in)
{
	Functional::HUTensor<TT_DATA_TYPE> gin(in);
    Functional::HUTensor<TT_DATA_TYPE> gout(out);

    cudaSetDevice(in->getDeviceId().no);
    int length = in->size();
    int threads = std::min(MAX_THREADS, length);
    int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
    gElementSwish<TT_DATA_TYPE><<<blocks, threads>>>(gout, gin);
#ifndef CUDA_DEBUG
    cudaStreamSynchronize(0);
#endif
}

/*
extern "C" __global__ void gElementNeg(Functional::HUTensor<TT_DATA_TYPE> ga, Functional::HUTensor<TT_DATA_TYPE> gb)
{
    int length = ga.shape().elements();
    for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
        int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
        if(index < length) {
            ga[index] = - gb[index];
        }   
    }   
}


void NegOP(HUPtr<HUTensor> &a, HUPtr<HUTensor> b)
{
	cudaSetDevice(a->getDeviceId().no);

	Functional::HUTensor<TT_DATA_TYPE> ga(a);
	Functional::HUTensor<TT_DATA_TYPE> gb(b);
	int length = a->size();
	int threads = std::min(MAX_THREADS, length);
	int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

	gElementNeg<TT_DATA_TYPE><<<blocks, threads>>>(ga, gb);
}
*/

template <typename T>
__global__ 
void gElementNeg(T* a, 
                 const T* b,
                 const int length)
{
    for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
        int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
        if(index < length) {
            a[index] = - b[index];
        }   
    }   
}

void NegOP(HUPtr<HUTensor> &a, HUPtr<HUTensor> b)
{
    cudaSetDevice(a->getDeviceId().no);
    int length = b->shape().elements();
    int threads = std::min(MAX_THREADS, length);
    int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
    gElementNeg<TT_DATA_TYPE><<<blocks, threads>>>(a->data(), b->data(), length);
}

template <bool add, typename T>
__global__ 
void gTranspose0213(T* out, 
                    const T* in, 
                    int rows, 
                    int cols, 
                    int stride1, 
                    int stride2) 
{
  int stride = stride1 * stride2;
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOut = out + j * cols;

      int z = j / stride;
      int y = (j % stride) / stride1;
      int x = (j % stride) % stride1;
      int j2 = z * stride + x * stride2 + y;

      const T* rowIn = in + j2 * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(add)
            rowOut[i] += rowIn[i];
          else
            rowOut[i] = rowIn[i];
        }
      }
    }
  }
}

/*
template <bool add>
__global__ void gTransposeND(
    Functional::HUTensor<float> &out,
    const Functional::HUTensor<float> in,
    const int* permute) {
  //constexpr size_t N = ;
  //functional::Array<int, N> oDims;
  //functional::Array<int, N> pDims;
  int oDims[4];
  int pDims[4];

  int length = out.shape().elements();
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out.shape().dims(index, oDims);
      for(int i = 0; i < 4; ++i)
        pDims[permute[i]] = oDims[i];
      if(add)
        out[index] += in.GetItemByIndices(pDims);
      else
        out[index] = in.GetItemByIndices(pDims);
    }
  }
}
*/

/*
template <bool add, typename T>
__global__ 
void gTransposeND(T* out, 
                  const T* in, 
                  const Functional::Array<int, 4> permute, 
                  const int length) 
{
  //size_t N = 4;
  Functional::Array<int, 4> oDims;
  Functional::Array<int, 4> pDims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out.shape().dims(index, oDims);
      for(int i = 0; i < 4; ++i)
        pDims[permute[i]] = oDims[i];
      if(add)
        out[index] += in[pDims];
      else
        out[index] = in[pDims];
    }
  }
} */

template <bool add, typename T>
__global__ 
void gTransposeND(Functional::HUTensor<T> out, 
                  const Functional::HUTensor<T> in, 
                  const Functional::Array<int, 4> permute) 
{
  //size_t N = 4;
  Functional::Array<int, 4> oDims;
  Functional::Array<int, 4> pDims;

  int length = out.shape().elements();
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out.shape().dims(index, oDims);
      for(int i = 0; i < 4; ++i)
        pDims[permute[i]] = oDims[i];
      if(add)
        out[index] += in[pDims];
      else
        out[index] = in[pDims];
    }
  }
}

void TransposeND(HUPtr<HUTensor> &out, HUPtr<HUTensor> in, const std::vector<int>& vAxis)
{
	//cudaSetDevice(out->getDeviceId().no);
	if(vAxis == std::vector<int>({0, 2, 1, 3})) {
		cudaSetDevice(out->getDeviceId().no);
    	int rows = out->shape().elements() / out->shape().back();
    	int cols = out->shape().back();

    	int blocks = std::min(MAX_BLOCKS, rows);
    	int threads = std::min(MAX_THREADS, cols);

    	int stride1 = out->shape()[-2];
    	int stride2 = out->shape()[-3];

    	gTranspose0213<false, TT_DATA_TYPE><<<blocks, threads>>>(out->data(), in->data(), rows, cols, stride1, stride2);
  	}
	else {
    /*int axes[4];
    int diff = 4 - vAxis.size();
    for(int i = 0; i < 4; ++i)
      if(i < diff)
        axes[i] = i;
      else
        axes[i] = vAxis[i - diff] + diff;

	//for(int i=0; i < 4; i++)
	//	std::cout << axes[i] << std::endl;

    int length = out->shape().elements();
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
	
	//std::cout << out->shape().toString() << std::endl;
	//std::cout << in->shape().toString() << std::endl;
	//Functional::HUTensor<float> gout(*out);
	//Functional::HUTensor<float> gin(*in);
	cudaSetDevice(out->getDeviceId().no);

	std::cout << "test1" << std::endl;
    gTransposeND<false><<<blocks, threads>>>(out, in, axes);
	//std::cout << gin[0] << std::endl;
	std::cout << "test2" << std::endl;*/
	Functional::Array<int, 4> axes;
    int diff = 4 - vAxis.size();
    for(int i = 0; i < axes.size(); ++i)
      if(i < diff)
        axes[i] = i;
      else
        axes[i] = vAxis[i - diff] + diff;

    int length = out->shape().elements();
    int threads = std::min(MAX_THREADS, length);
    int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    gTransposeND<false, TT_DATA_TYPE><<<blocks, threads>>>(out, in, axes);
  }
#ifndef CUDA_DEBUG
    cudaStreamSynchronize(0);
#endif
}

//C= alpha*A*B + beta*C 
void Prod(HUPtr<HUTensor> &C, const HUPtr<HUTensor> & A, const HUPtr<HUTensor> & B, bool transA, bool transB, float beta, float alpha) {
  cudaSetDevice(C->getDeviceId().no);
  // float alpha = scalar;

  size_t m = A->shape().elements() / A->shape().back();
  size_t k = A->shape().back();
  if(transA)
    std::swap(m, k);

  size_t l = B->shape().elements() / B->shape().back();
  size_t n = B->shape().back();
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape().back();
  size_t ldb = B->shape().back();
  size_t ldc = B->shape().back();

  if(transB)
    ldc = B->shape().elements() / B->shape().back();

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto cublasHandle = C->getDevice()->getCublasHandle();

#if CUDA_VERSION >= 9000
  cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
#endif
 
  /*
  cublasSgemm(cublasHandle,
              opB,
              opA,
              n,
              m,
              k,
              &alpha,
              B->data(),
              ldb,
              A->data(),
              lda,
              &beta,
              C->data(),
              ldc);
  */

  cudaDataType_t BType, AType, CType;
  cudaDataType_t computeType;
  int cublasAlgo;
  if (sizeof(TT_DATA_TYPE) == sizeof(half))  // fp16 
  {
      BType = CUDA_R_16F;
      AType = CUDA_R_16F;
      CType = CUDA_R_16F;
      computeType = CUDA_R_16F;
      cublasAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      // cublasAlgo = CUBLAS_GEMM_DEFAULT;
  } 
  else  // fp32, and others data type
  {
      BType = CUDA_R_32F;
      AType = CUDA_R_32F;
      CType = CUDA_R_32F;
      computeType = CUDA_R_32F;
      cublasAlgo = CUBLAS_GEMM_DEFAULT;
  }

  TT_DATA_TYPE alpha_ = (TT_DATA_TYPE)alpha;
  TT_DATA_TYPE beta_ = (TT_DATA_TYPE)beta;
  cublasGemmEx(cublasHandle, 
               opB, 
               opA, 
               n, 
               m, 
               k, 
               &alpha_, 
               B->data(), BType, ldb, 
               A->data(), AType, lda, 
               &beta_, 
               C->data(), CType, ldc, 
               computeType, 
               static_cast<cublasGemmAlgo_t>(cublasAlgo));

  // FP16
  /*
  cublasGemmEx(cublasHandle,
               opB,
               opA,
               n,
               m,
               k,
               &alpha,
               B->data(), CUDA_R_16F,
               ldb,
               A->data(), CUDA_R_16F,
               lda,
               &beta,
               C->data(), CUDA_R_16F,
               ldc,
               CUDA_R_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  */

  /*
  cublasGemmEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                d_B, BType, n,
                d_A, AType, k,
                &beta,
                d_C, CType, n,
                computeType,
                static_cast<cublasGemmAlgo_t>(algo));
  */

#if CUDA_VERSION >= 9000
  cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
#endif

}

/*
__global__ void gAddBias_V2(float* out,
                            const float* bias,
                            size_t length,
                            size_t cols) 
{
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      size_t index2 = index % cols;
      out[index] += bias[index2];
    }
  }
} */

template <typename T>
__global__ 
void gElementAddBiasGelu(T* out, 
                         const T* bias, 
                         size_t length, 
                         size_t cols) 
{
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      float addBias = (float)out[index] + (float)bias[index % cols];
      out[index] = (T)( 0.5f * addBias * (1.0f + erf(addBias / sqrtf(2.0f))) ); 
    }
  }
}

/*
template <>
__global__
void gElementAddBiasGelu(half* out, 
                         const half* __restrict bias, 
                         int m, 
                         int n)
{
  half2* out_ptr = (half2*) out;
  const half2* bias_ptr = (half2*) bias;

  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    half2 reg_bias = __ldg(&bias_ptr[id % n]);
    half2 val = out_ptr[id] + reg_bias;
    out_ptr[id] = gelu(val);
  }
} */

template <typename T>
__global__ 
void gElementAddBiasRelu_V2(T* out, 
                            const T* bias, 
                            size_t length, 
                            size_t cols) 
{
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      T addBias = out[index] + bias[index % cols];
      out[index] = addBias > (T)0.f ? addBias : (T)0.f;
    }
  }
}

template <typename T>
__global__
void gElementAddBiasRelu(T* out, const T* __restrict bias, int m, int n)
{
  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    T val = out[id] + __ldg(&bias[id % n]);
    out[id] = val > (T)0.0f ? val : (T)0.0f;
  }
}

template <>
__global__
void gElementAddBiasRelu(half* out, const half* __restrict bias, int m, int n)
{
  half2* out_ptr = (half2*) out;
  const half2* bias_ptr = (half2*) bias;

  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    half2 reg_bias = __ldg(&bias_ptr[id % n]);
    half2 val = out_ptr[id] + reg_bias;
    val.x = val.x > (half)0.0f ? val.x : (half)0.0f;
    val.y = val.y > (half)0.0f ? val.y : (half)0.0f;
    out_ptr[id] = val;
  }
}

template <typename T>
__global__ 
void gElementAddBiasSwish(T* out, 
                          const T* bias, 
                          size_t length, 
                          size_t cols) 
{
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      float addBias = (float)out[index] + (float)bias[index % cols];
      float sigmoid = addBias > 0 ? (1.f / (1.f + expf(-addBias))) : (expf(addBias) / (1.f + expf(addBias)));
      out[index] = (T)(addBias * sigmoid);
    }
  }
}

template <typename T>
__global__ 
void gAddBias_V2(T* out, 
                 const T* bias, 
                 size_t length, 
                 size_t cols)
{
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      // size_t index2 = index % cols;
      out[index] += bias[index % cols];
    }
  }
}

void AddBias_V2(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias)
{
	cudaSetDevice(C->getDeviceId().no);

	int length = C->shape().elements();
	int cols = bias->shape().elements();

	int threads = std::min(MAX_THREADS, length);
	int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

	gAddBias_V2<TT_DATA_TYPE><<<blocks, threads>>>(C->data(), bias->data(), length, cols);
#ifndef CUDA_DEBUG
	cudaStreamSynchronize(0);
#endif
}

template <typename T>
__global__
void gAddBias(T* out, const T* __restrict bias, int m, int n)
{
  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    out[id] += __ldg(&bias[id % n]);
  }
}

template <>
__global__
void gAddBias(half* out, const half* __restrict bias, int m, int n)
{
  half2* out_ptr = (half2*) out;
  const half2* bias_ptr = (half2*) bias;

  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    half2 reg_bias = __ldg(&bias_ptr[id % n]);
    half2 val = out_ptr[id] + reg_bias;
    out_ptr[id] = val;
  }
}

void AddBias(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias)
{
    cudaSetDevice(C->getDeviceId().no);

    int m = C->shape().elements() / C->shape()[-1]; 
    int n = bias->shape().elements();
    const int data_type_factor = 4 / sizeof(TT_DATA_TYPE); // for fp32
    dim3 block, grid;
    if(n / 4 / data_type_factor <= 1024)
    {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else
    {
        block.x = 1024;
        grid.x = ceil(m * n / 1024.);
    }

    gAddBias<TT_DATA_TYPE><<<grid, block, 0>>>(C->data(), bias->data(), m, n / data_type_factor);
#ifndef CUDA_DEBUG
    cudaStreamSynchronize(0);
#endif
}

void AddBiasGeluOP(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias)
{
	cudaSetDevice(C->getDeviceId().no);

	int length = C->shape().elements();
	int cols = bias->shape().elements();

	int threads = std::min(MAX_THREADS, length);
	int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

	gElementAddBiasGelu<TT_DATA_TYPE><<<blocks, threads>>>(C->data(), bias->data(), length, cols);
#ifndef CUDA_DEBUG
	cudaStreamSynchronize(0);
#endif
}

void AddBiasReluOP_V2(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias)
{
    cudaSetDevice(C->getDeviceId().no);

    int length = C->shape().elements();
    int cols = bias->shape().elements();

    int threads = std::min(MAX_THREADS, length);
    int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    gElementAddBiasRelu_V2<<<blocks, threads>>>(C->data(), bias->data(), length, cols);
#ifndef CUDA_DEBUG
    cudaStreamSynchronize(0);
#endif
}

void AddBiasReluOP(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias)
{
    cudaSetDevice(C->getDeviceId().no);

    int m = C->shape().elements() / C->shape()[-1];
    int n = bias->shape().elements();
    const int data_type_factor = 4 / sizeof(TT_DATA_TYPE); // for fp32
    dim3 block, grid;
    if(n / 4 / data_type_factor <= 1024)
    {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else
    {
        block.x = 1024;
        grid.x = ceil(m * n / 1024.);
    }

    gElementAddBiasRelu<TT_DATA_TYPE><<<grid, block, 0>>>(C->data(), bias->data(), m, n / data_type_factor);
#ifndef CUDA_DEBUG
    cudaStreamSynchronize(0);
#endif
}

void AddBiasSwishOP(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias)
{
    cudaSetDevice(C->getDeviceId().no);

    int length = C->shape().elements();
    int cols = bias->shape().elements();

    int threads = std::min(MAX_THREADS, length);
    int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    gElementAddBiasSwish<TT_DATA_TYPE><<<blocks, threads>>>(C->data(), bias->data(), length, cols);
#ifndef CUDA_DEBUG
    cudaStreamSynchronize(0);
#endif
}

void ProdWithBias(HUPtr<HUTensor> &C, const HUPtr<HUTensor> & A, const HUPtr<HUTensor> & B, const HUPtr<HUTensor> &bias, bool transA, bool transB, float beta, float alpha)
{
    /*
	cudaEvent_t start, stop;
	float elapsedTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
    */
	
	Prod(C, A, B, transA, transB, beta, alpha);

    /*
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	LOG(info, "[item] Prod Time Cost: {}", elapsedTime);


	cudaEventRecord(start, 0);
    */

	AddBias(C, bias);


    /*
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	LOG(info, "[item] AddBias Time Cost: {}", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    */

}

void ProdBatchedOP(HUPtr<HUTensor> C, const HUPtr<HUTensor> A, const HUPtr<HUTensor> B, HUPtr<HUMemPool> mem, bool transA, bool transB, float beta, float alpha)
{
  cudaSetDevice(C->getDeviceId().no);

  size_t batchA = A->shape().elements() / (A->shape()[-1] * A->shape()[-2]);
  size_t batchB = B->shape().elements() / (B->shape()[-1] * B->shape()[-2]);

  size_t m = A->shape()[-2];
  size_t k = A->shape()[-1];
  if(transA)
    std::swap(m, k);

  size_t l = B->shape()[-2];
  size_t n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape()[-1];
  size_t ldb = B->shape()[-1];
  size_t ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape()[-2];

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto cublasHandle = C->getDevice()->getCublasHandle();

  int strideA = batchA == 1 ? 0 : m * k;
  int strideB = batchB == 1 ? 0 : n * k;
  int strideC = n * m;
  int batchC = std::max(batchA, batchB);

  std::vector<const TT_DATA_TYPE*> aptr;
  std::vector<const TT_DATA_TYPE*> bptr;
  std::vector<TT_DATA_TYPE*> cptr;

  for(int i = 0; i < batchC; i++) {
    aptr.push_back(A->data() + (i % batchA) * strideA);
    bptr.push_back(B->data() + (i % batchB) * strideB);
    cptr.push_back(C->data() + i * strideC);
  }

  auto mp_aptr = mem->alloc<const TT_DATA_TYPE*>(aptr.size());
  CudaCopy(
      aptr.data(), aptr.data() + aptr.size(), mp_aptr->data<const TT_DATA_TYPE*>());

  auto mp_bptr = mem->alloc<const TT_DATA_TYPE*>(bptr.size());
  CudaCopy(
      bptr.data(), bptr.data() + bptr.size(), mp_bptr->data<const TT_DATA_TYPE*>());

  auto mp_cptr = mem->alloc<TT_DATA_TYPE*>(cptr.size());
  CudaCopy(cptr.data(), cptr.data() + cptr.size(), mp_cptr->data<TT_DATA_TYPE*>());


#if CUDA_VERSION >= 9000
  cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
#endif
  /*
  cublasSgemmBatched(cublasHandle,
                     opB,
                     opA,
                     n,
                     m,
                     k,
                     &alpha,
                     mp_bptr->data<const float*>(),
                     ldb,
                     mp_aptr->data<const float*>(),
                     lda,
                     &beta,
                     mp_cptr->data<float*>(),
                     ldc,
                     batchC);
  */

  cudaDataType_t BType, AType, CType;
  cudaDataType_t computeType;
  int cublasAlgo;
  if (sizeof(TT_DATA_TYPE) == sizeof(half))  // fp16 
  {
      BType = CUDA_R_16F;
      AType = CUDA_R_16F;
      CType = CUDA_R_16F;
      computeType = CUDA_R_16F;
      cublasAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }
  else  // fp32, and others data type
  {
      BType = CUDA_R_32F;
      AType = CUDA_R_32F;
      CType = CUDA_R_32F;
      computeType = CUDA_R_32F;
      cublasAlgo = CUBLAS_GEMM_DEFAULT;
  }

  TT_DATA_TYPE alpha_ = (TT_DATA_TYPE)alpha;
  TT_DATA_TYPE beta_ = (TT_DATA_TYPE)beta;
  cublasGemmBatchedEx(cublasHandle, 
                      opB, 
                      opA,
                      n,
                      m,
                      k,
                      &alpha_,
                      mp_bptr->data<const void*>(), BType,
                      ldb, 
                      mp_aptr->data<const void*>(), AType, 
                      lda, 
                      &beta_, 
                      mp_cptr->data<void*>(), CType, 
                      ldc, 
                      batchC, 
                      computeType, 
                      static_cast<cublasGemmAlgo_t>(cublasAlgo));
 
  /*
  cublasGemmStridedBatchedEx(cublasHandle,
                             opB, 
                             opA,
                             n,
                             m, 
                             k, 
                             &alpha, 
                             mp_bptr->data<const float*>(), CUDA_R_32F, 
                             ldb, 
                             strideB, 
                             mp_aptr->data<const float*>(), CUDA_R_32F, 
                             lda, 
                             strideA, 
                             &beta, 
                             mp_cptr->data<float*>(), CUDA_R_32F, 
                             ldc, 
                             strideC, 
                             batchC, 
                             CUDA_R_32F, 
                             static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
   */

  /*
  cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq_len, seq_len, size_per_head,
        &alpha,
        k_buf->data(), CUDA_R_32F, size_per_head, seq_len * size_per_head,
        q_buf->data(), CUDA_R_32F, size_per_head, seq_len * size_per_head,
        &beta,
        qk_buf->data(), CUDA_R_32F, seq_len, seq_len * seq_len,
        batch_size * head_num,
        CUDA_R_32F,
        static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT)); */

#if CUDA_VERSION >= 9000
  cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
#endif

  mem->free(mp_aptr);
  mem->free(mp_bptr);
  mem->free(mp_cptr);
}

template <typename T>
__global__ 
void gSoftmax(T* out, 
              Functional::HUConstantShape outShape, 
              const T* in, const T* mask, 
              const Functional::HUConstantShape maskShape) 
{
  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  bool broadcast = outShape != maskShape;
  //functional::Array<int, functional::Shape::size()> dims;
  int dims[4];	

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* so = out + j * cols;
      const T* sp = in + j * cols;

      extern __shared__ float _share[];

      float* _max = _share + blockDim.x;
      _max[threadIdx.x] = -CUDA_FLT_MAX;  // mask
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float mVal = 1.f;
          if(mask) {
            int mIndex = id + j * cols;
            if(broadcast) {
              outShape.dims(mIndex, dims);
              mIndex = maskShape.bindex(dims);
            }
            mVal = (float)mask[mIndex];   // convert to fp32
          }

          float sp_val = (float)sp[id];   // convert to fp32
          if(mVal && sp_val > _max[threadIdx.x])
            _max[threadIdx.x] = sp_val;
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float mVal = 1.f;
          if(mask) {
            int mIndex = id + j * cols;
            if(broadcast) {
              outShape.dims(mIndex, dims);
              mIndex = maskShape.bindex(dims);
            }
            mVal = (float)mask[mIndex];   // convert to fp32
          }

          float ex = 0;
          float sp_val = (float)sp[id];   // convert to fp32
          if(mVal)
            ex = __expf(sp_val - max);
          so[id] = (T)ex;                 // convert to T

          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          so[id] = (T)((float)so[id] / _sum[0]);   // convert to T
        }
      }
    }
  }
}

void SoftmaxOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in, HUPtr<HUTensor> mask)
{
	cudaSetDevice(out->getDeviceId().no);

	size_t m = out->shape().elements() / out->shape().back();
	size_t k = out->shape().back();

	int blocks = std::min(MAX_BLOCKS, (int)m);
	int threads = std::min(MAX_THREADS, (int)k);
	// int shared = sizeof(TT_DATA_TYPE) * threads * 2; 
    int shared = sizeof(float) * threads * 2;

	if(mask)
		gSoftmax<TT_DATA_TYPE><<<blocks, threads, shared>>>(out->data(), out->shape(), in->data(), mask->data(), mask->shape());
	else 
		gSoftmax<TT_DATA_TYPE><<<blocks, threads, shared>>>(out->data(), out->shape(), in->data(), 0, out->shape());
}

template <typename T>
__global__ 
void gLogSoftmax(T* out, 
                 const Functional::HUConstantShape outShape, 
                 const T* in) 
{
  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* so = out + j * cols;
      const T* sp = in + j * cols;

      extern __shared__ float _share[];

      float* _max = _share + blockDim.x;
      _max[threadIdx.x] = (float)sp[threadIdx.x];    // convert to fp32
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        float sp_val = (float)sp[id];  // convert to fp32
        if(id < cols) {
          if(sp_val > _max[threadIdx.x])
            _max[threadIdx.x] = sp_val;
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float sm = (float)sp[id] - max;   // convert to fp32
          float ex = __expf(sm);
          so[id] = (T)sm;                   // convert to T
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          // so[id] -= __logf(_sum[0]);
          so[id] = (T)((float)so[id] - __logf(_sum[0]));  // convert to T
      }
    }
  }
}

template <typename T>
__global__ 
void gAddBiasLogSoftmax(T* out, 
                        const Functional::HUConstantShape outShape, 
                        const T* in, 
                        const T* bias) 
{
  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* so = out + j * cols;
      const T* sp = in + j * cols;

      extern __shared__ float _share[];

      float* _max = _share + blockDim.x;
      _max[threadIdx.x] = (float)sp[threadIdx.x] + (float)bias[threadIdx.x]; ////
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        float total = (float)sp[id] + (float)bias[id];
        if(id < cols) {
          if(total > _max[threadIdx.x]) ////
            _max[threadIdx.x] = total;  ////
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();
     
      float* _sum = _share + blockDim.x;

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float sm = (float)sp[id] + (float)bias[id] - max;
          float ex = __expf(sm);
          so[id] = (T)sm;            // convert to T
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          // so[id] -= __logf(_sum[0]);
          so[id] =(T)( (float)so[id] - __logf(_sum[0]) );  // convert to T
      }
    }
  }
}

template <typename T>
__global__ 
void gAddBiasLogSoftmax_V2(T* logits, 
                           const T* tmp_logits, 
                           const T* bias, 
                           const int n, 
                           const int beam_size, 
                           const uint8_t* finished)
{
  int bid = blockIdx.x;
  bool finish = (finished != nullptr) ? finished[bid / beam_size] : false;
  int offset = bid * n;
  int end_id = (int)EOS_ID; 
  
  float max_val = -1 * FLT_MAX;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  if(finish)
  { 
    for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
    {
      logits[offset + tid] = (tid == end_id) ? (T)0.f : -MAX_T_VAL;
    }
  }
  else
  {  
    for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
    {
      if(finish) {
        logits[offset + tid] = (tid == end_id) ? MAX_T_VAL : -MAX_T_VAL;
      }
      else {
        logits[offset + tid] = tmp_logits[offset + tid] + bias[tid];
      }
      max_val = max(max_val, (float)logits[offset + tid]);
    }

    max_val = blockReduceMax<float>((float)max_val);
    if(threadIdx.x == 0)
      s_max_val = max_val;
    __syncthreads();

    float sum_val = 0.0f;
    for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
    {
      logits[offset + tid] = (T)(__expf((float)logits[offset + tid] - s_max_val));
      sum_val += (float)logits[offset + tid];
    }

    sum_val = blockReduceSum<float>(sum_val);
    if(threadIdx.x == 0)
      s_sum_val = sum_val;
    __syncthreads();

    for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
    {
      logits[offset + tid] = (T)(logf((float)logits[offset + tid] / s_sum_val));
    }
  }
}

void LogSoftmaxOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in) {
  cudaSetDevice(out->getDeviceId().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t k = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(TT_DATA_TYPE) * threads * 2;

  gLogSoftmax<TT_DATA_TYPE><<<blocks, threads, shared>>>(
      out->data(), out->shape(), in->data());
}

void AddBiasLogSoftmaxOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in, const HUPtr<HUTensor> bias)
{
    cudaSetDevice(out->getDeviceId().no);

    size_t m = out->shape().elements() / out->shape().back();
    size_t k = out->shape().back();

    int blocks = std::min(MAX_BLOCKS, (int)m);
    int threads = std::min(MAX_THREADS, (int)k);
    // int shared = sizeof(TT_DATA_TYPE) * threads * 2;
    int  shared = sizeof(float) * threads * 2;

    gAddBiasLogSoftmax<TT_DATA_TYPE><<<blocks, threads, shared>>>(
            out->data(), out->shape(), in->data(), bias->data());
}

void AddBiasLogSoftmaxOP_V2(HUPtr<HUTensor> out, HUPtr<HUTensor> in, const HUPtr<HUTensor> bias, const int realDimBatch, uint8_t* isAllDone)
{
    cudaSetDevice(out->getDeviceId().no);
    int m = out->shape().elements() / out->shape().back();
    int n = out->shape().back();
    int local_beam_size = out->shape()[-3] / realDimBatch;  // if step=0, local_beam_size=1

    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    /* n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
    gAddBiasLogSoftmax_V2<TT_DATA_TYPE><<<grid, block, 0>>>(
            out->data(), in->data(), bias->data(), n, local_beam_size, isAllDone);
}

__global__ void gLNormalization_v2(float* out, 
                                   const float* input,  
                                   const float* gamma, 
                                   const float* beta, 
                                   int m, 
                                   int n,
                                   float eps=1e-9)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  for(int i = tid; i < n; i += blockDim.x)
    local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i]);

  mean = blockReduceSum(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum((local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + eps;
  __syncthreads();

  for(int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] = 
        (float)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

template <typename T>
__global__ 
void gLNormalization(T* out, 
                     const T* in, 
                     const T* alpha, 
                     const T* beta, 
                     int rows, 
                     int cols, 
                     float eps=1e-9) 
{
  extern __shared__ float _share[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* so = out + j * cols;
      const T* sp = in + j * cols;

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = 0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += (float)sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = _sum[0] / cols;
      __syncthreads();

      float* _sqSum = _share + blockDim.x;

      _sqSum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = (float)sp[id] - mean;
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (_sqSum[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float t = (float)alpha[id] * (((float)sp[id] - mean) / sigma);
          if(beta != nullptr)
            t += (float)beta[id];
          so[id] = (T)t;
        }
      }
    }
  }
}

template <typename T>
__global__ 
void add_bias_input_layer_norm_kernel_generalize(const T* __restrict input, 
                                                 const T* __restrict bias,
                                                 const T* __restrict gamma, 
                                                 const T* __restrict beta, 
                                                 T* output,
                                                 T* norm_output, 
                                                 int m, 
                                                 int n, 
                                                 float eps=1e-9)
{
  const int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for(int i = tid; i < n; i+= blockDim.x)
  {
    float local_out = (float)(__ldg(&input[blockIdx.x * n + i]));
    local_out += (float)(output[blockIdx.x * n + i]);
    local_out += (float)(__ldg(&bias[i]));
    output[blockIdx.x * n + i] = (T)local_out;
    local_sum += local_out;
  }

  mean = blockReduceSum<float>(local_sum);

  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for(int i = tid; i < n; i+= blockDim.x)
  {
    float diff = (float)(__ldg(&output[blockIdx.x * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if(threadIdx.x == 0) {
    s_variance = rsqrtf(variance / n + eps);
  }

  __syncthreads();

  for(int i = tid; i < n; i+= blockDim.x)
  {
    if (beta != nullptr) {
      norm_output[blockIdx.x * n + i] =
        (T)((( (float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
    }
    else {
      norm_output[blockIdx.x * n + i] =
        (T)((( (float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(__ldg(&gamma[i])));
    }
  }
}

template <typename T>
__global__ 
void layer_norm_kernel_generalize(const T* __restrict input, 
                                  const T* __restrict gamma, 
                                  const T* __restrict beta, 
                                  T* output, 
                                  int m, 
                                  int n, 
                                  float eps=1e-9)
{
  const int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for(int i = tid; i < n; i+= blockDim.x)
  {
    local_sum += (float)(__ldg(&input[blockIdx.x * n + i]));
  }

  mean = blockReduceSum<float>(local_sum);

  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for(int i = tid; i < n; i+= blockDim.x)
  {
    float diff = (float)(__ldg(&input[blockIdx.x * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if(threadIdx.x == 0) {
    s_variance = rsqrtf(variance / n + eps);
  }

  __syncthreads();

  for(int i = tid; i < n; i+= blockDim.x)
  {
    if (beta != nullptr) {
      output[blockIdx.x * n + i] = 
        (T)((( (float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
    }
    else {
      output[blockIdx.x * n + i] = 
        (T)((( (float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(__ldg(&gamma[i])));
    }
  }
}

void AddBiasInputLayerNormalOP(HUPtr<HUTensor> norm_out, HUPtr<HUTensor> out, HUPtr<HUTensor> in, const HUPtr<HUTensor> bias, HUPtr<HUTensor> gamma, HUPtr<HUTensor> beta, float eps)
{
    cudaSetDevice(norm_out->getDeviceId().no);

    int m = in->shape().elements() / in->shape().back();
    int n = in->shape().back();

    dim3 grid(m);
    dim3 block(min(n, 1024));

    if(n % 32 != 0) {
        block.x = 1024;
    }
    block.x = block.x / (4 / sizeof(TT_DATA_TYPE)); // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision*/
    add_bias_input_layer_norm_kernel_generalize<TT_DATA_TYPE><<<grid, block, 0>>>(in->data(), bias->data(), gamma->data(), \ 
            beta ? beta->data() : nullptr, out->data(), norm_out->data(), m, n, eps);
    // layer_norm_kernel_generalize<T><<<grid, block, 0, stream>>>(input, gamma, beta, output, m, n); // For gpt-3
}

void LayerNormalOP_V2(HUPtr<HUTensor> out, HUPtr<HUTensor> in, HUPtr<HUTensor> gamma, HUPtr<HUTensor> beta, float eps)
{
    cudaSetDevice(out->getDeviceId().no);

    int m = in->shape().elements() / in->shape().back();
    int n = in->shape().back();

    dim3 grid(m);
    dim3 block(min(n, 1024));

    if(n % 32 != 0) {
        block.x = 1024;
    }
    block.x = block.x / (4 / sizeof(TT_DATA_TYPE)); // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision */
    layer_norm_kernel_generalize<TT_DATA_TYPE><<<grid, block, 0>>>(in->data(), gamma->data(), \ 
            beta ? beta->data() : nullptr, out->data(), m, n, eps);

    // layer_norm_kernel_generalize<T><<<grid, block, 0, stream>>>(input, gamma, beta, output, m, n); // For gpt-3
}

void LayerNormalOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in, HUPtr<HUTensor> gamma, HUPtr<HUTensor> beta, float eps)
{
	 cudaSetDevice(out->getDeviceId().no);

	 int rows = in->shape().elements() / in->shape().back();
	 int cols = in->shape().back();

	 int blocks = std::min(MAX_BLOCKS, (int)rows);
	 int threads = std::min(MAX_THREADS, (int)cols);
	 int shared = 2 * threads * sizeof(TT_DATA_TYPE);

	 gLNormalization<TT_DATA_TYPE><<<blocks, threads, shared>>>(out->data(), in->data(), gamma->data(),\ 
             beta ? beta->data() : nullptr, rows, cols, eps);
}

void ConcatContOP(HUPtr<HUTensor> out, const std::vector<HUPtr<HUTensor> >& inputs, int axis) {
  cudaSetDevice(out->getDeviceId().no);
  int step = 1;
  for(int i = 0; i < axis; ++i)
    step *= out->shape()[i];

  size_t offset1 = 0;
  for(int i = 0; i < step; ++i) {
    for(auto in : inputs) {
      size_t size = in->shape().elements() / step;
      size_t offset2 = i * size;

      cudaMemcpy(out->data() + offset1,
                 in->data() + offset2,
                 size * sizeof(TT_DATA_TYPE),
                 cudaMemcpyDeviceToDevice);

      offset1 += size;
    }
  }

#ifndef CUDA_DEBUG
  cudaStreamSynchronize(0);
#endif
}

template <bool add, typename T>
__global__ 
void gInsertCols(T* out, 
                 const T* in, 
                 size_t rows, 
                 size_t cols, 
                 size_t cols_out, 
                 size_t cols_in, 
                 size_t offset_out, 
                 size_t offset_in) 
{
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOut = out + j * cols_out + offset_out;
      const T* rowIn = in + j * cols_in + offset_in;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          if(add)
            rowOut[i] += rowIn[i];
          else
            rowOut[i] = rowIn[i];
      }
    }
  }
}

void Concatenate1OP(HUPtr<HUTensor> out, const std::vector<HUPtr<HUTensor> >& inputs) {
  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();

  size_t offset = 0;
  int cols_out = out->shape().back();

  for(auto in : inputs) {
    ABORT_IF(rows != in->shape().elements() / in->shape().back(),
             "First dimension must be equal");
    int cols_in = in->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_in);

    gInsertCols<false, TT_DATA_TYPE><<<blocks, threads>>>(
        out->data(), in->data(), rows, cols_in, cols_out, cols_in, offset, 0);
    offset += cols_in;
  }

#ifndef CUDA_DEBUG
  cudaStreamSynchronize(0);
#endif
}

template <typename T>
__global__ 
void gJoin2(T* out, 
            size_t rowBatch, 
            size_t cols, 
            const T* in1, 
            size_t inStride1, 
            const T* in2, 
            size_t inStride2) 
{
  int outStride = inStride1 + inStride2;
  int rows = rowBatch * outStride;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOut = out + j * cols;

      int curBatch = j / outStride;
      int curPos = j % outStride;

      int jIn1 = (curBatch * inStride1) + curPos;
      int jIn2 = (curBatch * inStride2) + curPos - inStride1;

      const T* rowIn1 = in1 + jIn1 * cols;
      const T* rowIn2 = in2 + jIn2 * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(curPos < inStride1)
            rowOut[i] = rowIn1[i];
          else
            rowOut[i] = rowIn2[i];
        }
      }
    }
  }
}

void Concatenate2OP(HUPtr<HUTensor> out, HUPtr<HUTensor> in1, HUPtr<HUTensor> in2) {
  cudaSetDevice(out->getDeviceId().no);

  size_t rows = out->shape().elements() / out->shape().back();
  size_t cols = out->shape().back();

  size_t rowStride1 = in1->shape()[-2];
  size_t rowStride2 = in2->shape()[-2];

  size_t rowBatch = rows / out->shape()[-2];

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);

  gJoin2<TT_DATA_TYPE><<<blocks, threads>>>(out->data(), 
                                            rowBatch, 
                                            cols, 
                                            in1->data(), 
                                            rowStride1, 
                                            in2->data(), 
                                            rowStride2);
#ifndef CUDA_DEBUG
  cudaStreamSynchronize(0);
#endif
}

void ConcatenateOP(HUPtr<HUTensor> out, const std::vector<HUPtr<HUTensor> >& inputs, int ax) {
  if(ax == out->shape().size() - 1)
    Concatenate1OP(out, inputs);
  else if(ax == out->shape().size() - 2 && inputs.size() == 2)
    Concatenate2OP(out, inputs[0], inputs[1]);
  else
    ConcatContOP(out, inputs, ax);
}

/*
 * 1. << query_buf:   [batch_size*beam_size, 1, hidden_size]
 * 2. << Q_bias:      [hidden_size]; K_bias: [hidden_size]; V_bias: [hidden_size];
 * 3. << key_cache:   [batch_size*beam_size, seq_len, hidden_size]
 * 4. << value_cache: [batch_size*beam_size, seq_len, hidden_size]
 * 5. << lengths:     [batch_size*beam_size]
 * 6. >> context_buf: [batch_size*beam_size, 1, hidden_size] 
 *
 */
void CrossAttentionOP(
        HUPtr<HUTensor> query_buf, const HUPtr<HUTensor> Q_bias, 
        HUPtr<HUTensor> key_cache, const HUPtr<HUTensor> K_bias, 
        HUPtr<HUTensor> value_cache, const HUPtr<HUTensor> V_bias, 
        HUPtr<HUTensor> lengths, HUPtr<HUTensor> context_buf, 
        const int realDimBatch, const uint8_t* isAllDone, 
        const int head_num, const int step)
{
    cudaSetDevice(query_buf->getDeviceId().no);
    
    int local_beam_size = query_buf->shape()[-3] / realDimBatch;  // step=0, local_beam_size=1
    int batch_size = query_buf->shape()[-3];                  // batch*beam_size
    int seq_len = key_cache->shape()[-2];                     // seq_len
    int size_per_head = query_buf->shape()[-1] / head_num;

    const int block_sz = ATTENTION_BLOCK_SIZE;
    float scalar = 1.f / sqrtf(size_per_head * 1.0f);
    dim3 grid(batch_size * head_num);                         

    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    switch (cond)
    {
      case 32:
        cross_attention_kernel_opt<TT_DATA_TYPE, 32, block_sz><<<grid, block_sz, sizeof(float)*seq_len>>>(
          query_buf->data(), Q_bias->data(), 
          key_cache->data(), K_bias->data(), 
          value_cache->data(), V_bias->data(), 
          lengths->data(), context_buf->data(), 
          isAllDone, local_beam_size, 
          batch_size, head_num, step, seq_len, scalar);
        break;

      case 64:
        cross_attention_kernel_opt<TT_DATA_TYPE, 64, block_sz><<<grid, block_sz, sizeof(float)*seq_len>>>(
          query_buf->data(), Q_bias->data(), 
          key_cache->data(), K_bias->data(), 
          value_cache->data(), V_bias->data(), 
          lengths->data(), context_buf->data(), 
          isAllDone, local_beam_size, 
          batch_size, head_num, step, seq_len, scalar);
        break;

      case 128:
        cross_attention_kernel_opt<TT_DATA_TYPE, 128, block_sz><<<grid, block_sz, sizeof(float)*seq_len>>>(
          query_buf->data(), Q_bias->data(), 
          key_cache->data(), K_bias->data(), 
          value_cache->data(), V_bias->data(), 
          lengths->data(), context_buf->data(), 
          isAllDone, local_beam_size, 
          batch_size, head_num, step, seq_len, scalar);
        break;

      default:
        int block_size = 128;
        if(seq_len <= 64) {
          block_size = 64;
        }
        else if(seq_len <= 128 && seq_len > size_per_head) {
          block_size = 128;
        }
        else if(seq_len > 128 && seq_len <= 256) {
          block_size = 256;
        }
        else if(seq_len > 256 && seq_len <= 512) {
          block_size = 512;
        }
        else {
          block_size = 1024;
        }

        if(block_size < size_per_head) {
          block_size = size_per_head;
        }

        assert(block_size <= 1024);
        dim3 block(block_size);

        int shared_size = sizeof(TT_DATA_TYPE) * (size_per_head + seq_len);
        cross_attention_kernel<TT_DATA_TYPE><<<grid, block, shared_size>>>(
          query_buf->data(), Q_bias->data(), 
          key_cache->data(), K_bias->data(), 
          value_cache->data(), V_bias->data(), 
          lengths->data(), context_buf->data(), 
          isAllDone, local_beam_size, 
          batch_size, head_num, size_per_head, step, seq_len, scalar);
    }

#ifndef CUDA_DEBUG
    cudaStreamSynchronize(0);
#endif
    
}

void MaskedMultiHeadAttentionOP(
        const HUPtr<HUTensor> qkv_buf, const HUPtr<HUTensor> QKV_bias,
        HUPtr<HUTensor> key_cache, HUPtr<HUTensor> value_cache,
        HUPtr<HUTensor> context_buf, 
        const int realDimBatch, uint8_t* isAllDone, 
        const int head_num, const int step)
{
    using DataType = typename std::conditional<sizeof(TT_DATA_TYPE) == 4, float, uint16_t>::type;
    Masked_multihead_attention_params<DataType> params;
    memset(&params, 0, sizeof(params));

    //// int local_beam_size = qkv_buf->shape()[-3] / isAllDoneCopy.size(); // step=0, local_beam_size=1
    int local_beam_size = qkv_buf->shape()[-3] / realDimBatch; // step=0, local_beam_size=1
    int hidden_units = qkv_buf->shape()[-1] / 3;
    int size_per_head = hidden_units / head_num;

    params.q_bias = reinterpret_cast<const DataType *>(QKV_bias->data());
    params.k_bias = reinterpret_cast<const DataType *>(QKV_bias->data() + hidden_units);
    params.v_bias = reinterpret_cast<const DataType *>(QKV_bias->data() + 2 * hidden_units);

    params.out = reinterpret_cast<DataType *>(context_buf->data());

    params.q = reinterpret_cast<const DataType *>(qkv_buf->data());
    params.k = reinterpret_cast<const DataType *>(qkv_buf->data() + hidden_units);
    params.v = reinterpret_cast<const DataType *>(qkv_buf->data() + 2 * hidden_units);
    params.stride = 3 * hidden_units;
    params.finished = isAllDone;

    params.k_cache = reinterpret_cast<DataType *>(key_cache->data());
    params.v_cache = reinterpret_cast<DataType *>(value_cache->data());
    // params.batch_size = inference_batch_size;
    params.batch_size = qkv_buf->shape()[-3];

    params.seq_length = key_cache->shape()[-2];
    // params.seq_length = step+1;
    params.timestep = step;
    params.num_heads = head_num;
    params.hidden_size_per_head = size_per_head;
    params.inv_sqrt_dh = 1.f / sqrtf((float) params.hidden_size_per_head);
    params.beam_size = local_beam_size;

    /*
    std::cout << "seq_length: " << params.seq_length << "\thead_num: " << head_num << "\tsize_per_head: " << size_per_head
        << "\ttimestep: " << step << "\tsqrt_dh: " << params.inv_sqrt_dh << "\thidden_units: " << hidden_units << std::endl;
    */

    masked_multihead_attention(params);
}

#define NEW_TRANSPOSE_BATCH_MAJOR 1
template<typename T>
__global__ void transpose_4d_batch_major_k_cache(T* k_dst, const T* k_src,
                              const int head_num,
                              const int size_per_head,
                              const int seq_len,
                              const int max_seq_len)
{
  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  constexpr int X_ELEMS = (sizeof(T) == 4)? 4 : 8;

  auto key_src = reinterpret_cast<const uint4*>(k_src + batch_id * head_num * size_per_head * seq_len + head_id * size_per_head * seq_len);
  auto key_dst = reinterpret_cast<uint4*>(k_dst + batch_id * head_num * size_per_head * max_seq_len + head_id * size_per_head * max_seq_len);

  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size_per_head_div_x = size_per_head / X_ELEMS;
  if (out_idx >= head_num * size_per_head_div_x * max_seq_len) return;

  int idx = out_idx;
  const int k_seq_len_id = idx % max_seq_len;
  idx = (idx - k_seq_len_id) / max_seq_len;
  const int k_head_size_id = idx % size_per_head_div_x;

  if (k_seq_len_id < seq_len)
    key_dst[out_idx] = key_src[k_seq_len_id * size_per_head_div_x + k_head_size_id];
}

template<typename T>
__global__ void transpose_4d_batch_major_v_cache(T* v_dst, const T* v_src,
                              const int head_num,
                              const int size_per_head,
                              const int seq_len,
                              const int max_seq_len)
{
  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;

  // 16 byte loads will handle "x" dimension
  auto val_src = reinterpret_cast<const uint4*>(v_src + batch_id * head_num * size_per_head * seq_len + head_id * size_per_head * seq_len);
  auto val_dst = reinterpret_cast<uint4*>(v_dst + batch_id * head_num * size_per_head * max_seq_len + head_id * size_per_head * max_seq_len);

  // idx is over output dimension L * size_per_head / x for values
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr int X_ELEMS = (sizeof(T) == 4)? 4 : 8;
  const int size_per_head_div_x = size_per_head / X_ELEMS;

  if (idx >= size_per_head_div_x * seq_len) return;

  val_dst[idx] = val_src[idx];
}

template<typename T>
__global__ void transpose_4d_batch_major(T* k_dst, T* v_dst,
                              const T* k_src, const T* v_src,
                              const int head_num,
                              const int size_per_head,
                              const int seq_len,
                              const int max_seq_len)
{
    const int hidden_dim = head_num * size_per_head;
    const int x = (sizeof(T) == 4)? 4 : 8;
    const int size_per_head_split = size_per_head / x;
    const int batch_id = blockIdx.x;
    const int seq_id = blockIdx.y;

    for(int id = threadIdx.x; id < head_num * size_per_head_split * x; id += blockDim.x)
    {
        int tmp_id = id;
        int x_id = tmp_id % x;
        tmp_id = (tmp_id - x_id) / x;
        int size_id = tmp_id % size_per_head_split;
        tmp_id = (tmp_id - size_id) / size_per_head_split;
        int head_id = tmp_id % head_num;

        // key: [B, head_num, L, size_per_head / x, x] -> [B, head_num, size_per_head / x, L, x]
        k_dst[batch_id * hidden_dim * max_seq_len + head_id * size_per_head * max_seq_len + size_id * max_seq_len * x + seq_id * x + x_id] =
          k_src[batch_id * hidden_dim * seq_len + head_id * size_per_head * seq_len + seq_id * size_per_head + size_id * x + x_id];

        // value: [B, head_num, L, size_per_head / x, x] -> [B, head_num, L, size_per_head/x, x]
        v_dst[batch_id * hidden_dim * max_seq_len + head_id * size_per_head * max_seq_len + seq_id * size_per_head + size_id * x + x_id] =
          v_src[batch_id * hidden_dim * seq_len + head_id * size_per_head * seq_len + seq_id * size_per_head + size_id * x + x_id];
    }
}

// Use batch major
// put k/v_buf from shape [B, H, L, Dh]
// to cache [B, H, Dh/x, L, x]  and [B, H, L, Dh/x, x]
template<typename T>
void transpose_4d_batch_major_kernelLauncher(T* k_dst, /*T* v_dst,*/
                                  const T* k_src, /*const T* v_src,*/
                                  const int local_batch_size,
                                  const int seq_len,
                                  const int max_seq_len,
                                  const int size_per_head,
                                  const int local_head_num)
{
  constexpr int block_sz = 128;
// #if NEW_TRANSPOSE_BATCH_MAJOR == 1
  constexpr int x = (sizeof(T) == 4)? 4 : 8;
  int size = max_seq_len * size_per_head / x;
  dim3 grid((size + block_sz - 1) / block_sz, local_batch_size, local_head_num);
  dim3 grid_v((seq_len * size_per_head / x + block_sz - 1) / block_sz, local_batch_size, local_head_num);

  transpose_4d_batch_major_k_cache<<<grid, block_sz, 0>>>(
    k_dst, k_src,
    local_head_num,
    size_per_head,
    seq_len,
    max_seq_len
  );

  /*
  transpose_4d_batch_major_v_cache<<<grid_v, block_sz, 0>>>(
    v_dst, v_src,
    local_head_num,
    size_per_head,
    seq_len,
    max_seq_len
  ); 
#else
  dim3 grid(local_batch_size, seq_len);

  transpose_4d_batch_major<<<grid, block_sz, 0>>>(
    k_dst, v_dst,
    k_src, v_src,
    local_head_num,
    size_per_head,
    seq_len,
    max_seq_len
  );
#endif
*/
}

template void transpose_4d_batch_major_kernelLauncher(float* k_dst, /*float* v_dst,*/
  const float* k_src,/* const float* v_src,*/
  const int local_batch_size,
  const int seq_len,
  const int max_seq_len,
  const int size_per_head,
  const int local_head_num);

void Transpose4DBatchMajorOP(
        HUPtr<HUTensor> k_dst, /*HUPtr<HUTensor> v_dst, */
        const HUPtr<HUTensor> k_src, /*const HUPtr<HUTensor> v_src, */
        const int local_batch_size, const int seq_len, 
        const int max_seq_len, const int size_per_head, 
        const int local_head_num)
{
    transpose_4d_batch_major_kernelLauncher(
            k_dst->data(), /*v_dst->data(),*/
            k_src->data(), /*v_src->data(), */
            local_batch_size, seq_len,
            max_seq_len, size_per_head,
            local_head_num);
}

template <typename T>
__global__ void update_KV_batch_major_cache_kernel(
        const T* __restrict key_src_cache, 
        T* key_tgt_cache, 
        const T* __restrict value_src_cache, 
        T* value_tgt_cache, 
        const size_t* beam_ids, 
        const uint8_t* finished, 
        const int batch_size, 
        const int beam_width, 
        const int size_per_head, 
        const int step, 
        const int max_seq_len)
{
    int head_id = blockIdx.y;
    int bb_id = blockIdx.x;
    int batch_id = bb_id / beam_width;
    int beam_id = bb_id % beam_width;

    //// if(finished[batch_id * beam_width + beam_id]) return; 
    if(finished != nullptr && finished[batch_id]) return;

    const int hidden_dim = size_per_head * gridDim.y;

    int src_offset = (beam_ids[batch_id * beam_width + beam_id] * hidden_dim + 
                                                head_id * size_per_head) * max_seq_len;
    int tgt_offset = ((batch_id * beam_width + beam_id) * hidden_dim + 
                                                head_id * size_per_head) * max_seq_len;

    // for better memory access always do 16 byte loads.
    // [B, H, Dh/x, L, x]  and [B, H, L, Dh/x, x] (i.e. [B, H, L, Dh])
    auto key_src_ptr = reinterpret_cast<const uint4*>(key_src_cache + src_offset);
    auto value_src_ptr = reinterpret_cast<const uint4*>(value_src_cache + src_offset);
    auto key_tgt_ptr = reinterpret_cast<uint4*>(key_tgt_cache + tgt_offset);
    auto value_tgt_ptr = reinterpret_cast<uint4*>(value_tgt_cache + tgt_offset);
    constexpr int x = (sizeof(T) == 4)? 4 : 8;
    // constexpr int x = 4;

    // step starts from 1
    #if 0
    constexpr int WARP_SIZE = 32;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    for (int dhx = warp_id; dhx < size_per_head/x; dhx += num_warps)
    {
      for (int tid = lane_id; tid < step; tid += WARP_SIZE)
      {
        key_tgt_ptr[dhx * max_seq_len + tid] = key_src_ptr[dhx * max_seq_len + tid];
      }
    }
    #else
    // seems to be a bit faster
    for (int tid = threadIdx.x; tid < max_seq_len * size_per_head/x; tid += blockDim.x)
    {
      // could consider fast int division here
      if (tid % max_seq_len < step)
      {
        key_tgt_ptr[tid] = key_src_ptr[tid];
      }
    }
    #endif

    for (int tid = threadIdx.x; tid < step * size_per_head/x; tid += blockDim.x)
    {
      value_tgt_ptr[tid] = value_src_ptr[tid];
    }
}

template <typename T>
void update_KV_batch_major_cache_kernelLauncher(
        T* key_src_cache, T* key_tgt_cache,
        T* value_src_cache, T* value_tgt_cache, 
        const size_t* beam_ids, const uint8_t* isAllDone, 
        const int batch_size, const int beam_width, 
        const int head_num, const int size_per_head, 
        const int step, const int decoder_max_seq_len)
{
    dim3 grid(batch_size * beam_width, head_num);
    constexpr int block_sz = 128;

    update_KV_batch_major_cache_kernel<T><<<grid, block_sz, 0>>>(
        key_src_cache, key_tgt_cache,
        value_src_cache, value_tgt_cache,
        beam_ids, isAllDone, batch_size, beam_width, 
        size_per_head, step, decoder_max_seq_len);
}

template void update_KV_batch_major_cache_kernelLauncher(
        float* key_src_cache, float* key_tgt_cache,
        float* value_src_cache, float* value_tgt_cache,
        const size_t* beam_ids, const uint8_t* isAllDone, 
        const int batch_size, const int beam_width, 
        const int head_num, const int size_per_head, 
        const int step, const int decoder_max_seq_len);
template void update_KV_batch_major_cache_kernelLauncher(
        half* key_src_cache, half* key_tgt_cache,
        half* value_src_cache, half* value_tgt_cache,
        const size_t* beam_ids, const uint8_t* isAllDone, 
        const int batch_size, const int beam_width, 
        const int head_num, const int size_per_head, 
        const int step, const int decoder_max_seq_len);

void UpdateKVBatchMajorCacheOP(
        HUPtr<HUTensor> key_src_cache, HUPtr<HUTensor> key_tgt_cache, 
        HUPtr<HUTensor> value_src_cache, HUPtr<HUTensor> value_tgt_cache, 
        size_t* beams_ids, uint8_t* isAllDone, 
        const int batch_size, const int beam_width, 
        const int head_num, const int step)
{
    cudaSetDevice(key_src_cache->getDeviceId().no);
    /*
    size_t* d_beams_ids;
    CUDA_CHECK(cudaMalloc(&d_beams_ids, beams_ids.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_beams_ids, beams_ids.data(), 
                beams_ids.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    */

    const int hidden_units = key_src_cache->shape()[-1];
    const int decoder_max_seq_len = key_src_cache->shape()[-2];
    const int size_per_head = hidden_units / head_num;

    update_KV_batch_major_cache_kernelLauncher(
            key_src_cache->data(), key_tgt_cache->data(), 
            value_src_cache->data(), value_tgt_cache->data(), 
            beams_ids, isAllDone, batch_size, beam_width,
            head_num, size_per_head, step, decoder_max_seq_len);

    // CUDA_CHECK(cudaFree(d_beams_ids));
} 

/*
template <typename T>
__global__
void add_bias_input(T* output, const T* input, const T* bias, const int m, const int n)
{
  // This kernel can run with any block size and grid size
  // Since the hidden dimension of GPT-3 would be larger than 1024
  const int bid = blockIdx.x;
  const int blocks_per_row = n / blockDim.x;
  const int col_index = (bid % blocks_per_row) * blockDim.x + threadIdx.x;
  T bias_val = __ldg(&bias[col_index]);
  for(int index = bid * blockDim.x + threadIdx.x; index < m * n; index += blockDim.x * gridDim.x)
  {
    output[index] = output[index] + input[index] + bias_val;
  }
}

template<typename T>
void add_bias_input_kernelLauncher(T* output, const T* bias, const T* input, const int m, const int n)
{
  dim3 grid(min(m, 65536));
  dim3 block(min(n, 1024));

  add_bias_input<<<grid, block, 0>>>(output, input, bias, m, n);
}

template void add_bias_input_kernelLauncher<float>(
  float* output,
  const float* bias,
  const float* input,
  const int m,
  const int n);

void AddBiasInputOP(HUPtr<HUTensor> output, const HUPtr<HUTensor> bias, const HUPtr<HUTensor> input)
{
    const int m = output->shape()[-3];
    const int n = output->shape()[-1];
    add_bias_input_kernelLauncher(output->data(), bias->data(), input->data(), m, n);
} */

template <typename T> 
__global__
void gAddBiasInput(T* out, const T* bias, const T* input, size_t length, size_t cols)
{
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x)
  { 
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length)
    {
      // size_t index2 = index % cols;
      // out[index] += bias[index2];
      out[index] = out[index] + bias[index % cols] + input[index];
    }
  }
}

template <>
__global__
void gAddBiasInput(half* out, const half* bias, const half* input, size_t length, size_t cols)
{
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x)
  {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length)
    {
      // size_t index2 = index % cols;
      // out[index] += bias[index2];
      out[index] = out[index] + bias[index % cols] + input[index];
    }
  }
}

// template void gAddBiasInput<float>(float* out, const float* bias, const float* input, size_t length, size_t cols);
// template void gAddBiasInput<half>(half* out, const half* bias, const half* input, size_t length, size_t cols);

void AddBiasInputOP(HUPtr<HUTensor> output, const HUPtr<HUTensor> bias, const HUPtr<HUTensor> input)
{
    cudaSetDevice(output->getDeviceId().no);

    int length = output->shape().elements();
    int cols = bias->shape().elements();

    int threads = std::min(MAX_THREADS, length);
    int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    gAddBiasInput<TT_DATA_TYPE><<<blocks, threads>>>(output->data(), bias->data(), input->data(), length, cols);
    // gAddBiasInput<half><<<blocks, threads>>>(output->data(), bias->data(), input->data(), length, cols);
    // gAddBiasInput<T><<<blocks, threads>>>(output->data(), bias->data(), input->data(), length, cols);
#ifndef CUDA_DEBUG
    cudaStreamSynchronize(0);
#endif
} 

// template void AddBiasInputOP<float>(HUPtr<HUTensor> output, const HUPtr<HUTensor> bias, const HUPtr<HUTensor> input);
// template void AddBiasInputOP<half>(HUPtr<HUTensor> output, const HUPtr<HUTensor> bias, const HUPtr<HUTensor> input);

/*
template void update_KV_batch_major_cache_kernelLauncher(
        float* key_src_cache, float* key_tgt_cache,
        float* value_src_cache, float* value_tgt_cache,
        const size_t* beam_ids, const int batch_size,
        const int beam_width,  const int head_num,
        const int size_per_head, const int step,
        const int decoder_max_seq_len);
*/

/****************
template <typename T, int MAX_K>
void topK_softMax_kernelLauncher(const T* log_probs,
                                 const T* bias,
                                 float* cum_log_probs,
                                 int* ids,
                                 void* temp_storage,
                                 const int temp_storage_size,
                                 const int batch_size,
                                 const int beam_width,
                                 const int vocab_size,
                                 const int end_id,
                                 T diversity_rate)
{
    const int block_sz = (MAX_K < 16)? (MAX_K < 8)? SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE:128:64;
    //const int block_sz = SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE;

    assert(temp_storage_size % 2 == 0);
    assert(temp_storage_size >= 2 * batch_size * beam_width * beam_width);

    const int topk_buf_offset = ceil(batch_size * beam_width * beam_width / 4.) * 4;
    int* topk_tmp_id_buf = reinterpret_cast<int *>(temp_storage);
    T* topk_tmp_val_buf = reinterpret_cast<T *>(topk_tmp_id_buf + topk_buf_offset);
    float* tmp_buffer = reinterpret_cast<float *>(topk_tmp_val_buf + topk_buf_offset);

#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
    int voc_parts = 4;
    if (batch_size * beam_width < 256)
    {
        // Volta has 80 SMs, so we aim for three waves
        voc_parts = (240 + batch_size * beam_width - 1) / (batch_size * beam_width);
        voc_parts = std::min(128, voc_parts); // we implment up to 128
    }
    dim3 grid(batch_size * beam_width, voc_parts);
    cudaFuncSetAttribute(
            beam_online_softmax_topk_stage1_kernel<T, items_per_thread, MAX_K, block_sz>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxL1);
    beam_online_softmax_topk_stage1_kernel<T, items_per_thread, MAX_K, block_sz>
                            <<<grid, block_sz,0,stream>>>
                            (log_probs, bias, finished, tmp_buffer,
                            vocab_size, beam_width, end_id);
#endif
    if (beam_width > 1)
    {
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
        beam_online_softmax_topk_stage2_kernelLauncher<T, MAX_K>
                                (tmp_buffer, cum_log_probs, topk_tmp_id_buf, topk_tmp_val_buf,
                                    batch_size, beam_width, voc_parts, stream);
#else
        beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
                        <<<batch_size * beam_width, block_sz, 0, stream>>>
                                (log_probs, bias, cum_log_probs, finished, topk_tmp_id_buf,
                                topk_tmp_val_buf, vocab_size, beam_width, end_id);
#endif
#if 0
            // wrong result with diversity_rate != 0.f
            batch_topK_kernel<T, MAX_K, 32><<<batch_size, 32, 0, stream>>>
                                (topk_tmp_id_buf, topk_tmp_val_buf, ids, cum_log_probs);
#else
            batch_topk_kernel<T, MAX_K, 32><<<batch_size, 32, 0, stream>>>
                                (topk_tmp_id_buf, topk_tmp_val_buf,
                                ids, cum_log_probs, beam_width * beam_width, beam_width, diversity_rate);
#endif
    }
    else
    {
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
        beam_online_softmax_topk_stage2_kernelLauncher<float, MAX_K>
                                (tmp_buffer, cum_log_probs, ids, cum_log_probs,
                                batch_size, beam_width, voc_parts, stream);
#else
        beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
                            <<<batch_size * beam_width, block_sz, 0, stream>>>
                                    (log_probs, bias, cum_log_probs, finished, ids,
                                    cum_log_probs, vocab_size, beam_width, end_id);
#endif
    }
}

template <typename T>
void topK_softMax(const T* log_probs,
                  const T* bias,
                  float* cum_log_probs,
                  int* ids,
                  void* temp_storage,
                  DecodingBeamsearchArguments args)
{
    const int temp_storage_size = args.temp_storage_size_;
    const int batch_size = args.batch_size_;
    const int beam_width = args.beam_width_;
    const int vocab_size = args.vocab_size_padded_;
    const int end_id = args.end_id_;
    const T diversity_rate = args.beam_search_diversity_rate_;

    switch(beam_width)
    {
        case 1 :
            topK_softMax_kernelLauncher<T, 1>
                    (log_probs, bias, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 2 :
            topK_softMax_kernelLauncher<T, 2>
                    (log_probs, bias, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 3 :
            topK_softMax_kernelLauncher<T, 3>
                    (log_probs, bias, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 4 :
            topK_softMax_kernelLauncher<T, 4>
                    (log_probs, bias, cum_log_probs, ids, temp_storage, temp_storage_size,
                batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 8 :
            topK_softMax_kernelLauncher<T, 8>
                    (log_probs, bias, cum_log_probs, ids, temp_storage, temp_storage_size,
                batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 16 :
            topK_softMax_kernelLauncher<T, 16>
                    (log_probs, bias, cum_log_probs, ids, temp_storage, temp_storage_size,
                batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 32 :
            topK_softMax_kernelLauncher<T, 32>
                    (log_probs, bias, cum_log_probs, ids, temp_storage, temp_storage_size,
                batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        default :
            printf("[ERROR] Topk kernel does not support beamwidth = %d \n", beam_width);
            exit(0);
            break;
    }
}
******************/

template <typename T>
__global__ void embedding_lookup_sine_position_encoding_kernel(
        T* from_tensor, 
        const T* embedding_table, 
        const T* position_encoding, 
        const size_t* word_ids, 
        const int batch_size, 
        const int hidden_units, 
        bool isScale)
  {
      // 1. lookup from embedding table
      // 2. multiply hidden_dim**0.5
      // 3. add the position encoding
      T scale = (T)sqrtf(float(hidden_units));
      for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * hidden_units; index += blockDim.x * gridDim.x)
      {
        const int row_index = index / hidden_units;
        const int col_index = index % hidden_units;
        from_tensor[index] = isScale ? 
            embedding_table[word_ids[row_index] * hidden_units + col_index] * scale + position_encoding[col_index]
            : embedding_table[word_ids[row_index] * hidden_units + col_index] + position_encoding[col_index];
      }
  }

template <typename T> 
void embedding_lookup_sine_position_encoding_kernel_launcher(
        T* from_tensor, 
        const T* embedding_table, 
        const T* position_encoding, 
        const size_t* word_ids, 
        const int batch_size, 
        const int hidden_units,
        bool isScale)
  {
      dim3 grid(min(batch_size, 65536));
      dim3 block(min(hidden_units, 1024));
      embedding_lookup_sine_position_encoding_kernel<T><<<grid, block, 0>>>(from_tensor, embedding_table, 
              position_encoding, word_ids, batch_size, hidden_units, isScale);
  }

/*
template 
void embedding_lookup_sine_position_encoding_kernel_launcher(
        float* from_tensor, const float* embedding_table, const float* position_encoding, 
        const size_t* word_ids, const int batch_size, const int hidden_units, bool isScale);
*/


////////////////// for Decoder Lookup table //////////////
void EmbeddingLookUpPositionEncodingOP(HUPtr<HUTensor> &output, const HUPtr<HUTensor> word_emb, const HUPtr<HUTensor> pos_emb, const std::vector<size_t> &word_ids, const size_t startPos, bool isScale)
{
    const int batch_size = word_ids.size();
    const int hidden_units = word_emb->shape()[-1];

    // std::cout << "[77]" << std::endl;
    size_t* d_word_ids;
    cudaSetDevice(word_emb->getDeviceId().no);
    CUDA_CHECK(cudaMalloc(&d_word_ids, word_ids.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_word_ids, word_ids.data(), word_ids.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    // std::cout << "[88]" << std::endl;
    embedding_lookup_sine_position_encoding_kernel_launcher(
            output->data(), word_emb->data(), 
            pos_emb->data() + startPos * hidden_units, 
            d_word_ids, batch_size, hidden_units, isScale);
    // std::cout << "[99]" << std::endl;

    CUDA_CHECK(cudaFree(d_word_ids));
}

template <typename T> 
__global__ void start_id_embedding_position_lookups_kernel(T* from_tensor, 
                                                           const T* embedding_table,
                                                           const T* pos_table,
                                                           const size_t* word_ids, 
                                                           const int length, 
                                                           const int batch_size,
                                                           const int hidden_units, 
                                                           bool isScale)
{
    T scale = (T)sqrtf(float(hidden_units));
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * length * hidden_units; index += blockDim.x * gridDim.x)
    {
        const int word_index = index / hidden_units;
        const int step = word_index % length;
        const int col_index = index % hidden_units;

        from_tensor[index] = isScale ? 
            embedding_table[word_ids[word_index] * hidden_units + col_index] * scale + pos_table[step * hidden_units + col_index] : 
            embedding_table[word_ids[word_index] * hidden_units + col_index] + pos_table[step * hidden_units + col_index];
    }
  }

//////////// The Encoder Lookup Table ////////////
template <typename T> 
void start_id_embedding_position_lookups_kernel_launcher(T* from_tensor, 
                                                         const T* embedding_table,
                                                         const T* pos_table,
                                                         const size_t* word_ids,
                                                         const int length,
                                                         const int batch_size,
                                                         const int hidden_units, 
                                                         bool isScale)
{
    dim3 grid(min(batch_size * length, 65536));
    dim3 block(min(hidden_units, 1024));
    start_id_embedding_position_lookups_kernel<T><<<grid, block, 0>>>(
            from_tensor, embedding_table, pos_table, word_ids, 
            length, batch_size, hidden_units, isScale);
}

template 
void start_id_embedding_position_lookups_kernel_launcher(float* from_tensor, 
                                                         const float* embedding_table,
                                                         const float* pos_table,
                                                         const size_t* word_ids, 
                                                         const int length, 
                                                         const int batch_size,
                                                         const int hidden_units, 
                                                         bool isScale);
template
void start_id_embedding_position_lookups_kernel_launcher(half* from_tensor,
                                                         const half* embedding_table,
                                                         const half* pos_table,
                                                         const size_t* word_ids,
                                                         const int length,
                                                         const int batch_size,
                                                         const int hidden_units,
                                                         bool isScale);


void StartIdEmbeddingLookUpPositionEncodingOP(HUPtr<HUTensor> &output, const HUPtr<HUTensor> word_emb, const HUPtr<HUTensor> pos_emb, const std::vector<size_t> &word_ids, const int batch_size, bool isScale)
{
    const int length = word_ids.size() / batch_size;
    const int hidden_units = word_emb->shape()[-1];

    size_t* d_word_ids;
    cudaSetDevice(word_emb->getDeviceId().no);
    CUDA_CHECK(cudaMalloc(&d_word_ids, word_ids.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_word_ids, word_ids.data(), word_ids.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    start_id_embedding_position_lookups_kernel_launcher(
            output->data(), word_emb->data(), pos_emb->data(), d_word_ids, 
            length, batch_size, hidden_units, isScale);

    CUDA_CHECK(cudaFree(d_word_ids));
}

/////////////////////////// Self-Attention for Encoder /////////////////////////////////////////

template<typename T>
__global__
void add_QKV_bias(
        T* Q, const T* bias_Q, 
        T* K, const T* bias_K, 
        T* V, const T* bias_V, 
        T* q_buf_, T* k_buf_, T* v_buf_, 
        const int batch_size, const int seq_len, 
        const int head_num, const int size_per_head, const int word_per_block)
{

  T* data_ptr;
  T* buf_ptr;
  const T* bias_ptr;

  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

  int qkv_id = blockIdx.x * word_per_block / m;
  int row_offset = (blockIdx.x * word_per_block % m) * n;

  if(qkv_id == 0)
  {
    data_ptr = Q + row_offset;
    buf_ptr = q_buf_;
    bias_ptr = bias_Q;
  }
  else if(qkv_id == 1)
  {
    data_ptr = K + row_offset;
    buf_ptr = k_buf_;
    bias_ptr = bias_K;
  }
  else
  {
    data_ptr = V + row_offset;
    buf_ptr = v_buf_;
    bias_ptr = bias_V;
  }

  int batch_id = (blockIdx.x * word_per_block % m) / seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  int word_start_id = (blockIdx.x * word_per_block) % seq_len;

  T bias = __ldg(&bias_ptr[threadIdx.x]);

  for(int i = word_start_id; i < word_start_id + word_per_block; ++i)
  {
    T tmp = data_ptr[threadIdx.x] + bias;

    int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head +
      i * size_per_head + id_in_head;

    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
}

template<typename T>
__global__
void add_QKV_bias_generalized(
        const T* __restrict Q, const T* __restrict bias_Q, 
        const T* __restrict K, const T* __restrict bias_K, 
        const T* __restrict V, const T* __restrict bias_V, 
        T* q_buf_, T* k_buf_, T* v_buf_, 
        const int batch_size, const int seq_len, 
        const int head_num, const int size_per_head, const int word_per_block)
{

  const T* data_ptr;
  T* buf_ptr;
  T bias;

  int n = head_num * size_per_head;
  const int blocks_per_word = n / blockDim.x;
  const int blocks_per_buffer = gridDim.x / 3;
  const int qkv_id = blockIdx.x / blocks_per_buffer;
  const int block_id_in_buffer = blockIdx.x % blocks_per_buffer;
  const int offset = block_id_in_buffer * blockDim.x + threadIdx.x;
  const int bias_id = offset % n;

  if(qkv_id == 0)
  {
    data_ptr = Q + offset;
    buf_ptr = q_buf_;
    bias = __ldg(&bias_Q[bias_id]);
  }
  else if(qkv_id == 1)
  {
    data_ptr = K + offset;
    buf_ptr = k_buf_;
    bias = __ldg(&bias_K[bias_id]);
  }
  else
  {
    data_ptr = V + offset;
    buf_ptr = v_buf_;
    bias = __ldg(&bias_V[bias_id]);
  }

  const int head_id = bias_id / size_per_head;
  const int size_id = bias_id % size_per_head;

  for(int i = 0; i < word_per_block; i++)
  {
    const int block_lane = i * blocks_per_buffer;
    const int batch_id = (block_id_in_buffer + block_lane) / seq_len / blocks_per_word;
    const int word_id = ((block_id_in_buffer + block_lane) / blocks_per_word) % seq_len;

    int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head +
      word_id * size_per_head + size_id;
    buf_ptr[target_id] = __ldg(&data_ptr[block_lane * blockDim.x]) + bias;
  }
}

template <typename T>
void add_QKV_bias_transpose_kernelLauncher(
  T* q_buf, T* k_buf, T* v_buf,
  T* Q, const T* bias_Q,
  T* K, const T* bias_K,
  T* V, const T* bias_V,
  const int batch_size, const int seq_len,
  const int head_num, const int size_per_head)
{
  const int k = head_num * size_per_head;
  dim3 grid, block;
  if(k <= 1024)
  {
    if(sizeof(T) == 4)
    {
      const int m = batch_size * seq_len;
      const int word_per_block = 1;
      assert(k <= 1024);
      assert(m / word_per_block * 3 <= 65536);

      dim3 grid(m / word_per_block * 3);
      dim3 block(k);
      add_QKV_bias<T><<<grid, block, 0>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf, k_buf, v_buf,
        batch_size, seq_len, head_num, size_per_head, word_per_block);
    }
    else
    {
      const int word_per_block = 1;
      grid.x = batch_size * seq_len / word_per_block;
      block.x = head_num * size_per_head * word_per_block / 2;

      assert(block.x <= 1024);

      add_QKV_bias<T><<<grid, block, 0>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf, k_buf,
      v_buf, batch_size, seq_len, head_num, size_per_head / 2, word_per_block);
    }
  }
  else
  {
    // k > 1024, so split into many block
    if(sizeof(T) == 4)
    {
      const int m = batch_size * seq_len;
      const int word_per_block = 4;
      dim3 block;
      if(k % 512 == 0)
        block.x = 512;
      else if(k % 384 == 0)
        block.x = 384;
      else if(k % 256 == 0)
        block.x = 256;
      else if(k % 128 == 0)
        block.x = 128;
      else
        printf("[ERROR] no supported k %d \n", k);
      assert(k % block.x == 0);
      dim3 grid(m * k / block.x / word_per_block * 3);
      assert(grid.x <= 65536 && grid.x > 0);
      add_QKV_bias_generalized<T><<<grid, block, 0>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf, k_buf, v_buf,
        batch_size, seq_len, head_num, size_per_head, word_per_block);

    }
    else
    {
      const int m = batch_size * seq_len;
      const int word_per_block = 4;
      const int half_k = k / 2;
      dim3 block;
      if(half_k % 512 == 0)
        block.x = 512;
      else if(half_k % 384 == 0)
        block.x = 384;
      else if(half_k % 256 == 0)
        block.x = 256;
      else if(half_k % 128 == 0)
        block.x = 128;
      else if(half_k % 64 == 0)
        block.x = 64;
      else
        printf("[ERROR] no supported half_k %d \n", half_k);
      assert(half_k % block.x == 0);
      dim3 grid(m * half_k / block.x / word_per_block * 3);
      assert(grid.x <= 65536 && grid.x > 0);
      add_QKV_bias_generalized<half2><<<grid, block, 0>>>((const half2*)Q, (const half2*)bias_Q,
                                                          (const half2*)K, (const half2*)bias_K,
                                                          (const half2*)V, (const half2*)bias_V,
                                                          (half2*)q_buf, (half2*)k_buf, (half2*)v_buf,
                                                          batch_size, seq_len, head_num, size_per_head / 2, word_per_block);
    }
  }
}

//////  1. add_QKV_bias_transpose_kernelLauncher //////
template void add_QKV_bias_transpose_kernelLauncher(
        float* q_buf, float* k_buf, float* v_buf, 
        float* Q, const float* bias_Q, 
        float* K, const float* bias_K, 
        float* V, const float* bias_V, 
        const int batch_size, const int seq_len, 
        const int head_num, const int size_per_head);

template void add_QKV_bias_transpose_kernelLauncher(
        half* q_buf, half* k_buf, half* v_buf,
        half* Q, const half* bias_Q,
        half* K, const half* bias_K,
        half* V, const half* bias_V,
        const int batch_size, const int seq_len,
        const int head_num, const int size_per_head);

template<typename T>
__global__
void add_QKV_bias_rebuild_padding(
        T* Q, const T* bias_Q, 
        T* K, const T* bias_K, 
        T* V, const T* bias_V, 
        T* q_buf_, T* k_buf_, T* v_buf_, 
        const int batch_size, const int seq_len, 
        const int head_num, const int size_per_head, const int* mask_offset)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int bdim = blockDim.x;

  const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
  const int tgt_head_id = tid / size_per_head;
  const int tgt_hidden_id = tid % size_per_head;

  const int src_id = bid * bdim + tid;
  const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + \
                    tgt_head_id * seq_len * size_per_head + \
                    tgt_seq_id * size_per_head + \
                    tgt_hidden_id;

  q_buf_[tgt_id] = Q[src_id] + bias_Q[tid];
  k_buf_[tgt_id] = K[src_id] + bias_K[tid];
  v_buf_[tgt_id] = V[src_id] + bias_V[tid];
}

template<typename T>
void add_QKV_bias_rebuild_padding_kernelLauncher(
        T* Q, const T* bias_Q, 
        T* K, const T* bias_K, 
        T* V, const T* bias_V, 
        T* q_buf, T* k_buf, T* v_buf, 
        const int batch_size, const int seq_len, const int head_num, 
        const int size_per_head, const int valid_word_num, const int* mask_offset)
{
  const int k = head_num*size_per_head;

  if(std::is_same<T, float>::value)
  {
    add_QKV_bias_rebuild_padding<<<valid_word_num, k, 0>>>(
            Q, bias_Q, K, bias_K, V, bias_V, 
            q_buf, k_buf, v_buf, 
            batch_size, seq_len, 
            head_num, size_per_head, mask_offset);
  }
  // TODO [half]
  /*
  else
  {
    add_QKV_bias_rebuild_padding<<<valid_word_num, k / 2, 0, stream>>>((half2*)Q, (const half2*)bias_Q,
      (half2*)K, (const half2*)bias_K, (half2*)V, (const half2*)bias_V,
      (half2*)q_buf, (half2*)k_buf, (half2*)v_buf,
       batch_size, seq_len, head_num, size_per_head / 2, mask_offset);
  }
  */
}

//////  2. add_QKV_bias_rebuild_padding_kernelLauncher //////
template
void add_QKV_bias_rebuild_padding_kernelLauncher(float* Q, const float* bias_Q, float* K, const float* bias_K, float* V, const float* bias_V, float* q_buf, float* k_buf, float* v_buf, const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int valid_word_num, const int* mask_offset);

template <typename T>
__global__
void softmax_kernel(
        T* qk_buf_, const T* attr_mask, 
        const int batch_size, const int head_num, 
        const int seq_len, const T scalar)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;
    //// int mask_offset = batch_id * seq_len * seq_len;
    int mask_offset = batch_id * seq_len;   // modify

    __shared__ float s_sum, s_max;

    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
      //// float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : -100000000.f;  // modify

      //// mask_val = (1.0f - mask_val) * -10000.0f;

      float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val): -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-6f;
      }
      __syncthreads();

      if(threadIdx.x < seq_len)
        qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

      qk_offset += seq_len;
      //// mask_offset += seq_len;
    }
}

template <typename T>
__global__
void softmax_kernel_v2(
        T* qk_buf_, const T* attr_mask, 
        const int batch_size, const int head_num, 
        const int seq_len, const float scalar)
{
    int batch_id = blockIdx.x / head_num / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    //// int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;
    int mask_offset = batch_id * seq_len + seq_id;   // modify

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    //// float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : -100000000.f;

    //// mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

template <typename T>
__global__
void softmax_kernel_v3(
        T* qk_buf_, const T* attr_mask, 
        const int batch_size, const int head_num, 
        const int seq_len, const T scalar)
{

  bool qual = threadIdx.x < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    float tmp = -1e20f;
    int qk_offset;
    __shared__ float s_mean, s_max;
    if (qual){
      qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len + threadIdx.x;
      //// int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + threadIdx.x;

      //// int mask_offset = blockIdx.y * seq_len + threadIdx.x;
      /// int mask_offset = blockIdx.y * seq_len;
      /// int mask_offset = blockIdx.y * seq_len + seq_id + threadIdx.x;
      int mask_offset = blockIdx.y * seq_len + seq_id;
      /// int mask_offset = seq_id * seq_len;

      float qk = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

      // mask_val = (1.0f - mask_val) * -10000.0f;

      tmp = qk * static_cast<float>(scalar) + mask_val;
    }

    float max_val = blockReduceMax<float>(tmp);
    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    float qk_tmp = qual ? __expf(tmp - s_max) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual)
      qk_buf_[qk_offset] = (T)(qk_tmp * s_mean);
  }
}

template <typename T>
__global__
void softmax_kernel_v3_LE32(
        T* qk_buf_, const T* attr_mask, 
        const int batch_size, const int head_num, 
        const int seq_len, const T scalar)
{
  bool qual = threadIdx.x < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int qk_offset;
    __shared__ float s_mean, s_max;
    float tmp = -1e20f;
    if (qual){
      qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len + threadIdx.x;
      //// int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + threadIdx.x;
      //// int mask_offset = blockIdx.y * seq_len + threadIdx.x;
      /// int mask_offset = blockIdx.y * seq_len;
      //// int mask_offset = blockIdx.y * seq_len + seq_id + threadIdx.x;
      int mask_offset = blockIdx.y * seq_len + seq_id;

      float qk = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

      // mask_val = (1.0f - mask_val) * -10000.0f;

      tmp = static_cast<float>(qk) * static_cast<float>(scalar) + mask_val;
    }
    float max_val = warpReduceMax<float>(tmp);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    tmp = qual ? __expf(tmp - s_max) : 0.0f;
    float sum_val = warpReduceSum<float>(tmp);

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual)
      qk_buf_[qk_offset] = (T)(tmp * s_mean);
  }
}

template<typename T>
void attn_softmax_kernelLauncher(
  T* buffer,
  const T* attr_mask,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const T scalar)
{
  dim3 grid, block;

  //deal with odd seq_len
  if (seq_len % 2 != 0) {
    if(seq_len <= 32)
      block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
      block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
      block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
      block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
      block.x = 512;
    else
      block.x = 1024;

    if(batch_size * head_num <= 120)
    {
      // std::cout << "[softmax_kernel_v2]" << std::endl; 
      /*
      grid.x = batch_size * head_num * seq_len;
      softmax_kernel_v2<T><<<grid, block, 0>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
      */

      grid.x = batch_size * head_num;
      softmax_kernel<T><<<grid, block, 0>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);

    }
    else
    { 
      // std::cout << "[softmax_kernel]" << std::endl;
      grid.x = batch_size * head_num;
      softmax_kernel<T><<<grid, block, 0>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
    } 
  }
  //deal with even seq_len 
  else{
    grid.x = seq_len;
    if (batch_size * head_num > 360)
      grid.x = ceil(float(seq_len)/32.0f);
    grid.y = batch_size;
    grid.z = head_num;
    if (seq_len <= 32){
      // std::cout << "[softmax_kernel_v3_LE32]" << std::endl;
      block.x = 32;
      softmax_kernel_v3_LE32<T><<<grid, block, 0>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
    }
    else{
      if (sizeof(T) == 2){
        block.x = (seq_len/2 + 31)/32*32;
        softmax_kernel_v3<T><<<grid, block, 0>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
      }
      else{
        // std::cout << "[softmax_kernel_v3]" << std::endl;
        block.x = (seq_len + 31)/32*32;
        softmax_kernel_v3<T><<<grid, block, 0>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
      }
    }
    grid.x = grid.y = grid.z = 1;
  }

}

//////  3. attn_softmax_kernelLauncher //////

template void attn_softmax_kernelLauncher(
    float* buffer,
    const float* attr_mask,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const float scalar); 

template void attn_softmax_kernelLauncher(
    half* buffer,
    const half* attr_mask,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const half scalar); 


template<typename T>
__global__
void transpose(
        T* src, T* dst, 
        const int batch_size, const int seq_len, 
        const int head_num, const int size_per_head)
{
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
    + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template <typename T>
void transpose_kernelLauncher(
  T* dst, T* src,
  const int batch_size, const int seq_len,
  const int head_num, const int size_per_head)
{
  dim3 grid, block;
  if(sizeof(T) == 2)
  {
    const int seq_per_block = 4;
    grid.x = batch_size * head_num * seq_len / seq_per_block;
    block.x = seq_per_block * size_per_head / 2;

    assert(grid.x * seq_per_block == batch_size * head_num * seq_len);

    transpose<T><<<grid, block, 0>>>(src, dst, batch_size, seq_len, head_num, size_per_head / 2);
  }
  else
  {
    const int seq_per_block = 1;
    grid.x = batch_size * head_num * seq_len / seq_per_block;
    block.x = seq_per_block * size_per_head;
    transpose<T><<<grid, block, 0>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
  }
}

template<typename T>
__global__
void transpose_rebuild_padding(
        T* src, T* dst, 
        const int batch_size, const int seq_len, 
        const int head_num, const int size_per_head, const int* mask_offset)
{
  // TODO: optimize this kernel? 
  // do remove_sequence_length_padding
  const int tid = threadIdx.x; // batch * seq_len or valid_word_num
  const int bid = blockIdx.x; // head_num * size_per_head

  const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int src_seq_id = (bid + mask_offset[bid]) % seq_len;

  const int dst_seq_id = bid;

  const int head_id = tid / size_per_head;
  const int hidden_id = tid % size_per_head;
  dst[dst_seq_id * head_num * size_per_head + tid] = src[ src_batch_id * head_num * seq_len * size_per_head +
    head_id * seq_len * size_per_head + src_seq_id * size_per_head + hidden_id];
}

template<typename T>
void transpose_rebuild_padding_kernelLauncher(
        T* src, T* dst, 
        const int valid_word_num, const int batch_size, 
        const int seq_len, const int head_num, 
        const int size_per_head, const int* mask_offset)
{
  int k = head_num * size_per_head;
  if (std::is_same<T, float>::value)
  {
    transpose_rebuild_padding<<<valid_word_num, k, 0>>>(src, dst,
            batch_size, seq_len, head_num, size_per_head, mask_offset);
  }
  /*  TODO [half]
  else
  {
    transpose_rebuild_padding<half2><<<valid_word_num, k / 2, 0, stream>>>(
            (half2*)src, (half2*)dst,
            batch_size, seq_len, head_num, size_per_head / 2, mask_offset);
  } */
}

//////  3. transpose_rebuild_padding_kernelLauncher //////
template
void transpose_rebuild_padding_kernelLauncher(float* src, float* dst, const int valid_word_num,
                                              const int batch_size, const int seq_len,
                                              const int head_num, const int size_per_head,
                                              const int* mask_offset);

/* TODO [half]
template
void transpose_rebuild_padding_kernelLauncher(half* src, half* dst, const int valid_word_num,
                                              const int batch_size, const int seq_len,
                                              const int head_num, const int size_per_head,
                                              const int* mask_offset, cudaStream_t stream);
*/

/*
void multiHeadAttr_nofuse_kernelLauncher(
        float* Q, const float* bias_Q, 
        float* K, const float* bias_K, 
        float* V, const float* bias_V,
        const float* attr_mask, float* dst, 
        const int batch_size, const int seq_len, 
        const int head_num, const int size_per_head, const float scalar)
{
    const int k = head_num * size_per_head;
    if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
    {
        add_QKV_bias_transpose_kernelLauncher(
                q_buf_, k_buf_, v_buf_, 
                Q, bias_Q, K, bias_K, V, bias_V, 
                batch_size_, seq_len, head_num, size_per_head);
    }
    else
    {
        // if we use remove padding, then initialize the q_buf_, k_buf_ and v_buf_ to prevent bugs.
        cudaMemsetAsync(q_buf_, 0, 3 * batch_size_ * seq_len * head_num * size_per_head * sizeof(float), param_.stream);

        add_QKV_bias_rebuild_padding_kernelLauncher(
                Q, bias_Q, K, bias_K, V, bias_V, 
                q_buf_, k_buf_, v_buf_, 
                batch_size, seq_len, head_num, 
                size_per_head, param_.valid_word_num, param_.sequence_id_offset);
    }
    
    float alpha = 1.0f, beta = 0.0f;
    check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq_len, seq_len, size_per_head,
        &alpha,
        k_buf_, AType_, size_per_head, seq_len * size_per_head,
        q_buf_, BType_, size_per_head, seq_len * size_per_head,
        &beta,
        qk_buf_, CType_, seq_len, seq_len * seq_len,
        batch_size * head_num,
        computeType_,
        static_cast<cublasGemmAlgo_t>(cublasBmmAlgo_[0])));
    
    attn_softmax_kernelLauncher(qk_buf_, attr_mask, batch_size, seq_len, head_num, scalar);
    check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_per_head, seq_len, seq_len,
        &alpha,
        v_buf_, AType_, size_per_head, seq_len * size_per_head,
        qk_buf_, BType_, seq_len, seq_len * seq_len,
        &beta,
        transpose_dst_, CType_, size_per_head, seq_len * size_per_head,
        batch_size * head_num,
        computeType_,
        static_cast<cublasGemmAlgo_t>(cublasBmmAlgo_[1])));

    if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
    {
        transpose_kernelLauncher(transpose_dst_, dst, batch_size, seq_len, head_num, size_per_head);
    }
    else
    {
        transpose_rebuild_padding_kernelLauncher(transpose_dst_, dst, param_.valid_word_num,
                                                 batch_size, seq_len, head_num, size_per_head,
                                                 param_.sequence_id_offset);
    }
} */

void multiHeadAttr_nofuse_kernelLauncher(
        float* Q, const float* bias_Q, 
        float* K, const float* bias_K, 
        float* V, const float* bias_V,
        const float* att_mask, float* dst, 
        float* q_buf, float* k_buf, float* v_buf, 
        float* qk_buf, float* transpose_dst, 
        const int batch_size, const int seq_len, 
        const int head_num, const int size_per_head, const float scalar, 
        cublasHandle_t cublas_handle)
{
    const int k = head_num * size_per_head;
    /* 1. add_bias & transpose -> q_buf, k_buf, v_buf */
    add_QKV_bias_transpose_kernelLauncher(
                q_buf, k_buf, v_buf, 
                Q, bias_Q, K, bias_K, V, bias_V, 
                batch_size, seq_len, head_num, size_per_head);
    
    /* 2. caculate attention weights -> qk_buf */
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq_len, seq_len, size_per_head,
        &alpha,
        k_buf, CUDA_R_32F, size_per_head, seq_len * size_per_head,
        q_buf, CUDA_R_32F, size_per_head, seq_len * size_per_head,
        &beta,
        qk_buf, CUDA_R_32F, seq_len, seq_len * seq_len,
        batch_size * head_num,
        CUDA_R_32F,
        static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
    
    /* 3. softmax fuction */
    attn_softmax_kernelLauncher(qk_buf, att_mask, batch_size, seq_len, head_num, scalar);

    /* 4. weights * v */
    cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_per_head, seq_len, seq_len,
        &alpha,
        v_buf, CUDA_R_32F, size_per_head, seq_len * size_per_head,
        qk_buf, CUDA_R_32F, seq_len, seq_len * seq_len,
        &beta,
        transpose_dst, CUDA_R_32F, size_per_head, seq_len * size_per_head,
        batch_size * head_num,
        CUDA_R_32F,
        static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

    /* 5. transpose for final result -> transpose_dst_*/
    transpose_kernelLauncher(transpose_dst, dst, batch_size, seq_len, head_num, size_per_head);
}


void EncoderUnFusedSelfAttentionOP(
        HUPtr<HUTensor> q_tmp, const HUPtr<HUTensor> Q_bias, 
        HUPtr<HUTensor> k_tmp, const HUPtr<HUTensor> K_bias, 
        HUPtr<HUTensor> v_tmp, const HUPtr<HUTensor> V_bias, 
        HUPtr<HUTensor> att_mask, HUPtr<HUTensor> att_out, 
        HUPtr<HUTensor> q_buf, HUPtr<HUTensor> k_buf, HUPtr<HUTensor> v_buf, 
        HUPtr<HUTensor> qk_buf, HUPtr<HUTensor> att_out_transpose_buf, const int head_num, HUPtr<HUMemPool> mem)
{
    cudaSetDevice(q_tmp->getDeviceId().no);

    /*
    cudaDataType_t BType, AType, CType;
    cudaDataType_t computeType;
    int cublasAlgo;
    if (sizeof(TT_DATA_TYPE) == sizeof(half))  // fp16 
    {
        BType = CUDA_R_16F;
        AType = CUDA_R_16F;
        CType = CUDA_R_16F;
        computeType = CUDA_R_16F;
        cublasAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        // cublasAlgo = CUBLAS_GEMM_DEFAULT;
    }
    else  // fp32, and others data type
    {
        BType = CUDA_R_32F;
        AType = CUDA_R_32F;
        CType = CUDA_R_32F;
        computeType = CUDA_R_32F;
        cublasAlgo = CUBLAS_GEMM_DEFAULT;
    } */

    // auto cublas_handle = q_tmp->getDevice()->getCublasHandle();

    const int size_per_head = q_tmp->shape()[-1] / head_num;
    const int seq_len = q_tmp->shape()[-2];
    const int batch_size = q_tmp->shape()[-3];

    // float scalar = 1 / sqrtf(size_per_head * 1.0f);
    // TT_DATA_TYPE scalar = (TT_DATA_TYPE) 1/sqrtf(size_per_head * 1.0f);

    // module test
    const int k = head_num * size_per_head;
    /* 1. add_bias & transpose -> q_buf, k_buf, v_buf */
    add_QKV_bias_transpose_kernelLauncher(
                q_buf->data(), k_buf->data(), v_buf->data(),
                q_tmp->data(), Q_bias->data(), 
                k_tmp->data(), K_bias->data(), 
                v_tmp->data(), V_bias->data(), 
                batch_size, seq_len, head_num, size_per_head);
    // LOG(trace, "[TenTrans][HUMultiHeadAttention] qhs {}", q_buf->debug());
    // LOG(trace, "[TenTrans][HUMultiHeadAttention] khs {}", k_buf->debug());
    // LOG(trace, "[TenTrans][HUMultiHeadAttention] vhs {}", v_buf->debug());

    /* 2. caculate attention weights -> qk_buf */
    auto cublas_handle = q_tmp->getDevice()->getCublasHandle();
    TT_DATA_TYPE alpha = (TT_DATA_TYPE)1.0f, beta = (TT_DATA_TYPE)0.0f;

    cudaDataType_t BType, AType, CType;
    cudaDataType_t computeType;
    int cublasAlgo;
    if (sizeof(TT_DATA_TYPE) == sizeof(half))  // fp16 
    {
        BType = CUDA_R_16F;
        AType = CUDA_R_16F;
        CType = CUDA_R_16F;
        computeType = CUDA_R_16F;
        cublasAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        // cublasAlgo = CUBLAS_GEMM_DEFAULT;
    }
    else  // fp32, and others data type
    {
        BType = CUDA_R_32F;
        AType = CUDA_R_32F;
        CType = CUDA_R_32F;
        computeType = CUDA_R_32F;
        cublasAlgo = CUBLAS_GEMM_DEFAULT;
    } 

    cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq_len, seq_len, size_per_head,
        &alpha,
        k_buf->data(), BType, size_per_head, seq_len * size_per_head,
        q_buf->data(), AType, size_per_head, seq_len * size_per_head,
        &beta,
        qk_buf->data(), CType, seq_len, seq_len * seq_len,
        batch_size * head_num,
        computeType,
        static_cast<cublasGemmAlgo_t>(cublasAlgo));
    
    // ProdBatchedOP(qk_buf, q_buf, k_buf, mem, false, true, 0.f, scalar);

    //// LOG(trace, "[TenTrans][HUMultiHeadAttention] qk_buf {}", qk_buf->debug());

    /* 3. softmax fuction */
    TT_DATA_TYPE scalar = (TT_DATA_TYPE) (1 / sqrtf(size_per_head * 1.0f));
    attn_softmax_kernelLauncher(qk_buf->data(), att_mask->data(), batch_size, seq_len, head_num, scalar);

    //// LOG(trace, "[TenTrans][HUMultiHeadAttention] softmax {}", qk_buf->debug());

    /* 4. weights * v */
    cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_per_head, seq_len, seq_len,
        &alpha,
        v_buf->data(), BType, size_per_head, seq_len * size_per_head,
        qk_buf->data(), AType, seq_len, seq_len * seq_len,
        &beta,
        att_out_transpose_buf->data(), CType, size_per_head, seq_len * size_per_head,
        batch_size * head_num,
        computeType,
        static_cast<cublasGemmAlgo_t>(cublasAlgo));

    //// LOG(trace, "[TenTrans][HUMultiHeadAttention] att_out_transpose_buf {}", att_out_transpose_buf->debug());

    /* 5. transpose for final result -> transpose_dst_*/
    transpose_kernelLauncher(att_out->data(), att_out_transpose_buf->data(), batch_size, seq_len, head_num, size_per_head);  

    //// LOG(trace, "[TenTrans][HUMultiHeadAttention] att_out {}", att_out->debug());

    /*
    multiHeadAttr_nofuse_kernelLauncher(
            q_tmp->data(), Q_bias->data(), 
            k_tmp->data(), K_bias->data(), 
            v_tmp->data(), V_bias->data(), 
            att_mask->data(), att_out->data(), 
            q_buf->data(), k_buf->data(), v_buf->data(), 
            qk_buf->data(), att_out_transpose_buf->data(), 
            batch_size, seq_len, head_num, size_per_head, scalar, 
            cublas_handle);
    */

}

/*
template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_1_opt2_general(
    const T* __restrict log_probs,
    T* tmp_log_probs,
    int* topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    const int k,
    const int vocab_size
)
{
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_id = bid / BLOCKS_PER_BEAM; // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM; // block id for a beam 
    const int tmp_log_buf_index = row_id * vocab_size;
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM * k + block_lane * k;
    TopK_2<T> partial;

    for(int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM)
    {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index];
    }


    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM)
        {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -MAX_T_VAL;
        }
        __syncthreads();
    }
}

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_2_opt2_general(
    const int* __restrict topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    int* ids,
    const int k)
{
    const int size = k * k * BLOCKS_PER_BEAM;
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T *s_val = topk_tmp_val_buf + batch_id * size;
    int *s_id = (int*)(array);

    TopK_2<T> partial;

    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int i = tid; i < size; i+= BLOCK_SIZE)
        {
            partial.insert(s_val[i], i);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if(tid == 0)
        {
            s_id[ite] = total.p;
            s_val[total.p] = -MAX_T_VAL;
        }
        __syncthreads();
    }
    if(tid < k) ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
}

#define CASE_K_DIV(K,BLOCK_SIZE_1, BLOCK_SIZE_2) \
  case K: \
    beam_topK_kernel<T, K, BLOCK_SIZE_2><<<batch_size * beam_width, BLOCK_SIZE_2, 0, stream>>>(log_probs, \
        topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate); \
    if (K < 10) \
      batch_topK_kernel<T, K, BLOCK_SIZE_1><<<batch_size, BLOCK_SIZE_1, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids); \
    else \
      batch_topK_kernel_v2<T, K, 32><<<batch_size, 32, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids); \
  break; \

#define CASE_K(K, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_) \
  case K: \
    topk_stage_1_opt3<float, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_><<<batch_size * K * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>( \
        log_probs, \
        temp_log_probs, \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        finished, \
        beam_width, vocab_size, end_id); \
    topk_stage_2_opt3<float, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_><<<batch_size, BLOCK_SIZE_2_, K * sizeof(int), stream>>>( \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        ids, \
        beam_width); \
  break; \

void TopKOP(HUPtr<HUTensor> logProbs, HUPtr<HUTensor> topKIds, HUPtr<HUTensor> topKValues, const int K, const int batch_size, const int beam_size)
{
    const int vocab_size = logProbs->shape()[-1];

    const int max_block_per_beam = 8;
    int temp_log_probs_buf_size = batch_size * K * vocab_size; // type float
    int topk_tmp_ids_buf_size = batch_size * K * beam_width * max_block_per_beam;      // type int
    int topk_tmp_val_buf_size = batch_size * K * beam_width * max_block_per_beam;    

    // prevent memory misalinged address
    temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
    topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

    // tensor 
    if(diversity_rate == 0.0f)
    {
        switch(beam_width)
        {
            CASE_K(1,128,128,8);
            CASE_K(4,128,128,8);
            CASE_K(10,128,128,8);
            CASE_K(16,128,128,5);
            CASE_K(32,256,128,1);
            CASE_K(64,256,256,1);
            default:
                topk_stage_1_opt2_general<T, 128, 1><<<batch_size * beam_width * 1, 128, 0>>>(
                        log_probs->data(),
                        temp_log_probs->data(),
                        topk_tmp_id_buf->data(),
                        topk_tmp_val_buf->data(),
                        beam_width, vocab_size);
                topk_stage_2_opt2_general<T, 128, 1><<<batch_size, 128, 
                    beam_width*beam_width*1*sizeof(float) + beam_width * sizeof(int)>>>(
                        topk_tmp_id_buf->data(),
                        topk_tmp_val_buf->data(),
                        ids->data(),
                        beam_width);
            break;
        }
    }
    else
    {
        switch(beam_width)
        {
            CASE_K_DIV(1,256,256);
            CASE_K_DIV(4,256,256);
            CASE_K_DIV(16,256,64);
            CASE_K_DIV(64,256,64);
            default:
                printf("[ERROR] Topk kernel does not support beamwidth = %d \n", beam_width);
                exit(0);
            break;
        }
    }

} */

/*
void MaskedMultiHeadAttentionOP(
        const HUPtr<HUTensor> qkv_buf, const HUPtr<HUTensor> QKV_bias,
        HUPtr<HUTensor> key_cache, HUPtr<HUTensor> value_cache,
        HUPtr<HUTensor> context_buf, const std::vector<bool> &isAllDone,
        const int head_num, const int step);

void EncoderFusedQKVSelfAttentionOP(
        HUPtr<HUTensor> input, HUPtr<HUTensor> attr_mask, 
        const HUPtr<HUTensor> Q, const HUPtr<HUTensor> Q_bias, 
        const HUPtr<HUTensor> K, const HUPtr<HUTensor> K_bias, 
        const HUPtr<HUTensor> V, const HUPtr<HUTensor> V_bias, 
        HUPtr<HUTensor> att_out, int* sequence_id_offset, 
        int* trt_seqlen_offset, int* trt_seqlen_size)
{
    MultiHeadInitParam<float> multi_head_init_param;

}

void EncoderFusedQKVSelfAttention()
{
    int algoId = getAlgoIdFromMap(cublasAlgoMap_, 3, n, m, k, AType_ == CUDA_R_16F ? HALF_DATATYPE : FLOAT_DATATYPE); 
    check_cuda_error(cublasGemmBatchedEx(param_.cublas_handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         n, m, k,
                                         &alpha,
                                         (const void* const*) qkv_kernel_, AType_, n,
                                         (const void* const*) qkv_input_, BType_, k, 
                                         &beta,
                                         (void* const*)qkv_buf_, CType_, n, 
                                         3,
                                         computeType_,
                                         static_cast<cublasGemmAlgo_t>(algoId)));
    
    DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);
    multiHeadAttr_nofuse_kernelLauncher(param_.stream, 
                                        param_.cublas_handle, 
                                        param_.cublaslt_handle, 
                                        query_buf_, 
                                        param_.self_attention.query_weight.bias, 
                                        key_buf_, 
                                        param_.self_attention.key_weight.bias, 
                                        value_buf_, 
                                        param_.self_attention.value_weight.bias,
                                        param_.attr_mask,
                                        param_.attr_out,
                                        batch_size_,
                                        from_seq_len_,
                                        head_num_,
                                        size_per_head_,
                                        int8_mode_,
                                        scalar);
}
*/
template<typename T>
 __global__ 
void broadcast_kernel(T* out, 
                      T* log_probs, 
                      T* cum_log_probs,
                      const int vocab_size,
                      const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = tid / vocab_size;

    if(tid < N) {
        out[tid] = log_probs[tid] + cum_log_probs[bid];
    }
}


/*
void broadcast_kernelLauncher(float* out, 
                              float* log_probs, 
                              float* cum_log_probs, 
                              const int batch_beam_size, 
                              const int vocab_size)
{
    int N = batch_beam_size * vocab_size;
    dim3 block(1024);
    dim3 grid((N - 1) / block.x + 1);

    broadcast_kernel<float><<<grid, block, 0>>>(out, log_probs, cum_log_probs, vocab_size, N);
} */

template<typename T>
void broadcast_kernelLauncher(T* out, 
                              T* log_probs, 
                              T* cum_log_probs, 
                              const int batch_beam_size, 
                              const int vocab_size)
{
    int N = batch_beam_size * vocab_size;
    dim3 block(1024);
    dim3 grid((N - 1) / block.x + 1);

    broadcast_kernel<T><<<grid, block, 0>>>(out, log_probs, cum_log_probs, vocab_size, N);
} 

void BroadCastPlusOP(HUPtr<HUTensor> &out, HUPtr<HUTensor> log_probs, HUPtr<HUTensor> cum_log_probs)
{
    const int batch_beam_size = log_probs->shape()[-2];
    const int vocab_size = log_probs->shape()[-1];
    broadcast_kernelLauncher<TT_DATA_TYPE>(out->data(), log_probs->data(), cum_log_probs->data(), batch_beam_size, vocab_size);
}


template<typename T>
 __global__
void broadcast_2_kernel(T* out, 
                        T* log_probs,
                        T* cum_log_probs, 
                        const T* bias, 
                        const int vocab_size,
                        const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = tid / vocab_size;
    int col_id = tid % vocab_size;

    if(tid < N) {
        out[tid] = log_probs[tid] + bias[col_id] + cum_log_probs[bid];
    }
}

/*
void broadcast_2_kernelLauncher(float* out,
                                float* log_probs,
                                float* cum_log_probs,
                                const float* bias, 
                                const int batch_beam_size,
                                const int vocab_size)
{
    int N = batch_beam_size * vocab_size;
    dim3 block(1024);
    dim3 grid((N - 1) / block.x + 1);

    broadcast_2_kernel<float><<<grid, block, 0>>>(out, log_probs, cum_log_probs, bias, vocab_size, N);
} 
*/
template<typename T>
void broadcast_2_kernelLauncher(T* out,
                                T* log_probs,
                                T* cum_log_probs,
                                const T* bias, 
                                const int batch_beam_size,
                                const int vocab_size)
{
    int N = batch_beam_size * vocab_size;
    dim3 block(1024);
    dim3 grid((N - 1) / block.x + 1);

    broadcast_2_kernel<T><<<grid, block, 0>>>(out, log_probs, cum_log_probs, bias, vocab_size, N);
} 

void BroadCastPlusWithBiasOP(HUPtr<HUTensor> &out, HUPtr<HUTensor> log_probs, HUPtr<HUTensor> cum_log_probs, const HUPtr<HUTensor> bias)
{
    const int batch_beam_size = log_probs->shape()[-2];
    const int vocab_size = log_probs->shape()[-1];
    broadcast_2_kernelLauncher<TT_DATA_TYPE>(out->data(), log_probs->data(), cum_log_probs->data(), \ 
            bias->data(), batch_beam_size, vocab_size);
}


//////////////// TopK - version 1 ////////////////////////////
template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__
void beam_topK_kernel(const T* log_probs, 
                      int* topk_tmp_id_buf, 
                      T* topk_tmp_val_buf, 
                      const int vocab_size, 
                      T diversity_rate)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;

    for(int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }

    for(int elem_id = thread_id; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE)
    {
        int index = elem_id + block_id * vocab_size;
        partial.insert( (T)log_probs[index], index);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        int index = block_id * MAX_K;

        for(int i = 0; i < MAX_K; ++i)
        {
            topk_tmp_id_buf[index + i] = total.p[i];
            topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T)i;
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__
void batch_topK_kernel(int* topk_tmp_id_buf, 
                       T* topk_tmp_val_buf, 
                       int* id_buf)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;
    if (thread_id == 0)
    {
        for(int i = 0; i < MAX_K; ++i)
        {
            partial.p[i] = -1;
            partial.u[i] = -FLT_MAX;
        }

        int index = block_id * MAX_K * MAX_K;
        for(int i = 0; i < MAX_K * MAX_K; i++)
        {
            partial.insert( (T)topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i]);
        }

        index = block_id * MAX_K;
        for(int i = 0; i < MAX_K; i++)
        {
            id_buf[index + i] = partial.p[i];
        }
    }
}

template <typename T>
void topK_kernelLauncher(T* log_probs, 
                         int* topk_tmp_id_buf, 
                         T* topk_tmp_val_buf, 
                         int* ids, 
                         const int batch_size,
                         const int beam_width,
                         const int vocab_size)
{
    // const int batch_size = args.batch_size_;
    // const int beam_width = args.beam_width_;
    // const int vocab_size = args.vocab_size_padded_;
    // const int diversity_rate = args.beam_search_diversity_rate_;
    const int diversity_rate = 0;
    const int block_size = SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE;

    switch(beam_width)
    {
        case 1 :
            beam_topK_kernel<T, 1, block_size><<<batch_size * beam_width, block_size, 0>>>(log_probs, 
                    topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate);
            batch_topK_kernel<T, 1, block_size><<<batch_size, block_size, 0>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);
            break;

        case 2 :
            beam_topK_kernel<T, 2, block_size><<<batch_size * beam_width, block_size, 0>>>(log_probs, 
                    topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate);
            batch_topK_kernel<T, 2, block_size><<<batch_size, block_size, 0>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);
            break;

        case 3 :
            beam_topK_kernel<T, 3, block_size><<<batch_size * beam_width, block_size, 0>>>(log_probs, 
                    topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate);
            batch_topK_kernel<T, 3, block_size><<<batch_size, block_size, 0>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);
            break;

        case 4 :
            beam_topK_kernel<T, 4, block_size><<<batch_size * beam_width, block_size, 0>>>(log_probs, 
                    topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate);
            batch_topK_kernel<T, 4, block_size><<<batch_size, block_size, 0>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);
            break;

        case 6 :
            beam_topK_kernel<T, 6, block_size><<<batch_size * beam_width, block_size, 0>>>(log_probs, 
                    topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate);
            batch_topK_kernel<T, 6, block_size><<<batch_size, block_size, 0>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);
            break;

        case 8 :
            beam_topK_kernel<T, 8, block_size><<<batch_size * beam_width, block_size, 0>>>(log_probs, 
                    topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate);
            batch_topK_kernel<T, 8, block_size><<<batch_size, block_size, 0>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);
            break;

        case 32 :
            beam_topK_kernel<T, 32, block_size><<<batch_size * beam_width, block_size, 0>>>(log_probs, 
                    topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate);
            batch_topK_kernel<T, 32, block_size><<<batch_size, block_size, 0>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids);
            break;

        default:
            printf("[ERROR] Topk kernel does not support beamwidth = %d \n", beam_width);
            exit(0);
            break;
    }
}

template void topK_kernelLauncher<float>(float* log_probs, 
                                         int* topk_tmp_id_buf,
                                         float* topk_tmp_val_buf, 
                                         int* ids, 
                                         const int batch_size, 
                                         const int beam_width, 
                                         const int vocab_size);

void TopKOP(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int vocab_size)
{
    // const int batch_size = log_probs->shape()[-2] / beam_size;
    // const int vocab_size = log_probs->shape()[-1];
    const int batch_size = log_probs->shape()[-2];
    const int beam_size = log_probs->shape()[-1] / vocab_size;
    const int K = topKIds.size() / batch_size ;
    // const int vocab_size = log_probs->shape()[-1];
    std::cout << "batch_size: " << batch_size << "\tbeam_size: "<< beam_size << "\tvocab size: " << vocab_size << std::endl;

    cudaSetDevice(log_probs->getDeviceId().no);
    int *ids, *topk_tmp_id_buf;
    CUDA_CHECK(cudaMalloc(&ids, topKIds.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&topk_tmp_id_buf, topKIds.size() * sizeof(int)));
    TT_DATA_TYPE *topk_tmp_val_buf;
    CUDA_CHECK(cudaMalloc(&topk_tmp_val_buf, topKValues.size() * sizeof(TT_DATA_TYPE)));

    topK_kernelLauncher<TT_DATA_TYPE>(log_probs->data(), topk_tmp_id_buf, 
                                      topk_tmp_val_buf, ids, 
                                      batch_size, K, vocab_size);

    // Device_data to Host_data
    CUDA_CHECK(cudaMemcpy(topKIds.data(), ids, topKIds.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(topKValues.data(), topk_tmp_val_buf, topKIds.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(ids));
    CUDA_CHECK(cudaFree(topk_tmp_id_buf));
    CUDA_CHECK(cudaFree(topk_tmp_val_buf));
}

void TopKOP_V2(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int K, const int vocab_size, void* tmp_storage)
{
    /*
    const int batch_size = log_probs->size() / vocab_size;
    const int temp_size = batch_size * K * MAX_BLOCKS_PER_BEAM;
    */

    const int batch_size = log_probs->shape()[-2];
    const int depth = log_probs->shape()[-1];
    const int temp_size = batch_size * K * MAX_BLOCKS_PER_BEAM;

    int* topk_tmp_id_buf = (int*)tmp_storage;
    TT_DATA_TYPE* topk_tmp_val_buf = (TT_DATA_TYPE*)(topk_tmp_id_buf + temp_size);
    int* topk_id_buf = (int*)(topk_tmp_val_buf + temp_size);
    TT_DATA_TYPE* topk_val_buf = (TT_DATA_TYPE*)(topk_id_buf + topKIds.size());
    /*
    cudaSetDevice(log_probs->getDeviceId().no);
    int* topk_tmp_id_buf;
    CUDA_CHECK(cudaMalloc(&topk_tmp_id_buf, temp_size * sizeof(int)));
    float* topk_tmp_val_buf;
    CUDA_CHECK(cudaMalloc(&topk_tmp_val_buf, temp_size * sizeof(float)));

    int* topk_id_buf;
    CUDA_CHECK(cudaMalloc(&topk_id_buf, topKIds.size() * sizeof(int)));
    float* topk_val_buf;
    CUDA_CHECK(cudaMalloc(&topk_val_buf, topKValues.size() * sizeof(float)));
    */


    fastertransformer::topK_kernelLauncher<TT_DATA_TYPE>(log_probs->data(), 
                                                         topk_tmp_id_buf, topk_tmp_val_buf,
                                                         topk_id_buf, topk_val_buf,
                                                         batch_size, 1, K, /*vocab_size*/ depth);

    CUDA_CHECK(cudaMemcpy(topKIds.data(), topk_id_buf, topKIds.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(topKValues.data(), topk_val_buf, topKValues.size() * sizeof(float), cudaMemcpyDeviceToHost));
    /*
    CUDA_CHECK(cudaFree(topk_id_buf));
    CUDA_CHECK(cudaFree(topk_val_buf));

    CUDA_CHECK(cudaFree(topk_tmp_id_buf));
    CUDA_CHECK(cudaFree(topk_tmp_val_buf));
    */
}



struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.m > b.m); 
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res; 
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template<typename T, int MAX_K>
struct TopKMD
{
    MD md;
    TopK<T, MAX_K> topk;
};

template<typename T, int MAX_K>
__device__ __forceinline__ TopKMD<T, MAX_K> reduce_topk_md_op(const TopKMD<T, MAX_K>& a, const TopKMD<T, MAX_K>& b)
{
    TopKMD<T, MAX_K> res;
    res.md = reduce_md_op(a.md, b.md);
    res.topk = reduce_topk_op(a.topk, b.topk);
    return res;
}

#define TOPK_FP16_STORAGE 0
template<typename T, int ITEMS_PER_THREAD, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1)
__global__ void beam_online_softmax_topk_stage1_kernel(
    const T * __restrict x,                  // log_probs:  [batch_size*beam_width, vocab_size]
    const T * __restrict b,                  // bias:       [vocab_size]
    const uint8_t  * __restrict finished,    // finished:   [batch_size]
    float * __restrict t,                    // tmp_buf:    []
    int beam_width,                          // - beam_width
    int V,                                   // - vocab_size
    int E)                                   // - end ID
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    const int PACKED_TOP_KMD_SIZE = 2 * MAX_K + 2;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;

    // one will have multiple sections per V
    const int v_local = (V + gridDim.y - 1) / gridDim.y;
    const int section_start = v_local * blockIdx.y;
    int section_end = section_start + v_local;
    section_end = (section_end > V)? V : section_end;

    // reposition x to data for the current vector
    x += vector_id * V;
#if TOPK_FP16_STORAGE == 1
    typedef cub::BlockReduce<TopKMD<__half, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
#else
    typedef cub::BlockReduce<TopKMD<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
#endif
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float buf_s[PACKED_TOP_KMD_SIZE]; // save intermediate result

#if TOPK_FP16_STORAGE == 1
    TopKMD<__half, MAX_K> partial;
#else
    TopKMD<T, MAX_K> partial;
#endif
    //// bool finish = finished[vector_id];
    bool finish = finished[vector_id / beam_width];
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;
 
    if (finish)
    {
        #pragma unroll 1
        for(int elem_id = section_start + thread_id; elem_id < section_end; elem_id += THREADBLOCK_SIZE)
        {
            float elem = (elem_id == E) ? MAX_T_VAL : -MAX_T_VAL;
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }
    else
    {
        #pragma unroll 1
        for(int elem_id = section_start + thread_id; elem_id < section_end; elem_id += THREADBLOCK_SIZE)
        {
            T bias = b == nullptr ? (T)0.0f : b[elem_id]; // gpt-2 does not use bias
            T elem = x[elem_id] + bias;
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }

#if TOPK_FP16_STORAGE == 1
    TopKMD<__half, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<__half, MAX_K>);
#else
    TopKMD<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<T, MAX_K>);
#endif

    if (thread_id == 0)
    {
        for (int i = 0; i < MAX_K; i++)
        {
            reinterpret_cast<int *>(buf_s)[i] = total.topk.p[i] + vector_id * V; // faster transformer needs absolute id
            buf_s[MAX_K + i] = total.topk.u[i];
        }
        buf_s[2 * MAX_K] = total.md.d;
        buf_s[2 * MAX_K + 1] = total.md.m;
    }
    __syncthreads();
    if (threadIdx.x < PACKED_TOP_KMD_SIZE)
    {
        t[blockIdx.x * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE + threadIdx.x] = buf_s[threadIdx.x];
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void beam_online_softmax_topk_stage2_kernel(
    const float * __restrict x,
    const float * __restrict c,
    int * __restrict z,
    T * __restrict v, 
    int parts_per_beam)
{
    const int vector_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int PACKED_TOP_KMD_SIZE = 2 * MAX_K + 2;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;

    extern __shared__ char buf_s_[]; // intermediate result
    float * buf_s = reinterpret_cast<float *>(buf_s_);
    //__shared__ float buf_s[PACKED_TOP_KMD_SIZE * THREADBLOCK_SIZE]; // intermediate result

    typedef cub::BlockReduce<TopKMD<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    x += vector_id * PACKED_TOP_KMD_SIZE * parts_per_beam;

    TopKMD<T, MAX_K> partial; 
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;

    // load and unpack into registers through smem
    for (int idx = thread_id; idx < PACKED_TOP_KMD_SIZE * parts_per_beam; idx += THREADBLOCK_SIZE)
    {
        buf_s[idx] = x[idx];
    }
    __syncthreads();

    if (threadIdx.x < parts_per_beam)
    {
        float * b_s = buf_s + thread_id * PACKED_TOP_KMD_SIZE;
        for (int i = 0; i < MAX_K; i++)
        {
            partial.topk.p[i] = reinterpret_cast<int *>(b_s)[i];
            partial.topk.u[i] = b_s[MAX_K + i];
        }
        partial.md.d = b_s[2 * MAX_K]; 
        partial.md.m = b_s[2 * MAX_K + 1];
    }
    __syncthreads();

    TopKMD<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<T, MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * MAX_K;
        v += vector_id * MAX_K;
        c += vector_id;

        float d_total_log = logf(total.md.d);
        for(int i = 0; i < MAX_K; ++i)
        {
            float val = (float)total.topk.u[i] - total.md.m - d_total_log;
            if (i < MAX_K)
            {
                z[i] = total.topk.p[i];
                v[i] = (float)val + (float)c[0];
            }
        }
    }
}

template<typename T, int MAX_K>
void beam_online_softmax_topk_stage2_kernelLauncher(
    const float * temp_storage,
    const float * cum_log_probs,
    int * ids,
    T * vals, 
    int batch_size,
    int beam_width,
    int parts_per_beam)
{
    // might rewrite beam_online_softmax_topk_stage2_kernel no to depend on constant block size
    // in oreder to reduce compilation time
    int smem_stage2_size = parts_per_beam * (2 * MAX_K + 2) * sizeof(float);

    /*
    if (parts_per_beam <= 4)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 4>
        <<<batch_size * beam_width, 4, smem_stage2_size>>>
                (temp_storage, cum_log_probs, ids, vals, parts_per_beam);
        return;
    }
    if (parts_per_beam <= 8)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 8>
        <<<batch_size * beam_width, 8, smem_stage2_size>>>
                (temp_storage, cum_log_probs, ids, vals, parts_per_beam);
        return;
    }
    if (parts_per_beam <= 16)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 16>
        <<<batch_size * beam_width, 16, smem_stage2_size>>>
                (temp_storage, cum_log_probs, ids, vals, parts_per_beam);
        return;
    } */ 
    if (parts_per_beam <= 32)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 32>
        <<<batch_size * beam_width, 32, smem_stage2_size>>>
                (temp_storage, cum_log_probs, ids, vals, parts_per_beam);
        return;
    }
    if (parts_per_beam <= 64)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 64>
        <<<batch_size * beam_width, 64, smem_stage2_size>>>
                (temp_storage, cum_log_probs, ids, vals, parts_per_beam);
        return;
    }
    if (parts_per_beam <= 128)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 128>
        <<<batch_size * beam_width, 128, smem_stage2_size>>>
                (temp_storage, cum_log_probs, ids, vals, parts_per_beam);
        return;
    }
    assert(0);
}

template<typename T, int ITEMS_PER_THREAD, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void beam_online_softmax_topk_kernel(
    const T * __restrict x,                     // log_probs:     [batch_size * beam_width, vocab_size]
    const T * __restrict b,                     // bias:          [vocab_size]
    const float * __restrict c,                 // cum_log_probs: [batch_size * beam_width * MAX_K] 
    const uint8_t  * __restrict finished,       // finished:      [batch_size]
    int * __restrict z,                         // tmp_ids:       [batch_size * beam_width * MAX_K]
    T * __restrict v,                           // tmp_vals:      [batch_size * beam_width * MAX_K]
    int beam_width,                             // - beam_width
    int V,                                      // - vocab_size
    int E)                                      // - end ID
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;

    // reposition y to data for the current vector
    x += vector_id * V;

    typedef cub::BlockReduce<TopKMD<float, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopKMD<float, MAX_K> partial;
    bool finish = finished[vector_id / beam_width];
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;

    if (finish)
    {
        for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        {
            float elem = (elem_id == E) ? MAX_T_VAL : -MAX_T_VAL;
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }
    else
    {
        for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        {
            float elem = x[elem_id] + b[elem_id];
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }

    TopKMD<float, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<float, MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * MAX_K;
        v += vector_id * MAX_K;
        c += vector_id;

        float d_total_log = logf(total.md.d);
        for(int i = 0; i < MAX_K; ++i)
        {
            float val = total.topk.u[i] - total.md.m - d_total_log;
            if (i < MAX_K)
            {
                z[i] = total.topk.p[i] + vector_id * V; // faster transformer needs absolute id
                v[i] = val + c[0];
            }
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void batch_topk_kernel(
    const int * __restrict x,     // tmp_ids:      [batch_size*beam_size, MAX_K] 
    const T * __restrict y,       // tmp_values:   [batch_size*beam_size, MAX_K]
    int * __restrict z,           // ids:          [batch_size, MAX_K]
    float * __restrict v,         // values:       [batch_size, MAX_K]
    int V,                        // - vocab_size
    T diversity_rate)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition x, y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopK<T, MAX_K> partial;
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        int i = elem_id % MAX_K;
        T elem = y[elem_id] + diversity_rate * (T) i;
        int elem_idx = elem_id; //x[elem_id];
        partial.insert(elem, elem_idx);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * MAX_K;
        v += vector_id * MAX_K;

        for(int i = 0; i < MAX_K; ++i)
        {
            if (i < MAX_K)
            {
                z[i] = x[total.p[i]];
                v[i] = (float)y[total.p[i]];
            }
        }
    }
}

template <typename T, int MAX_K>
void topK_softMax_kernelLauncher(const T* log_probs,          // [batch_size * beam_size, vocab_size]
                                 const T* bias,               // [vocab_size]
                                 const uint8_t* finished,     // [batch_size]
                                 float* cum_log_probs,        // [batch_size, MAX_K]
                                 int* ids,                    // [batch_size, MAX_K]
                                 void* temp_storage,
                                 const int temp_storage_size, 
                                 const int batch_size,
                                 const int beam_width,
                                 const int vocab_size,
                                 const int end_id,
                                 T diversity_rate)
{
    // std::cout << "beam_width:" << beam_width << "\tbatch_size:" << batch_size << "\tMAX_K:" << MAX_K << std::endl;
    const int items_per_thread = 1;
    const int block_sz = (MAX_K < 16)? (MAX_K < 8)? SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE:128:64;
    //// const int block_sz = SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE;

    assert(temp_storage_size % 2 == 0);
    assert(temp_storage_size >= 2 * batch_size * beam_width * MAX_K);
    //// assert(temp_storage_size >= 2 * batch_size * MAX_K/2 * MAX_K);

    const int topk_buf_offset = (int)(ceil(batch_size * beam_width * MAX_K / 4.) * 4);
    //// const int topk_buf_offset = (int)(ceil(batch_size * MAX_K/2 * MAX_K / 4.) * 4);
    int* topk_tmp_id_buf = reinterpret_cast<int *>(temp_storage);
    T* topk_tmp_val_buf = reinterpret_cast<T *>(topk_tmp_id_buf + topk_buf_offset);
    // T* topk_tmp_val_buf = reinterpret_cast<T *>(topk_tmp_id_buf + topk_buf_offset);

    if (batch_size * beam_width < 256)
    {
        // std::cout << "Small TopKSoftmax" << std::endl;
        float* tmp_buffer = reinterpret_cast<float *>(topk_tmp_val_buf + topk_buf_offset);

        // Volta has 80 SMs, so we aim for three waves
        int voc_parts = (240 + batch_size * beam_width - 1) / (batch_size * beam_width);
        voc_parts = std::min(128, voc_parts); // we implment up to 128
        //// voc_parts = 4;

        dim3 grid(batch_size * beam_width, voc_parts);
        cudaFuncSetAttribute(
                beam_online_softmax_topk_stage1_kernel<T, items_per_thread, MAX_K, block_sz>, 
                cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
        
        beam_online_softmax_topk_stage1_kernel<T, items_per_thread, MAX_K, block_sz> 
                <<<grid, block_sz, 0>>>
                (log_probs, bias, finished, tmp_buffer, beam_width, vocab_size, end_id);        

        if (beam_width > 1)
        {
            beam_online_softmax_topk_stage2_kernelLauncher<T, MAX_K> 
                (tmp_buffer, cum_log_probs, topk_tmp_id_buf, topk_tmp_val_buf, 
                 batch_size, beam_width, voc_parts);

            batch_topk_kernel<T, MAX_K, 32><<<batch_size, 32, 0>>>
                (topk_tmp_id_buf, topk_tmp_val_buf, ids, cum_log_probs, 
                 beam_width * MAX_K, diversity_rate);
        }
        else
        {
            beam_online_softmax_topk_stage2_kernelLauncher<float, MAX_K> 
                (tmp_buffer, cum_log_probs, ids, cum_log_probs, 
                 batch_size, beam_width, voc_parts);
        }
    }
    else
    {
        // std::cout << "normal TopKSoftmax" << std::endl;
        if (beam_width > 1)
        {
            beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz> 
                <<<batch_size * beam_width, block_sz, 0>>> 
                (log_probs, bias, cum_log_probs, finished, topk_tmp_id_buf, topk_tmp_val_buf, 
                 beam_width, vocab_size, end_id);

            batch_topk_kernel<T, MAX_K, 32><<<batch_size, 32, 0>>> 
                (topk_tmp_id_buf, topk_tmp_val_buf, ids, cum_log_probs, 
                 beam_width * MAX_K, diversity_rate);
        }
        else
        {
            /*
            beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz> 
                <<<batch_size * beam_width, block_sz, 0>>>
                (log_probs, bias, cum_log_probs, finished, ids, cum_log_probs, 
                 beam_width, vocab_size, end_id);
             */
        }
    }
    

    /*
    std::cout << "SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE:" << SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE
        << "\tSMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS:" << SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS
        << "\tMAX_K:" << MAX_K << std::endl;

    std::cout << "items_per_thread:" << items_per_thread << "\tblock_sz:" << block_sz << "\tbatch_size:"
        << batch_size << "\tbeam_width:" << beam_width << "\tvocab_size:" << vocab_size << "\tend_id:" << end_id << std::endl;
    */

    //// std::cout << "test1" << std::endl;

/*
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
    int voc_parts = 4;
    if (batch_size * beam_width < 256)
    {  
        // Volta has 80 SMs, so we aim for three waves
        voc_parts = (240 + batch_size * beam_width - 1) / (batch_size * beam_width);
        voc_parts = std::min(128, voc_parts); // we implment up to 128
    }
    //// std::cout << "test2" << std::endl;
    dim3 grid(batch_size * beam_width, voc_parts);
    cudaFuncSetAttribute(
            beam_online_softmax_topk_stage1_kernel<T, items_per_thread, MAX_K, block_sz>, 
            cudaFuncAttributePreferredSharedMemoryCarveout, 
            cudaSharedmemCarveoutMaxL1);
    beam_online_softmax_topk_stage1_kernel<T, items_per_thread, MAX_K, block_sz>
                            <<<grid, block_sz, 0>>>
                            (log_probs, bias, finished, tmp_buffer,
                             beam_width, vocab_size, end_id);
    //// std::cout << "test3" << std::endl;
#endif

    if (beam_width > 1)
    {
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
        //// std::cout << "test4" << std::endl;
        beam_online_softmax_topk_stage2_kernelLauncher<T, MAX_K>
            (tmp_buffer, cum_log_probs, topk_tmp_id_buf, topk_tmp_val_buf, 
             batch_size, beam_width, voc_parts);
        //// std::cout << "test5" << std::endl;
#else 
        beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
            <<<batch_size * beam_width, block_sz, 0>>>
                (log_probs, bias, cum_log_probs, finished, topk_tmp_id_buf, 
                 topk_tmp_val_buf, beam_width, vocab_size, end_id);
#endif
        //// std::cout << "test6" << std::endl;
        batch_topk_kernel<T, MAX_K, 32><<<batch_size, 32, 0>>> 
                (topk_tmp_id_buf, topk_tmp_val_buf, ids, 
                 cum_log_probs, beam_width * MAX_K, diversity_rate);
        //// std::cout << "test7" << std::endl;
    }
    else
    {
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
        //// std::cout << "test8" << std::endl;
        beam_online_softmax_topk_stage2_kernelLauncher<float, MAX_K>
            (tmp_buffer, cum_log_probs, ids, cum_log_probs, 
             batch_size, beam_width, voc_parts);
        //// std::cout << "test9" << std::endl;

#else
        //// std::cout << "test10" << std::endl;
        beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
            <<<batch_size * beam_width, block_sz, 0>>>
            (log_probs, bias, cum_log_probs, finished, ids, 
             cum_log_probs, beam_width, vocab_size, end_id);
        //// std::cout << "test11" << std::endl;
#endif
    } */
}

template <typename T>
void topK_softMax(const T* log_probs,
                  const T* bias,
                  const uint8_t* finished,
                  float* cum_log_probs,
                  int* ids,
                  const int K,
                  void* temp_storage,
                  const int temp_storage_size,
                  const int batch_size,
                  const int beam_width,
                  const int vocab_size,
                  const int end_id,
                  const T diversity_rate)
{
    switch(beam_width)
    {
        case 1 :
            switch(K)
            {
                case 1:
                    topK_softMax_kernelLauncher<T, 1>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 2:
                    topK_softMax_kernelLauncher<T, 2>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 4:
                    topK_softMax_kernelLauncher<T, 4>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 8:
                    topK_softMax_kernelLauncher<T, 8>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 10:
                    topK_softMax_kernelLauncher<T, 10>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 16:
                    topK_softMax_kernelLauncher<T, 16>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 32:
                    topK_softMax_kernelLauncher<T, 32>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                default :
                    topK_softMax_kernelLauncher<T, 32>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;
            }
            break;
    
        case 2 :
            topK_softMax_kernelLauncher<T, 4>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 3 :
            topK_softMax_kernelLauncher<T, 6>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 4 :
            topK_softMax_kernelLauncher<T, 8>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 5 :
            topK_softMax_kernelLauncher<T, 10>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 8 :
            topK_softMax_kernelLauncher<T, 16>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 16 :
            topK_softMax_kernelLauncher<T, 32>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 32 :
            topK_softMax_kernelLauncher<T, 64>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        default :
            printf("[ERROR] Topk kernel does not support beamwidth = %d \n", beam_width);
            exit(0);
            break;
    }
}

template void topK_softMax<float>(const float* log_probs,
                                  const float* bias,
                                  const uint8_t* finished,
                                  float* cum_log_probs,
                                  int* ids,
                                  const int K,
                                  void* tmp_storage,
                                  const int temp_storage_size,
                                  const int batch_size,
                                  const int beam_width,
                                  const int vocab_size,
                                  const int end_id,
                                  const float diversity_rate);

/*
 * >> log_probs:     [batch_size*beam_size, vocab_size]
 * >> cum_log_probs: [batch_size*beam_size]
 * >> topKIds:       [batch_size*K]
 *
 */
void TopKSoftmaxOP(HUPtr<HUTensor> log_probs,
                   const HUPtr<HUTensor> bias,
                   std::vector<float> &cum_log_probs,
                   std::vector<int> &topKIds,
                   const int K,
                   void* temp_storage,
                   const int temp_storage_size,
                   uint8_t* isAllDone)
{
    const int batch_size = topKIds.size() / K;
    const int beam_width = log_probs->shape()[-2] / batch_size; // if step=0, beam_width=1
    const int vocab_size = bias->shape()[-1];

    cudaSetDevice(bias->getDeviceId().no);
    float* device_cum_log_probs;
    CUDA_CHECK(cudaMalloc(&device_cum_log_probs, cum_log_probs.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_cum_log_probs, cum_log_probs.data(),
                cum_log_probs.size() * sizeof(float), cudaMemcpyHostToDevice));

    int* device_topKIds;
    CUDA_CHECK(cudaMalloc(&device_topKIds, topKIds.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_topKIds, topKIds.data(), 
                topKIds.size() * sizeof(int), cudaMemcpyHostToDevice));

    const TT_DATA_TYPE diversity_rate = (TT_DATA_TYPE)0.0f;
    const int end_id = (int)EOS_ID;

    topK_softMax<TT_DATA_TYPE>(log_probs->data(), 
                 bias->data(), 
                 isAllDone, 
                 device_cum_log_probs, 
                 device_topKIds, 
                 K, 
                 temp_storage, 
                 temp_storage_size, 
                 batch_size, 
                 beam_width, 
                 vocab_size, 
                 end_id, 
                 diversity_rate);

    //// std::cout << "test 11" << std::endl;
    /// std::cout << "Size: " << topKIds.size() << std::endl;
    CUDA_CHECK(cudaMemcpy(topKIds.data(), device_topKIds, topKIds.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cum_log_probs.data(), device_cum_log_probs, topKIds.size() * sizeof(float), cudaMemcpyDeviceToHost));

    /*
    for(int i = 0; i < topKIds.size(); i++) {
        std::cout << topKIds[i] << "\t" << cum_log_probs[i] << std::endl;
    } */

    //// std::cout << "test 12" << std::endl;
    CUDA_CHECK(cudaFree(device_cum_log_probs));
    CUDA_CHECK(cudaFree(device_topKIds)); 
    //// std::cout << "test 13" << std::endl;

}

} // namespace TenTrans


#include "cub.cuh"
namespace fastertransformer 
{

#define NOT_FOUND -1

  template <typename T>
  __device__ __forceinline__ bool greater(const T& a, const T& b) {
    return a > b;
  }

#if !CUDA_CAN_USE_HALF
  template<>
  __device__ __forceinline__ bool greater(const __half& a, const __half& b) {
    return float(a) > float(b);
  }
#endif

  template <typename T>
  struct TopK {
    int p = NOT_FOUND;
    T u = cub::FpLimits<T>::Lowest();

    __device__ __forceinline__ void insert(T elem, int elem_id) {
      if (greater(elem, u)) {
        u = elem;
        p = elem_id;
      }
    }

    __device__ __forceinline__ void init() {
      u = cub::FpLimits<T>::Lowest();
      p = NOT_FOUND;
    }
  };

  template <typename T>
  __device__ __forceinline__ TopK<T>
  reduce_topk_op(const TopK<T>& a, const TopK<T>& b) {
    return greater(a.u, b.u) ? a : b;
  }

  template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
  __global__ void topk_stage_1(T* log_probs,
                               int* topk_tmp_id_buf,
                               T* topk_tmp_val_buf,
                               const int k,
                               const int vocab_size) {
    typedef cub::BlockReduce<TopK<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_id = bid / BLOCKS_PER_BEAM_; // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM_; // block id for a beam
    const int tmp_log_buf_index = row_id * vocab_size;
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
    TopK<T> partial;

    for (int ite = 0; ite < k; ite++) {
      partial.init();
      #pragma unroll
      for (int elem_id = tid + block_lane * BLOCK_SIZE_;
           elem_id < vocab_size;
           elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
        int index = elem_id + tmp_log_buf_index;
        partial.insert(log_probs[index], index);
      }

      TopK<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T>);

      if (tid == 0) {
        const int index = tmp_topk_buf_index + ite;
        topk_tmp_id_buf[index] = total.p;
        topk_tmp_val_buf[index] = total.u;
        // If we found a max, blank out the value in the log prob array before starting the next iteration
        if (total.p != NOT_FOUND)
          log_probs[total.p] = cub::FpLimits<T>::Lowest();
      }
      __syncthreads();
    }

    // Update prob array with original values.
    for (int beam = tid; beam < k; beam += BLOCK_SIZE_) {
      const int index = tmp_topk_buf_index + beam;
      int k_idx = topk_tmp_id_buf[index];
      if (k_idx != NOT_FOUND)
        log_probs[k_idx] = topk_tmp_val_buf[index];
    }
  }

  template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
  __global__ void topk_stage_2(const int* __restrict topk_tmp_id_buf,
                               T* topk_tmp_val_buf,
                               int* topk_id_buf,
                               T* topk_val_buf,
                               const int beams_per_batch,
                               const int vocab_size,
                               const int k) {

    const int size = beams_per_batch * k * BLOCKS_PER_BEAM_;
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;

    typedef cub::BlockReduce<TopK<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T *s_val = topk_tmp_val_buf + batch_id * size;
    TopK<T> *topks = (TopK<T>*)(array);

    TopK<T> partial;

    for (int ite = 0; ite < k; ite++) {
      partial.init();
      #pragma unroll
      for (int i = tid; i < size; i+= BLOCK_SIZE_) {
        partial.insert(s_val[i], i);
      }

      TopK<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T>);

      if (tid == 0) {
        topks[ite] = total;
        s_val[total.p] = cub::FpLimits<T>::Lowest();
      }
      __syncthreads();
    }

    for (int beam = tid; beam < k; beam += BLOCK_SIZE_) {
      int indexInRow = topks[beam].p == NOT_FOUND? 0: topks[beam].p;
      int id = topk_tmp_id_buf[batch_id * size + indexInRow];
      id = id == NOT_FOUND? 0 : id; // If no max found, all values were equal to T::min so just return 0
      const int offset = batch_id * k + beam;
      topk_id_buf[offset] = id % vocab_size;
      topk_val_buf[offset] = topks[beam].u;
    }
  }

#define CASE_K(K,BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)        \
  case K:                                                               \
  topk_stage_1<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_>                      \
  <<<batch_size * beams_per_batch * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0>>>( \
    const_cast<T*>(log_probs),                                          \
    topk_tmp_id_buf,                                                    \
    topk_tmp_val_buf,                                                   \
    k, vocab_size);                                                     \
  topk_stage_2<T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                      \
  <<<batch_size, BLOCK_SIZE_2_, K * sizeof(TopK<T>)>>>(                 \
    topk_tmp_id_buf,                                                    \
    topk_tmp_val_buf,                                                   \
    topk_id_buf,                                                        \
    topk_val_buf,                                                       \
    beams_per_batch,                                                    \
    vocab_size,                                                         \
    k);                                                                 \
  break

  template <typename T>
  void topK_kernelLauncher(const T* log_probs,
                           int* topk_tmp_id_buf,
                           T* topk_tmp_val_buf,
                           int* topk_id_buf,
                           T* topk_val_buf,
                           const int batch_size,
                           const int beams_per_batch,
                           const int k,
                           const int vocab_size) 
 {
    switch (k) {
      CASE_K(1,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(2,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(4,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(6,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(8,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(10,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(16,128,128,5);
      CASE_K(32,256,128,1);
      CASE_K(64,256,256,1);
    default:
      topk_stage_1<T, 128, 1>
        <<<batch_size * beams_per_batch * 1, 128, 0>>>(const_cast<T*>(log_probs),
                                                               topk_tmp_id_buf,
                                                               topk_tmp_val_buf,
                                                               k,
                                                               vocab_size);

      topk_stage_2<T, 128, 1>
        <<<batch_size, 128, k * sizeof(TopK<T>)>>>(topk_tmp_id_buf,
                                                           topk_tmp_val_buf,
                                                           topk_id_buf,
                                                           topk_val_buf,
                                                           beams_per_batch,
                                                           vocab_size,
                                                           k);
      break;
    }
  }
}   // namespace fastertransformer   

//////////////////////////////// TopK Softmax ///////////////////////////// 
/*
#include "cub.cuh"
namespace fastertransformer_V2 {

static const float HALF_FLT_MAX = 65504.F;
template<typename T, int MAX_K>
struct TopK
{
    int p[MAX_K];
    T u[MAX_K];

    __device__ __forceinline__ void insert(T elem, int elem_id)
    {
        if (elem > u[MAX_K-1] || (p[MAX_K-1] == -1) || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        //if (elem > u[MAX_K-1] || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        {
            u[MAX_K-1] = elem;
            p[MAX_K-1] = elem_id;
        }

        for(int k = MAX_K - 2; k >= 0; --k)
        {
            if ((u[k+1] > u[k]) || (p[k] == -1) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            //if ((u[k+1] > u[k]) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            {
                T u2 = u[k];
                int p2 = p[k];
                u[k] = u[k+1];
                p[k] = p[k+1];
                u[k+1] = u2;
                p[k+1] = p2;
            }
        }
    }

    __device__ __forceinline__ void init()
    {
        const bool IS_FP16 = std::is_same<T, half>::value;
        const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;

        for(int i = 0; i < MAX_K; i++)
        {
            p[i] = -1;
            u[i] = -MAX_T_VAL;
        }
    }
};

struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.m > b.m); 
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res; 
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template<typename T, int MAX_K>
struct TopKMD
{
    MD md;
    TopK<T, MAX_K> topk;
};

template<typename T, int MAX_K>
__device__ __forceinline__ TopKMD<T, MAX_K> reduce_topk_md_op(const TopKMD<T, MAX_K>& a, const TopKMD<T, MAX_K>& b)
{
    TopKMD<T, MAX_K> res;
    res.md = reduce_md_op(a.md, b.md);
    res.topk = reduce_topk_op(a.topk, b.topk);
    return res;
}

template<typename T, int ITEMS_PER_THREAD, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void beam_online_softmax_topk_kernel(
    const T * __restrict x,                     // log_probs:     [batch_size * beam_width, vocab_size]
    const T * __restrict b,                     // bias:          [vocab_size]
    const float * __restrict c,                 // cum_log_probs: [batch_size * beam_width * MAX_K] 
    const uint8_t  * __restrict finished,       // finished:      [batch_size]
    int * __restrict z,                         // tmp_ids:       [batch_size * beam_width * MAX_K]
    T * __restrict v,                           // tmp_vals:      [batch_size * beam_width * MAX_K]
    int beam_width,                             // - beam_width
    int V,                                      // - vocab_size
    int E)                                      // - end ID
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;

    // reposition y to data for the current vector
    x += vector_id * V;

    typedef cub::BlockReduce<TopKMD<float, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopKMD<float, MAX_K> partial;
    bool finish = finished[vector_id / beam_width];
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;

    if (finish)
    {
        for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        {
            float elem = (elem_id == E) ? MAX_T_VAL : -MAX_T_VAL;
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }
    else
    {
        for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        {
            float elem = x[elem_id] + b[elem_id];
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }

    TopKMD<float, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<float, MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * MAX_K;
        v += vector_id * MAX_K;
        c += vector_id;

        float d_total_log = logf(total.md.d);
        for(int i = 0; i < MAX_K; ++i)
        {
            float val = total.topk.u[i] - total.md.m - d_total_log;
            if (i < MAX_K)
            {
                z[i] = total.topk.p[i] + vector_id * V; // faster transformer needs absolute id
                v[i] = val + c[0];
            }
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void batch_topk_kernel(
    const int * __restrict x,     // tmp_ids:      [batch_size*beam_size, MAX_K] 
    const T * __restrict y,       // tmp_values:   [batch_size*beam_size, MAX_K]
    int * __restrict z,           // ids:          [batch_size, MAX_K]
    float * __restrict v,         // values:       [batch_size, MAX_K]
    int V,                        // - vocab_size
    T diversity_rate)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition x, y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopK<T, MAX_K> partial;
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        int i = elem_id % MAX_K;
        T elem = y[elem_id] + diversity_rate * (T) i;
        int elem_idx = elem_id; //x[elem_id];
        partial.insert(elem, elem_idx);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * MAX_K;
        v += vector_id * MAX_K;

        for(int i = 0; i < MAX_K; ++i)
        {
            if (i < MAX_K)
            {
                z[i] = x[total.p[i]];
                v[i] = (float)y[total.p[i]];
            }
        }
    }
}

template <typename T, int MAX_K>
void topK_softMax_kernelLauncher(const T* log_probs,          // [batch_size * beam_size, vocab_size]
                                 const T* bias,               // [vocab_size]
                                 const uint8_t* finished,     // [batch_size]
                                 float* cum_log_probs,        // [batch_size, MAX_K]
                                 int* ids,                    // [batch_size, MAX_K]
                                 void* temp_storage,
                                 const int temp_storage_size, 
                                 const int batch_size,
                                 const int beam_width,
                                 const int vocab_size,
                                 const int end_id,
                                 T diversity_rate)
{
    const int items_per_thread = 1;
    const int block_sz = (MAX_K < 16)? (MAX_K < 8)? SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE:128:64;

    assert(temp_storage_size % 2 == 0);
    assert(temp_storage_size >= 2 * batch_size * beam_width * MAX_K);

    const int topk_buf_offset = ceil(batch_size * beam_width * MAX_K / 4.) * 4;
    int* topk_tmp_id_buf = reinterpret_cast<int *>(temp_storage);
    T* topk_tmp_val_buf = reinterpret_cast<T *>(topk_tmp_id_buf + topk_buf_offset);
    //// float* tmp_buffer = reinterpret_cast<float *>(topk_tmp_val_buf + topk_buf_offset);

    std::cout << "SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE:" << SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE
        << "\tSMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS:" << SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS
        << "\tMAX_K:" << MAX_K << std::endl;

    std::cout << "items_per_thread:" << items_per_thread << "\tblock_sz:" << block_sz << "\tbatch_size:"
        << batch_size << "\tbeam_width:" << beam_width << "\tvocab_size:" << vocab_size << "\tend_id:" << end_id << std::endl;

    if (beam_width > 1)
    {
        beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
            <<<batch_size * beam_width, block_sz, 0>>>
                (log_probs, bias, cum_log_probs, finished, topk_tmp_id_buf, 
                 topk_tmp_val_buf, beam_width, vocab_size, end_id);

        batch_topk_kernel<T, MAX_K, 32><<<batch_size, 32, 0>>> 
                (topk_tmp_id_buf, topk_tmp_val_buf, ids, 
                 cum_log_probs, beam_width * MAX_K, diversity_rate);
    }
    else
    {
        beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
            <<<batch_size * beam_width, block_sz, 0>>>
            (log_probs, bias, cum_log_probs, finished, ids, 
             cum_log_probs, beam_width, vocab_size, end_id);
    }
}

template <typename T>
void topK_softMax(const T* log_probs,
                  const T* bias,
                  const uint8_t* finished,
                  float* cum_log_probs,
                  int* ids,
                  const int K,
                  void* temp_storage,
                  const int temp_storage_size,
                  const int batch_size,
                  const int beam_width,
                  const int vocab_size,
                  const int end_id,
                  const T diversity_rate)
{
    switch(beam_width)
    {
        case 1 :
            switch(K)
            {
                case 1:
                    topK_softMax_kernelLauncher<T, 1>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 2:
                    topK_softMax_kernelLauncher<T, 2>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 4:
                    topK_softMax_kernelLauncher<T, 4>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 8:
                    topK_softMax_kernelLauncher<T, 8>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 10:
                    topK_softMax_kernelLauncher<T, 10>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 16:
                    topK_softMax_kernelLauncher<T, 16>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                case 32:
                    topK_softMax_kernelLauncher<T, 32>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;

                default :
                    topK_softMax_kernelLauncher<T, 32>
                        (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                         batch_size, beam_width, vocab_size, end_id, diversity_rate);
                    break;
            }
            break;
    
        case 2 :
            topK_softMax_kernelLauncher<T, 4>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 3 :
            topK_softMax_kernelLauncher<T, 6>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 4 :
            topK_softMax_kernelLauncher<T, 8>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 5 :
            topK_softMax_kernelLauncher<T, 10>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 8 :
            topK_softMax_kernelLauncher<T, 16>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 16 :
            topK_softMax_kernelLauncher<T, 32>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        case 32 :
            topK_softMax_kernelLauncher<T, 64>
                    (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size,
                    batch_size, beam_width, vocab_size, end_id, diversity_rate);
            break;
        default :
            printf("[ERROR] Topk kernel does not support beamwidth = %d \n", beam_width);
            exit(0);
            break;
    }
}

template void topK_softMax<float>(const float* log_probs,
                                  const float* bias,
                                  const uint8_t* finished,
                                  float* cum_log_probs,
                                  int* ids,
                                  const int K,
                                  void* tmp_storage,
                                  const int temp_storage_size,
                                  const int batch_size,
                                  const int beam_width,
                                  const int vocab_size,
                                  const int end_id,
                                  const float diversity_rate); 

} // namespace fastertransformer_V2    */
