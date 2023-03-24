#pragma once
#include "HUGlobal.h"
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>
#include "HUTensor.h"
#include "cnpy.h"
#include "HUTensorOP.h"
#include "nvidia_export/attention_kernels.cuh"
//using namespace std;


/*
struct EncoderSelfAttentionBuffer {
    HUPtr<HUTensor> q_tmp, k_tmp, v_tmp;
    HUPtr<HUTensor> q_buf, k_buf, v_buf;
    HUPtr<HUTensor> qk_buf;
    HUPtr<HUTensor> att_out_transpose_buf;
};
*/

namespace TenTrans{

struct EncoderSelfAttentionBuffer {
    HUPtr<HUTensor> q_tmp, k_tmp, v_tmp;
    HUPtr<HUTensor> q_buf, k_buf, v_buf;
    HUPtr<HUTensor> qk_buf;
    HUPtr<HUTensor> att_out_transpose_buf;
}; 

class HUTensorUtil{
public:
	//Get selected tensor shape by indices (CopyRows)
	static HUShape GetCopyRowsShape(HUPtr<HUTensor> a, const std::vector<size_t>& indices);
	//Get transposed tensor shape by axes (Transpose)
	static HUShape GetTransposeShape(HUPtr<HUTensor> a, const std::vector<int>& axes);
	//Get affined tensor shape by tensors to be affined (Affine)
	static HUShape GetAffineShape(HUPtr<HUTensor> a, HUPtr<HUTensor> b, bool transA=false, bool transB=false);
	//Get Dot Product tensor shape by tensors to be dot procted (DotBatched)
	static HUShape GetDotBatchedShape(HUPtr<HUTensor> a, HUPtr<HUTensor> b, bool transA, bool transB, float scalar);
	//Get Concatenate tensor shape (Concatenate)
	static HUShape GetConcatenateShape(std::vector<HUPtr<HUTensor> >& nodes, int &ax);

	//Save tensor into file
	//name : file name 
	//a : tensor to be saved 
	//tensorName : tensor name 
	static void Save(const std::string name, HUPtr<HUTensor> a, const std::string tensorName);

	//Reshape tensor into specified dimension
	static HUPtr<HUTensor> AtLeastNd(HUPtr<HUTensor> a, size_t	dims);

	static HUPtr<HUTensor> TransposeTimeBatch(HUPtr<HUTensor> input, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

	static HUPtr<HUTensor> Transpose(HUPtr<HUTensor> input, const std::vector<int>& axes, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

	static HUPtr<HUTensor> Neg(HUPtr<HUTensor> input, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);
	static HUPtr<HUTensor> Plus(HUPtr<HUTensor> a, HUPtr<HUTensor> b, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

	static HUPtr<HUTensor> ScaleAndShift(HUPtr<HUTensor> &a, float scale, float shift);

	//convert multiplicative 1/0 mask to additive 0/-inf log mask, and transpose to match result of bdot() op in Attention()
	static HUPtr<HUTensor> TransposedLogMask(HUPtr<HUTensor> mask, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

	//y = w*x + b
	//y is output tensor
	static void Affine(HUPtr<HUTensor> &y, HUPtr<HUTensor> x, HUPtr<HUTensor> w, HUPtr<HUTensor> b, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, bool transA=false, bool transB=false, float beta=0.f, float alpha=1.f);
	static HUPtr<HUTensor> Affine(HUPtr<HUTensor> x, HUPtr<HUTensor> w, HUPtr<HUTensor> b, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, 
            bool transA=false, bool transB=false, float beta=0.f, float alpha=1.f);

    // y = w*x
    static HUPtr<HUTensor> Multiply(HUPtr<HUTensor> x, HUPtr<HUTensor> w, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, 
            bool transA=false, bool transB=false, float beta=0.f, float alpha=1.f);
    static void Multiply_v2(HUPtr<HUTensor> &y, HUPtr<HUTensor> x, HUPtr<HUTensor> w, bool transA=false, bool transB=false, float beta=0.f, float alpha=1.f);

	static HUPtr<HUTensor> ProdBatched(HUPtr<HUTensor> A, HUPtr<HUTensor> B, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, bool transA=false, bool transB=false, float beta = 0, float scalar = 1);

	static HUPtr<HUTensor> Softmax(HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, HUPtr<HUTensor> mask = nullptr);

	static HUPtr<HUTensor> LogSoftmax(HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);
    static HUPtr<HUTensor> AddBiasLogSoftmax(HUPtr<HUTensor> in, const HUPtr<HUTensor> bias, const int realDimBatch, uint8_t* isAllDone, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

	static HUPtr<HUTensor> LayerNormalization(HUPtr<HUTensor> in, HUPtr<HUTensor> gamma, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, HUPtr<HUTensor> beta=nullptr, float eps=1e-9);
    static HUPtr<HUTensor> AddBiasInputLayerNormalization(HUPtr<HUTensor> in, HUPtr<HUTensor> x, const HUPtr<HUTensor> bias, HUPtr<HUTensor> gamma, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device, HUPtr<HUTensor> beta=nullptr, float eps=1e-9);

	static HUPtr<HUTensor> Activation(HUPtr<HUTensor> in, ActivationType type, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

	static HUPtr<HUTensor> Relu(HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);
	static HUPtr<HUTensor> Swish(HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);
    static HUPtr<HUTensor> Gelu(HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);
	static void AddBiasActivation(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias, ActivationType type);

	static HUPtr<HUTensor> Zeros(HUShape inShape, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);
	static HUPtr<HUTensor> Ones(HUShape inShape, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);
    static HUPtr<HUTensor> Set(HUShape inShape, const float num, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

	static HUPtr<HUTensor> Reshape(HUPtr<HUTensor> in, HUShape shape);

	static HUPtr<HUTensor> CopyRows(HUPtr<HUTensor> in, const std::vector<size_t>& indices, HUPtr<HUMemPool> mem);
    static HUPtr<HUTensor> CopyRows_V2(HUPtr<HUTensor> in, size_t* indices, int num, HUPtr<HUMemPool> mem);

	static HUPtr<HUTensor> Concatenate(std::vector<HUPtr<HUTensor> > nodes, int ax, HUPtr<HUMemPool> mem);
    static void Split(HUPtr<HUTensor> in, int num, std::vector<HUPtr<HUTensor> > &nodes, HUPtr<HUMemPool> mem, int ax=-1);

	static HUPtr<HUTensor> Repeat(HUPtr<HUTensor> a, size_t repeats, int ax, HUPtr<HUMemPool> mem);

	static HUPtr<HUTensor> ConstantFloat(HUShape inShape, std::vector<TT_DATA_TYPE> data, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

    // out = in
    static void CopyFrom(HUPtr<HUTensor> &out, HUPtr<HUTensor> in, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);
    static void Add_QKV_Bias_Transpose(
            HUPtr<HUTensor> buf_q, 
            HUPtr<HUTensor> buf_k, 
            HUPtr<HUTensor> buf_v, 
            HUPtr<HUTensor> Q, 
            HUPtr<HUTensor> b_Q, 
            HUPtr<HUTensor> K, 
            HUPtr<HUTensor> b_K, 
            HUPtr<HUTensor> V, 
            HUPtr<HUTensor> b_V, 
            const int batch_size, 
            const int seq_len, 
            const int head_num, 
            const int size_per_head);
    
    static HUPtr<HUTensor> FusedQKVSelfAttention(
            const HUPtr<HUTensor> qkv_buf, const HUPtr<HUTensor> QKV_bias,
            HUPtr<HUTensor> key_cache, HUPtr<HUTensor> value_cache, 
            const int realDimBatch, uint8_t* isAllDone, 
            const int head_num, const int step,  
            HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);
   
    static HUPtr<HUTensor> EncoderUnFusedSelfAttention(
            HUPtr<HUTensor> input, HUPtr<HUTensor> att_mask,
            const HUPtr<HUTensor> Q, const HUPtr<HUTensor> Q_bias, 
            const HUPtr<HUTensor> K, const HUPtr<HUTensor> K_bias, 
            const HUPtr<HUTensor> V, const HUPtr<HUTensor> V_bias,
            const int head_num, EncoderSelfAttentionBuffer &params, 
            HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

    // static HUPtr<HUTensor> ReduceSum(HUPtr<HUTensor> in, int ax, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);
    static HUPtr<HUTensor> CrossAttention(
            HUPtr<HUTensor> query_buf, const HUPtr<HUTensor> Q_bias, 
            HUPtr<HUTensor> key_cache, const HUPtr<HUTensor> K_bias, 
            HUPtr<HUTensor> value_cache, const HUPtr<HUTensor> V_bias, 
            HUPtr<HUTensor> lengths, 
            const int realDimBatch, const uint8_t* isAllDone, 
            const int head_num, const int step, 
            HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

    static void Transpose4DBatchMajor(
            HUPtr<HUTensor> &k_dst, /*HUPtr<HUTensor> &v_dst, */
            const HUPtr<HUTensor> k_src, /*const HUPtr<HUTensor> v_src, */
            HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

    static void UpdateKVBatchMajorCache(
        HUPtr<HUTensor> key_src_cache, HUPtr<HUTensor> key_tgt_cache,
        HUPtr<HUTensor> value_src_cache, HUPtr<HUTensor> value_tgt_cache,
        size_t* beams_ids, uint8_t* isAllDone, 
        const int batch_size, const int beam_width, const int head_num, const int step);

    // output = output + input + bias
    static void AddBiasInput(HUPtr<HUTensor> output, const HUPtr<HUTensor> bias, const HUPtr<HUTensor> input);

    static HUPtr<HUTensor> EmbeddingLookUpPositionEncoding(const HUPtr<HUTensor> word_emb, const HUPtr<HUTensor> pos_emb, const std::vector<size_t> &word_ids, const size_t startPos, bool isScale, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

    static HUPtr<HUTensor> StartIdEmbeddingLookUpPositionEncoding(const HUPtr<HUTensor> word_emb, const HUPtr<HUTensor> pos_emb, const std::vector<size_t> &word_ids, const int batch_size, bool isScale, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

    static HUPtr<HUTensor> BroadCastPlus(HUPtr<HUTensor> log_probs, HUPtr<HUTensor> cum_log_probs, HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

    static HUPtr<HUTensor> BroadCastPlusWithBias(HUPtr<HUTensor> log_probs, HUPtr<HUTensor> cum_log_probs, const HUPtr<HUTensor> bias,  HUPtr<HUMemPool> mem, HUPtr<HUDevice> device);

    static void TopK(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int vocab_size);
    // static void TopK_V2(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int K, const int vocab_size);

    static void TopK_V2(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int K, const int vocab_size, void* tmp_storage);

    static void TopKSoftmax(HUPtr<HUTensor> log_probs,
                            const HUPtr<HUTensor> bias,
                            std::vector<float> &cum_log_probs,
                            std::vector<int> &topKIds,
                            const int K,
                            void* temp_storage,
                            const int temp_storage_size,
                            uint8_t* isAllDone); 

    // static void TopK(HUPtr<HUTensor> logProbs, const int K, HUPtr<HUTensor> topKIds, HUPtr<HUTensor> topKValues);
}; // class HUTensorUtil

} // namespace TeTrans

