#pragma once
#include <cuda_runtime.h>
#include<iostream>
#include "HUTensor.h"
#include "HUFunctional.h"
#include "masked_multihead_attention.h"
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

void CopyRowsOP(HUPtr<HUTensor> out, const HUPtr<HUTensor> in, const std::vector<size_t>& indices);
void CopyRowsOP_V2(HUPtr<HUTensor> out, const HUPtr<HUTensor> in, size_t* indices, int rowsToCopy);

HUPtr<HUTensor> ReshapeOP(HUPtr<HUTensor> in, HUShape shape);

void ScaleAndShiftOP(HUPtr<HUTensor> &a, float scale, float shift);

/* 
 * tensor summation c = a + b * \beta (cuda version) 
 * >> a - a tensor
 * >> b - another tensor
 * >> c - where we put a+b*\beta. we save it in a if c is NULL
 * >> beta - the scaling factor
 * */
void Plus(HUPtr<HUTensor> &a, HUPtr<HUTensor> b, HUPtr<HUTensor> c, float scale);
void PlusBroadcast(HUPtr<HUTensor> &a, HUPtr<HUTensor> b, HUPtr<HUTensor> c);

//a : input
//b : output
void NegOP(HUPtr<HUTensor> &a, HUPtr<HUTensor> b);
	
//Transpose a tensor according to vAxis
void TransposeND(HUPtr<HUTensor> &out, HUPtr<HUTensor> in, const std::vector<int>& vAxis);

//C= alpha*A*B + beta*C
void Prod(HUPtr<HUTensor> &C, const HUPtr<HUTensor> & A, const HUPtr<HUTensor> & B, bool transA, bool transB, float beta=0, float alpha=1);

//A size is [N,M], B size is [M,B], bias size [N,1], C size is [N,B]
//C= alpha*A*B + beta*C + bias
void ProdWithBias(HUPtr<HUTensor> &C, const HUPtr<HUTensor> & A, const HUPtr<HUTensor> & B, const HUPtr<HUTensor> &bias, bool transA, bool transB, float beta=0, float alpha=1);

//C size is [N,M], bias is [N,1]
//C= C + bias
void AddBias(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias);

void ProdBatchedOP(HUPtr<HUTensor> C, const HUPtr<HUTensor> A, const HUPtr<HUTensor> B, HUPtr<HUMemPool> mem, bool transA, bool transB, float beta=0, float alpha=1);

void SoftmaxOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in, HUPtr<HUTensor> mask = nullptr);
void LogSoftmaxOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in);
void AddBiasLogSoftmaxOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in, const HUPtr<HUTensor> bias);
void AddBiasLogSoftmaxOP_V2(HUPtr<HUTensor> out, HUPtr<HUTensor> in, const HUPtr<HUTensor> bias, const int realDimBatch, uint8_t* isAllDone);

void LayerNormalOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in, HUPtr<HUTensor> scale, HUPtr<HUTensor> beta=nullptr, float eps=1e-9);
void LayerNormalOP_V2(HUPtr<HUTensor> out, HUPtr<HUTensor> in, HUPtr<HUTensor> scale, HUPtr<HUTensor> beta=nullptr, float eps=1e-9);
// 
void AddBiasInputLayerNormalOP(HUPtr<HUTensor> norm_out, HUPtr<HUTensor> out, HUPtr<HUTensor> in, const HUPtr<HUTensor> bias, HUPtr<HUTensor> gamma, HUPtr<HUTensor> beta=nullptr, float eps=1e-9);

// gelu(in+bias)
void AddBiasGeluOP(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias);
void AddBiasReluOP(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias);
void AddBiasSwishOP(HUPtr<HUTensor> &C, const HUPtr<HUTensor> bias);

// gelu: out = 0.5b * (1.0 + erf(out / sqrt(2.0)))
void GeluOP(HUPtr<HUTensor> out,  HUPtr<HUTensor> in);
// relu: out = max(0, in)
void ReluOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in);
void SwishOP(HUPtr<HUTensor> out, HUPtr<HUTensor> in);

void ConcatenateOP(HUPtr<HUTensor> out, const std::vector<HUPtr<HUTensor> >& inputs, int ax);
void Concatenate2OP(HUPtr<HUTensor> out, HUPtr<HUTensor> in1, HUPtr<HUTensor> in2);
void Concatenate1OP(HUPtr<HUTensor> out, const std::vector<HUPtr<HUTensor> >& inputs);
void ConcatContOP(HUPtr<HUTensor> out, const std::vector<HUPtr<HUTensor> >& inputs, int axis);

/* Cross-Attention */
void CrossAttentionOP(
        HUPtr<HUTensor> query_buf, const HUPtr<HUTensor> Q_bias, 
        HUPtr<HUTensor> key_cache, const HUPtr<HUTensor> K_bias, 
        HUPtr<HUTensor> value_cache, const HUPtr<HUTensor> V_bias, 
        HUPtr<HUTensor> lengths, HUPtr<HUTensor> context_buf, 
        const int realDimBatch, const uint8_t* isAllDone, 
        const int head_num, const int step);

void Transpose4DBatchMajorOP(
        HUPtr<HUTensor> k_dst, /*HUPtr<HUTensor> v_dst, */
        const HUPtr<HUTensor> k_src, /* const HUPtr<HUTensor> v_src, */
        const int local_batch_size, const int seq_len, 
        const int max_seq_len, const int size_per_head, 
        const int local_head_num);

/* Decoder Self-Attention */
void MaskedMultiHeadAttentionOP(
        const HUPtr<HUTensor> qkv_buf, const HUPtr<HUTensor> QKV_bias,
        HUPtr<HUTensor> key_cache, HUPtr<HUTensor> value_cache,
        HUPtr<HUTensor> context_buf, 
        const int realDimBatch, uint8_t* isAllDone, 
        const int head_num, const int step);

/* Encoder Self-Attention */
void EncoderUnFusedSelfAttentionOP(
        HUPtr<HUTensor> q_tmp, const HUPtr<HUTensor> Q_bias,
        HUPtr<HUTensor> k_tmp, const HUPtr<HUTensor> K_bias,
        HUPtr<HUTensor> v_tmp, const HUPtr<HUTensor> V_bias,
        HUPtr<HUTensor> att_mask, HUPtr<HUTensor> att_out,
        HUPtr<HUTensor> q_buf, HUPtr<HUTensor> k_buf, HUPtr<HUTensor> v_buf, 
        HUPtr<HUTensor> qk_buf, HUPtr<HUTensor> att_out_transpose_buf, const int head_num, HUPtr<HUMemPool> mem);

void UpdateKVBatchMajorCacheOP(
        HUPtr<HUTensor> key_src_cache, HUPtr<HUTensor> key_tgt_cache,
        HUPtr<HUTensor> value_src_cache, HUPtr<HUTensor> value_tgt_cache,
        size_t* beams_ids, uint8_t* isAllDone, 
        const int batch_size, const int beam_width,
        const int head_num, const int step);

void AddBiasInputOP(HUPtr<HUTensor> output, const HUPtr<HUTensor> bias, const HUPtr<HUTensor> input);

// for Decoder
void EmbeddingLookUpPositionEncodingOP(HUPtr<HUTensor> &output, const HUPtr<HUTensor> word_emb, const HUPtr<HUTensor> pos_emb, const std::vector<size_t> &word_ids, const size_t startPos, bool isScale);
// for Encoder
void StartIdEmbeddingLookUpPositionEncodingOP(HUPtr<HUTensor> &output, const HUPtr<HUTensor> word_emb, const HUPtr<HUTensor> pos_emb, const std::vector<size_t> &word_ids, const int batch_size, bool isScale);

// log_probs + cum_log_probs -> out
void BroadCastPlusOP(HUPtr<HUTensor> &out, HUPtr<HUTensor> log_probs, HUPtr<HUTensor> cum_log_probs);

// log_probs + cum_log_probs + bias -> out
void BroadCastPlusWithBiasOP(HUPtr<HUTensor> &out, HUPtr<HUTensor> log_probs, HUPtr<HUTensor> cum_log_probs, const HUPtr<HUTensor> bias);

// void TopKOP(HUPtr<HUTensor> logProbs, const int K, HUPtr<HUTensor> topKIds, HUPtr<HUTensor> topKValues);
void TopKOP(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int vocab_size);
// void TopKOP_V2(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int K, const int vocab_size);
void TopKOP_V2(HUPtr<HUTensor> log_probs, std::vector<int> &topKIds, std::vector<float> &topKValues, const int K, const int vocab_size, void* tmp_storage);

void TopKSoftmaxOP(HUPtr<HUTensor> log_probs,
                   const HUPtr<HUTensor> bias,
                   std::vector<float> &cum_log_probs,
                   std::vector<int> &topKIds,
                   const int K,
                   void* temp_storage,
                   const int temp_storage_size,
                   uint8_t* isAllDone);

} // namespace TenTrans
