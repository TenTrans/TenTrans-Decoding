
#pragma once
#include "HUTensor.h"
#include "HUGlobal.h"
#include "HUData.h"
#include "HUEncoderState.h"
#include "HUTensorUtil.h"
#include <vector>
#include <cuda_runtime.h>

namespace TenTrans{

struct State 
{
    /* need update for every steps */
#ifdef SELF_ATTENTION_FUSION            // 1. Pre-allocated memory, [dimBatch, MAX_DECODER_STEPS, dimModel]
    HUPtr<HUTensor> cacheKeys;
    HUPtr<HUTensor> cacheKeysTmp;

    // buffer for exchange
    HUPtr<HUTensor> cacheValues;
    HUPtr<HUTensor> cacheValuesTmp;
#else                                   // 2. Dynamic-allocated memory, [dimBatch, dimSteps, dimModel]
    HUPtr<HUTensor> cacheKeys;
    HUPtr<HUTensor> cacheValues;
#endif

    /* only update in first step */
    HUPtr<HUTensor> memoryKeys;
    HUPtr<HUTensor> memoryValues;

    State select(size_t* selIdx, int selIdxSize, uint8_t* isAllDone, int headNum, int beamSize, 
            int step, bool isBatchMajor, HUPtr<HUMemPool> memoryPool) const
    {
#ifdef SELF_ATTENTION_FUSION
        return selectBatchMajor(cacheKeys, cacheKeysTmp, cacheValues, cacheValuesTmp, memoryKeys, memoryValues, 
                selIdx, selIdxSize, isAllDone, headNum, beamSize, step, isBatchMajor, memoryPool);
#else
        return select(cacheKeys, cacheValues, memoryKeys, memoryValues, 
                selIdx, selIdxSize, headNum, beamSize, step, isBatchMajor, memoryPool);
#endif
    }

private:

#ifdef SELF_ATTENTION_FUSION
    /* 
     * 
     * Update Self-Attention Cache, Kernel Fusion
     * CacheKeys, CacheKeysTmp: [dimBatch, MAX_DECODER_STEPS, dimModel] 
     * CacheValues, CacheValuesTmp: [dimBatch, MAX_DECODER_STEPS, dimModel]  
     *
     */
    static State selectBatchMajor(HUPtr<HUTensor> keys, HUPtr<HUTensor> keysTmp, HUPtr<HUTensor> values, HUPtr<HUTensor> valuesTmp, 
            HUPtr<HUTensor> memoryK, HUPtr<HUTensor> memoryV, size_t* selIdx, int selIdxSize, uint8_t* isAllDone, 
            int headNum, int beamSize, int step, bool isBatchMajor, HUPtr<HUMemPool> memoryPool)
    {
        int dimBatchBeam = keys->shape()[-3];
        int MAX_SEQ_LEN = keys->shape()[-2];
        int dimModel = keys->shape()[-1];
        // int selIdxSize = selIdx.size();

        HUPtr<HUTensor> newKeys, newKeysTmp, newValues, newValuesTmp;
        /* beamSize > 1, need Update Cache_KV */
        if (beamSize > 1)
        {
            // step > 1, dimBatchBeam = batchSize*beamSize 
            if (dimBatchBeam == selIdxSize)
            {
                int batchSize = selIdxSize / beamSize;
                HUTensorUtil::UpdateKVBatchMajorCache(
                        keys, keysTmp, 
                        values, valuesTmp, 
                        selIdx, isAllDone, 
                        batchSize, beamSize, 
                        headNum, step);

                // exchange
                newKeys = keysTmp;
                newKeysTmp = keys;
                newValues = valuesTmp;
                newValuesTmp = values;
            }
            else
            {
                /* step = 1, repeat beam_size times */
                if (selIdxSize / dimBatchBeam == beamSize)
                {
                    // std::cout << "[1]" << std::endl;
                    keys = HUTensorUtil::Reshape(keys, {dimBatchBeam, 1, MAX_SEQ_LEN*dimModel});
                    HUPtr<HUTensor> repeatKeys = HUTensorUtil::Reshape(HUTensorUtil::Repeat(keys, beamSize, -2, memoryPool), 
                        {selIdxSize, MAX_SEQ_LEN, dimModel});
                    keysTmp = HUTensorUtil::Reshape(keysTmp, {dimBatchBeam, 1, MAX_SEQ_LEN*dimModel}); 
                    HUPtr<HUTensor> repeatKeysTmp = HUTensorUtil::Reshape(HUTensorUtil::Repeat(keysTmp, beamSize, -2, memoryPool), 
                        {selIdxSize, MAX_SEQ_LEN, dimModel});
                    // std::cout << "[2]" << std::endl;
                     
                    values = HUTensorUtil::Reshape(values, {dimBatchBeam, 1, MAX_SEQ_LEN*dimModel});
                    HUPtr<HUTensor> repeatValues = HUTensorUtil::Reshape(HUTensorUtil::Repeat(values, beamSize, -2, memoryPool), 
                        {selIdxSize, MAX_SEQ_LEN, dimModel});
                    valuesTmp = HUTensorUtil::Reshape(valuesTmp, {dimBatchBeam, 1, MAX_SEQ_LEN*dimModel});
                    HUPtr<HUTensor> repeatValuesTmp = HUTensorUtil::Reshape(HUTensorUtil::Repeat(valuesTmp, beamSize, -2, memoryPool), 
                        {selIdxSize, MAX_SEQ_LEN, dimModel});
                    // std::cout << "[3]" << std::endl;

                    memoryPool->free(keys->memory());
                    memoryPool->free(keysTmp->memory());
                    memoryPool->free(values->memory());
                    memoryPool->free(valuesTmp->memory());
                    // std::cout << "[4]" << std::endl;

                    int batchSize = dimBatchBeam;
                    HUTensorUtil::UpdateKVBatchMajorCache(
                            repeatKeys, repeatKeysTmp, 
                            repeatValues, repeatValuesTmp, 
                            selIdx, isAllDone, 
                            batchSize, beamSize, 
                            headNum, step);

                    // exchange 
                    newKeys = repeatKeysTmp; 
                    newKeysTmp = repeatKeys; 
                    newValues = repeatValuesTmp; 
                    newValuesTmp = repeatValues;
                }
                else
                {
                    std::cout << "[HUDecoderState] Self-Attention Cache_KV format is not correct ... " << std::endl;
                    newKeys = keys;
                    newKeysTmp = keysTmp;
                    newValues = values;
                    newValuesTmp = valuesTmp;
                }
            }
        }
        else  /* not update Cache_KV */
        {
            // not exchange
            newKeys = keys;
            newKeysTmp = keysTmp;
            newValues = values;
            newValuesTmp = valuesTmp;
        }

        return {newKeys, newKeysTmp, newValues, newValuesTmp, memoryK, memoryV};
    }
#endif

    /* 
     * 
     * Update Self-Attention Cache.
     * CacheKeys: [dimBatch, dimHeads*dimPerHead/dimX, dimSteps, dimX] 
     * CacheValues: [dimBatch, dimHeads, dimSteps, dimPerHead]  
     *
     */ 
    static State selectBatchMajor_V1(HUPtr<HUTensor> keys, HUPtr<HUTensor> values, HUPtr<HUTensor> memoryK, HUPtr<HUTensor> memoryV,
            const std::vector<size_t>& selIdx, int beamSize, bool isBatchMajor, HUPtr<HUMemPool> memoryPool)
    {
        int dimBatchBeam = keys->shape()[-4];
        int dimTmp = keys->shape()[-3];
        int dimSteps = keys->shape()[-2];
        int dimX = keys->shape()[-1];
        int selIdxSize = selIdx.size();

        /*
        cudaEvent_t start, stop;
        float elapsedTime = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        */

        HUPtr<HUTensor> selKeys, selValues;
        if (beamSize > 1)  // update Cache_KV
        {
            // step > 1
            if (dimBatchBeam == selIdxSize)
            {
                keys = HUTensorUtil::Reshape(keys, {selIdxSize, dimTmp*dimSteps*dimX});
                selKeys = HUTensorUtil::CopyRows(keys, selIdx, memoryPool);
                selKeys = HUTensorUtil::Reshape(selKeys, {selIdxSize, dimTmp, dimSteps, dimX});
                
                values = HUTensorUtil::Reshape(values, {selIdxSize, dimTmp*dimSteps*dimX});
                selValues = HUTensorUtil::CopyRows(values, selIdx, memoryPool);
                selValues = HUTensorUtil::Reshape(selValues, {selIdxSize, dimTmp, dimSteps, dimX});
            }
            else
            {
                // if the first step, repeat beam_size times
                if (selIdxSize / dimBatchBeam == beamSize)
                {
                    keys = HUTensorUtil::Reshape(keys, {dimBatchBeam, 1, dimTmp*dimSteps*dimX});
                    HUPtr<HUTensor> repeatKeys = HUTensorUtil::Reshape(HUTensorUtil::Repeat(keys, beamSize, -2, memoryPool), 
                            {selIdxSize, dimTmp*dimSteps*dimX});
                    selKeys = HUTensorUtil::CopyRows(repeatKeys, selIdx, memoryPool);
                    selKeys = HUTensorUtil::Reshape(selKeys, {selIdxSize, dimTmp, dimSteps, dimX});
                    
                    values = HUTensorUtil::Reshape(values, {dimBatchBeam, 1, dimTmp*dimSteps*dimX});
                    HUPtr<HUTensor> repeatValues = HUTensorUtil::Reshape(HUTensorUtil::Repeat(values, beamSize, -2, memoryPool), 
                            {selIdxSize, dimTmp*dimSteps*dimX});
                    selValues = HUTensorUtil::CopyRows(repeatValues, selIdx, memoryPool);
                    selValues = HUTensorUtil::Reshape(selValues, {selIdxSize, dimTmp, dimSteps, dimX});
                    
                    memoryPool->free(repeatKeys->memory());
                    memoryPool->free(repeatValues->memory());
                }
                else
                {
                    std::cout << "[HUDecoderState] Self-Attention Cache_KV format is not correct ... " << std::endl;
                    selKeys = keys;
                    selValues = values;
                }
            }
        }
        else  // not update Cache_KV
        {
            selKeys = keys;
            selValues = values;
        }

        /*
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Time Cost(ms): " << elapsedTime << std::endl;
        */

        return {selKeys, selValues, memoryK, memoryV};

    } 


    /*
     * Update Self-Attention Cache
     * CacheKeys: [dimBatch, dimSteps, dimModel] 
     * CacheValues: [dimBatch, dimSteps, dimModel]  
     *
     */
	static State select(HUPtr<HUTensor> keys, HUPtr<HUTensor> values, HUPtr<HUTensor> memoryK, HUPtr<HUTensor> memoryV, 
            size_t* selIdx, int selIdxSize, int headNum, int beamSize, int step, bool isBatchMajor, HUPtr<HUMemPool> memoryPool) 
    {
        int dimBatchBeam = keys->shape()[-3];
        int dimSteps = keys->shape()[-2];
        int dimModel = keys->shape()[-1];
        // int selIdxSize = selIdx.size();
        
        HUPtr<HUTensor> selKeys, selValues;
        if (beamSize > 1)    // update Cache_KV
        {
            // step > 1
            if (dimBatchBeam == selIdxSize)
            {
                keys = HUTensorUtil::Reshape(keys, {selIdxSize, dimSteps*dimModel});
                // selKeys = HUTensorUtil::CopyRows(keys, selIdx, memoryPool);
                selKeys = HUTensorUtil::CopyRows_V2(keys, selIdx, selIdxSize, memoryPool);
                selKeys = HUTensorUtil::Reshape(selKeys, {selIdxSize, dimSteps, dimModel});
                
                values = HUTensorUtil::Reshape(values, {selIdxSize, dimSteps*dimModel});
                // selValues = HUTensorUtil::CopyRows(values, selIdx, memoryPool);
                selValues = HUTensorUtil::CopyRows_V2(values, selIdx, selIdxSize, memoryPool);
                selValues = HUTensorUtil::Reshape(selValues, {selIdxSize, dimSteps, dimModel});
            } 
            else
            {
                // if the first step, repeat beam_size times
                if (selIdxSize / dimBatchBeam == beamSize)
                {
                    keys = HUTensorUtil::Reshape(keys, {dimBatchBeam, 1, dimSteps, dimModel});
                    HUPtr<HUTensor> repeatKeys = HUTensorUtil::Reshape(HUTensorUtil::Repeat(keys, beamSize, -3, memoryPool),
                            {selIdxSize, dimSteps*dimModel});
                    // selKeys = HUTensorUtil::CopyRows(repeatKeys, selIdx, memoryPool);
                    selKeys = HUTensorUtil::CopyRows_V2(repeatKeys, selIdx, selIdxSize, memoryPool);
                    selKeys = HUTensorUtil::Reshape(selKeys, {selIdxSize, dimSteps, dimModel});

                    values = HUTensorUtil::Reshape(values, {dimBatchBeam, 1, dimSteps, dimModel});
                    HUPtr<HUTensor> repeatValues = HUTensorUtil::Reshape(HUTensorUtil::Repeat(values, beamSize, -3, memoryPool), 
                            {selIdxSize, dimSteps*dimModel});
                    // selValues = HUTensorUtil::CopyRows(repeatValues, selIdx, memoryPool);
                    selValues = HUTensorUtil::CopyRows_V2(repeatValues, selIdx, selIdxSize, memoryPool);
                    selValues = HUTensorUtil::Reshape(selValues, {selIdxSize, dimSteps, dimModel});
                    
                    memoryPool->free(repeatKeys->memory());
                    memoryPool->free(repeatValues->memory());
                }
                else
                {
                    std::cout << "[HUDecoderState] Self-Attention Cache_KV format is not correct ... " << std::endl;
                    selKeys = keys;
                    selValues = values;
                }
            }
        }
        else    // not update Cache_KV
        {
            selKeys = keys;
            selValues = values;
        }

        return {selKeys, selValues, memoryK, memoryV};
	}

};

class States 
{
private:
  std::vector<State> states_;

public:
  States() {}
  States(const std::vector<State>& states) : states_(states) {}
  States(size_t num, State state) : states_(num, state) {}

  std::vector<State>::iterator begin() { return states_.begin(); }
  std::vector<State>::iterator end()   { return states_.end(); }
  std::vector<State>::const_iterator begin() const { return states_.begin(); }
  std::vector<State>::const_iterator end()   const { return states_.end(); }

  State& operator[](size_t i) { return states_[i]; };
  const State& operator[](size_t i) const { return states_[i]; };

  State& back() { return states_.back(); }
  const State& back() const { return states_.back(); }

  State& front() { return states_.front(); }
  const State& front() const { return states_.front(); }

  size_t size() const { return states_.size(); };

  void push_back(const State& state) { states_.push_back(state); }

  // create updated set of states that reflect reordering and dropping of hypotheses
  States select(size_t* selIdx, int selIdxSize, uint8_t* isAllDone, // [beamIndex * activeBatchSize + batchIndex] 
          int headNum, int beamSize, int step, bool isBatchMajor, HUPtr<HUMemPool> memoryPool) const {
    States selected;
    for(auto& state : states_) {
      selected.push_back(state.select(selIdx, selIdxSize, isAllDone, headNum, beamSize, step, isBatchMajor, memoryPool));
    }
    return selected;
  }

  void reverse() { std::reverse(states_.begin(), states_.end()); }

  void clear() { states_.clear(); }
};


class HUDecoderState {
private:
	States states_;
	HUPtr<HUTensor> logProbs_;
	HUPtr<HUEncoderState> encState_;
	HUPtr<HUBatch> batch_;

	HUPtr<HUTensor> targetEmbeddings_;
	HUPtr<HUTensor> targetMask_;
	HUPtr<HUTensor> targetIndices_;
	HUPtr<HUMemPool> memPool_;

	size_t position_{0};                 // i-th step
    int heads_;

public:
	HUDecoderState(const States& states, HUPtr<HUTensor> logProbs, const HUPtr<HUEncoderState> encState, HUPtr<HUBatch> batch, int headNum, HUPtr<HUMemPool> mem)
		: states_(states), logProbs_(logProbs), encState_(encState), batch_(batch), heads_(headNum), memPool_(mem) {}

	void Free()
	{
		if(states_.size() == 0) {
			return;
        }

        // memPool_->free(logProbs_->memory());

#ifdef SELF_ATTENTION_FUSION
        memPool_->free(logProbs_->memory());
#else
		memPool_->free(logProbs_->memory());
        for(int i=0; i < states_.size(); i++) 
        {
            // std::cout << ">>>>>>>> wkx <<<<<<<<" << std::endl;
            memPool_->free((states_[i].cacheKeys)->memory());
            memPool_->free((states_[i].cacheValues)->memory());
        }
#endif

	}

    void FinalFree() 
    {
        // std::cout << "[targetEmbeddings_] free ..." << std::endl;
        // memPool_->free(targetEmbeddings_->memory());
        // std::cout << "[targetMask] free ..." << std::endl;
        // memPool_->free(targetMask_->memory());
        // std::cout << "[targetIndices_] free ..." << std::endl;
        // memPool_->free(targetIndices_->memory());

        // std::cout << "[encContext] free ..." << std::endl;
        auto encContext = encState_->getContext();
        memPool_->free(encContext->memory());

        // std::cout << "[encMask] free ..." << std::endl;
        auto encMask = encState_->getMask();
        memPool_->free(encMask->memory());

        memPool_->free(logProbs_->memory());

        for(int i = 0; i < states_.size(); i++)
        {
            memPool_->free((states_[i].memoryKeys)->memory());
            memPool_->free((states_[i].memoryValues)->memory());

    #ifdef SELF_ATTENTION_FUSION    // free
            memPool_->free((states_[i].cacheKeys)->memory());
            memPool_->free((states_[i].cacheValues)->memory());
            memPool_->free((states_[i].cacheKeysTmp)->memory());
            memPool_->free((states_[i].cacheValuesTmp)->memory());
    #else
            memPool_->free((states_[i].cacheKeys)->memory());
            memPool_->free((states_[i].cacheValues)->memory());
    #endif
        }
    }

	virtual const HUPtr<HUEncoderState>& getEncoderState() const {
    	return encState_;
  	}

	virtual HUPtr<HUTensor> getLogProbs() const { return logProbs_; }
	virtual void setLogProbs(HUPtr<HUTensor> logProbs) { logProbs_ = logProbs; }
	
	virtual HUPtr<HUDecoderState> select(size_t* selIdx, int selIdxSize, uint8_t* isAllDone, 
            int beamSize, HUPtr<HUMemPool> memoryPool) const 
    {
    	auto selectedState = HUNew<HUDecoderState>(states_.select(selIdx, selIdxSize, isAllDone, heads_, beamSize, 
                    position_, /*isBatchMajor=*/true, memoryPool), logProbs_, encState_, batch_, heads_, memoryPool);

    	// Set positon of new state based on the target token position of current state
    	selectedState->setPosition(getPosition());
    	return selectedState;
  	}

	virtual const States& getStates() const { return states_; }
	virtual HUPtr<HUTensor> getTargetEmbeddings() const { return targetEmbeddings_; }

	virtual void setTargetEmbeddings(HUPtr<HUTensor> targetEmbeddings) {
		targetEmbeddings_ = targetEmbeddings;
  	}

	virtual HUPtr<HUTensor> getTargetIndices() const { return targetIndices_; }

	virtual void setTargetIndices(HUPtr<HUTensor> targetIndices) {
    	targetIndices_ = targetIndices;
  	}

	virtual HUPtr<HUTensor> getTargetMask() const { return targetMask_; }

    virtual void setTargetMask(HUPtr<HUTensor> targetMask) { targetMask_ = targetMask; }
	
	virtual const std::vector<size_t>& getSourceWords() const {
    	return getEncoderState()->getSourceWords();
  	}

	HUPtr<HUBatch> getBatch() const { return batch_; }

	size_t getPosition() const { return position_; }
    int getHeads() const { return heads_; };
	void setPosition(size_t position) { position_ = position; }
};

}
