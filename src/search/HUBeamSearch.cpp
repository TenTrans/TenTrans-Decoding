#include "HUBeamSearch.h"

namespace TenTrans{

HUHistories HUBeamSearch::Search(HUPtr<HUBatch> batch)
{
    /* [batch_size, beam_size], HUPtr<HUBeamCell> note: 1. Beam->vector<HUPtr<HUBeamCell>>; 2. vector<Beam> Beams */
    size_t dimBatch = batch->batchSize();
    Beams batchBeams(dimBatch);
    for(auto& beam: batchBeams) {
        beam.resize(beamSize_, HUNew<HUBeamCell>());        // default -> HUBeamCell() : prevHyp_(nullptr), prevIndex_(0), word_(0), pathScore_(0.0) {}
    }

    /* Initialization for BeamSearch History */
    HUHistories histories;                                              // batch-sentence history
    size_t decodeLen = this->options_->get<size_t>("decode-length");    // for length penalty
    float alpha = this->options_->get<float>("normalize");              // alpha, for length penalty
    size_t maxStep = 0;                                                 // maximum time steps for beam search
    for (size_t i = 0; i < dimBatch; i++)
    {
        size_t sentId = batch->getSentenceIds()[i];    // sentence ID
        float curMaxLen = batch->lengths()[i] + decodeLen; 
        if (curMaxLen > maxStep) {
            maxStep = curMaxLen;
        }

        /*
        float curMaxLen = decodeLen;                   // srcLen+decodeLen-1, for length penalty
        size_t batchWidth = batch->batchWidth();
        for (size_t j = 0; j < batchWidth; j++) {
            size_t mskId = i * batchWidth + j;
            curMaxLen += (size_t)batch->mask()[mskId];
        }

        if (curMaxLen > maxStep) {
            maxStep = curMaxLen;
        }
        */

        // std::cout << "[beamSize]: " << beamSize_ << "[curMaxLen]: " << curMaxLen << std::endl;
        auto history = HUNew<HUHistory>(sentId, curMaxLen-1, beamSize_, alpha);   // single-sentence history
        histories.push_back(history);
    }
    // std::cout << "[maxStep]: " << maxStep << std::endl;

    for(int i = 0; i < dimBatch; ++i) {
        histories[i]->Add(batchBeams[i], trgEosId_);
    }

    /* Get Nbest, topK=2*localBeamSize */
    int topK = (this->beamSize_ == 1) ? 1 : 2*this->beamSize_;
    auto getNBestList = createGetNBestListFn(topK, dimBatch, device_->getDeviceId());

    bool isFirstStep = true;
    auto curState = encdec_->PrepareForDecoding(batch);   // HUPtr<HUDecoderState>, first DecoderState for decoding

    size_t curStep = 1;
    std::vector<bool> isAllDone(dimBatch, false);
    std::vector<uint8_t> isAllDoneCopy(dimBatch, 0);
   
    /* used for early stop */
    uint8_t* isAllDoneDevice = nullptr;
#ifdef DECODER_PADDING_OPTIMIZE
    cudaSetDevice(device_->getDeviceId().no);
    cudaMalloc(&isAllDoneDevice, isAllDoneCopy.size() * sizeof(uint8_t));
    cudaMemcpy(isAllDoneDevice, isAllDoneCopy.data(), isAllDoneCopy.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
#endif
   
    // avoid long decoding steps
    maxStep = maxStep > MAX_DECODER_STEPS ? MAX_DECODER_STEPS : maxStep;
    while(curStep < maxStep)
    {
        std::vector<size_t> hypIndices;     // [dimBatch*dimBeam] of previousState indices
        std::vector<size_t> embIndices;     // [dimBatch*dimBeam] of embed indices
        HUPtr<HUTensor> curPathScores;      // [dimBatch*dimBeam] of pathScores

        if (isFirstStep)
        {
            /* [dimBatch*dimBeam, 1], dimBeam=1 if isFirstStep=True */
            curPathScores = HUTensorUtil::Zeros({dimBatch, 1}, memPool_, device_);
        }
        else
        {
            std::vector<float> beamScores;
            for (size_t sentId = 0; sentId < dimBatch; sentId++) {
                auto& beam = batchBeams[sentId];
                // auto& beam = newBatchBeams[sentId];
                // std::cout << "[Start, BeamSearch sentId] "<< sentId << std::endl;
                for (size_t beamId = 0; beamId < beamSize_; beamId++) {
                    auto hyp = beam[beamId];
                    size_t prevStateId = (size_t)(sentId * beamSize_ + hyp->GetPrevStateIndex());
                    size_t wordId = (size_t)hyp->GetWord();
                    float pathScore = hyp->GetPathScore();
                    hypIndices.push_back(prevStateId);
                    embIndices.push_back(wordId);
                    beamScores.push_back(pathScore);
                    // std::cout << "previousIdx: " << prevStateId << "\t" << "embIdx: " << wordId << "\t" << "scores: " << pathScore << std::endl;
                    /*
                    hypIndices.push_back((size_t)hyp->GetPrevStateIndex()); 
                    embIndices.push_back((size_t)hyp->GetWord());
                    beamScores.push_back(hyp->GetPathScore());
                    std::cout << "previousIdx: " << hyp->GetPrevStateIndex() << "\t" << "embIdx: " << hyp->GetWord() << "\t" << "scores: " << hyp->GetPathScore() << std::endl; */
                }
            }
            curPathScores = HUTensorUtil::ConstantFloat({(int)dimBatch*beamSize_, 1}, beamScores, memPool_, device_);
        }

        // cudaSetDevice(key_src_cache->getDeviceId().no);
        // CUDA_CHECK(cudaMemcpy(isAllDoneDevice, isAllDone.data(), isAllDone.size() * sizeof(bool), cudaMemcpyHostToDevice));

        auto state = encdec_->Step(curState, hypIndices, embIndices, beamSize_, isAllDoneCopy, isAllDoneDevice);
        curState->Free();
        curState = state;
#ifdef DECODER_DEBUG
        LOG(trace, "[TenTrans][HUBeamSearch] Search, [begin] curPathScores {}", curPathScores->debug());
#endif

#ifdef DECODER_DEBUG
        LOG(trace, "[TenTrans][HUBeamSearch] Search, state->getLogProbs() {}", state->getLogProbs()->debug());
#endif
        /* [dimBatch*dimBeam, dimTgtWords] */
        auto curLogits = state->getLogProbs();
        // auto pathScores = HUTensorUtil::Plus(curPathScores, curLogits, memPool_, device_);

        HUPtr<HUTensor> pathScores;
        if (this->beamSize_ > 1) {
            pathScores = HUTensorUtil::BroadCastPlus(curLogits, curPathScores, memPool_, device_);
        }
        else {
            pathScores = HUTensorUtil::BroadCastPlusWithBias(curLogits, curPathScores, 
                    this->encdec_->GetOutputLayerBias(), memPool_, device_);
        }
        this->memPool_->free(curPathScores->memory());

        // [dimBatch*dimBeam, dimTgtWords] -> [dimBatch dimBeam*dimTgtWords]
        int dimBatchBeam = curLogits->shape()[-2];
        int dimTgtWords = curLogits->shape()[-1];
        curPathScores = HUTensorUtil::Reshape(pathScores, {dimBatch, dimBatchBeam/dimBatch*dimTgtWords});
        /*
        pathScores = HUTensorUtil::Reshape(pathScores, {dimBatch, dimBatchBeam/dimBatch*dimTgtWords});
        this->memPool_->free(curPathScores->memory());
        curPathScores = pathScores;
        */

#ifdef DECODER_DEBUG
        LOG(trace, "[TenTrans][HUBeamSearch] Search, [after Plus] curPathScores {}", curPathScores->debug());
#endif

        std::vector<size_t> batchTopKs(dimBatch, topK);             // [dimBatch, topK], topK=2*beamSize in transformer
        std::vector<unsigned int> outKeys;                          // [dimBatch * topK], record topK word indices
        std::vector<float> outPathScores;                           // [dimBatch * topK], record topK path scores

        /*
        cudaEvent_t start, stop;
        float elapsedTime = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        */

        /* curPathScores: [dimBatch, dimBeam*dimTgtWords] */
        // LOG(trace, "[TenTrans][HUBeamSearch] Search, curPathScores {}", curPathScores->debug());
        getNBestList(batchTopKs, curPathScores, outPathScores, outKeys, true);
        // this->memPool_->free(curPathScores->memory());
        /*
        std::cout << "topks" << std::endl;
        for(auto item: outKeys)
        {
            std::cout << item << " ";
        }
        std::cout << "\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<std::endl;

        std::cout << "topk values" << std::endl;
        for(auto item: outPathScores)
        {
            std::cout << item << " ";
        }
        std::cout << "\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<std::endl;

        LOG(trace, "[TenTrans][HUBeamSearch] Search, curPathScores {}", curPathScores->debug());
        std::vector<int> topKIds(dimBatch*topK, 0);
        std::vector<float> topKValues(dimBatch*topK, 0.f);
        HUTensorUtil::TopK_V2(curPathScores, topKIds, topKValues, topK, dimTgtWords);
        // HUTensorUtil::TopK(curPathScores, topKIds, topKValues, dimTgtWords);

        std::cout << "\n >>>>>>>>>>>>>>>>> new line >>>>>>>>>>>>>>>>>>>>>" <<std::endl;
        for(auto item: topKIds)
        {
            std::cout << item << " ";
        }
        std::cout << "\n >>>>>>>>>>>>>>>>> new line >>>>>>>>>>>>>>>>>>>>>" <<std::endl;

        for(auto item: topKValues)
        {
            std::cout << item << " ";
        }

        std::cout << "\n >>>>>>>>>>>>>>>>> new line >>>>>>>>>>>>>>>>>>>>>" <<std::endl;
        */

        this->memPool_->free(curPathScores->memory());
        /*
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Time Cost(ms): " << elapsedTime << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        */

        /*
        std::cout << "OutKey" << std::endl;
        for(int i=0; i< outKeys.size(); i++)
            std::cout << outKeys[i] << " ";
        std::cout << std::endl;

        std::cout << "outPathScores" << std::endl;
        for(int i=0; i< outPathScores.size(); i++)
            std::cout << float(outPathScores[i]) << " ";
        std::cout << std::endl;
        */

        Beams newBatchBeams(dimBatch);
        for (size_t sentId = 0; sentId < dimBatch; sentId++)
        {
            auto& beam = batchBeams[sentId];
            auto& newBeam = newBatchBeams[sentId];

            auto curSentBestScore = outPathScores[sentId * topK];
            isAllDone[sentId] = (isAllDone[sentId]) || (histories[sentId]->isDone(curSentBestScore));
            if (isAllDone[sentId])
            {
                auto hyp = HUNew<HUBeamCell>(beam[0], 0, PAD_ID, 0.f);
                newBeam.resize(beamSize_, hyp);
                // std::cout << "sentence " << sentId << " is finished .. " << std::endl;
                continue;
            }

            for (size_t i = 0; i < topK; i++)
            {
                int tmpDimBeam = 1;
                if (!isFirstStep) {
                    tmpDimBeam = beamSize_;
                }
                size_t beamIdx = (size_t)((outKeys[sentId * topK + i] - sentId * tmpDimBeam * dimTgtWords) / dimTgtWords);   // previous beam_index
                size_t embIdx = (size_t)(outKeys[sentId * topK + i] % dimTgtWords);                             // current word_index
                float pathScore = outPathScores[sentId * topK + i];
                // std::cout << "[BeamSearch sentId] "<< sentId << std::endl;
                // std::cout << "beamIdx: " << beamIdx << "\t" << "embIdx: " << embIdx << "\t" << "pathScore: " << pathScore << std::endl;
                if ((embIdx == trgEosId_) || (curStep+1 == maxStep)) {     // if is_finished, add finished-pool
                    histories[sentId]->AddFinished(beamIdx, pathScore);
                    // std::cout << "is finished... " << std::endl;
                }
                else
                {
                    auto hyp = HUNew<HUBeamCell>(beam[beamIdx], beamIdx, embIdx, pathScore);
                    newBeam.push_back(hyp);
                    if (newBeam.size() == beamSize_) {       
                        histories[sentId]->Add(newBeam, trgEosId_);
                        break;
                    }
                }
            }
        }

        // std::cout << "[Finished Step] " << curStep << std::endl;

        /* update batchBeams */
        /*
        for (auto& beam: newBatchBeams) {
            for (auto& item: beam) {
                std::cout << "embIdx: " << item->GetWord() << std::endl;
            }
        }

        for (size_t i = 0; i < dimBatch; i++)
        {
            for (size_t j = 0; j < beamSize_; j++)
            {
                auto hyp = newBatchBeams[i][j];
                std::cout << "previousIdx: " << hyp->GetPrevStateIndex() << "\twordIdx: " <<  hyp->GetWord() << "\tscore: " << hyp->GetPathScore() << std::endl;
            }
        }
       
        std::cout << "[BeamSearch Debug]" << std::endl;
        */
        batchBeams = newBatchBeams;

        isFirstStep = false;
        curStep += 1;

        bool isFinished = true;
        for (auto curSentIsDone: isAllDone) {
            if (curSentIsDone) {
                isFinished &= curSentIsDone;
            }
            else {
                isFinished = false;
                break;
            }
        }
        if (isFinished) {
            // this->memPool_->free(curPathScores->memory());
            // curState->Free();
            // curState->FinalFree();
            break;
        }

#ifdef DECODER_PADDING_OPTIMIZE
        for (int i = 0; i < isAllDone.size(); i++) {
            isAllDoneCopy[i] = (uint8_t)isAllDone[i];
        }
        cudaMemcpy(isAllDoneDevice, isAllDoneCopy.data(), isAllDoneCopy.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
#endif

    }
    curState->Free();
    curState->FinalFree();

#ifdef DECODER_PADDING_OPTIMIZE
    cudaFree(isAllDoneDevice);
#endif

    return histories;

}

/*
void HUBeamSearch::TopK(HUPtr<HUTensor> logProbs, const int K, HUPtr<HUTensor> topKIds, HUPtr<HUTensor> topKValues)
{

} */

}  // namespace TenTrans
