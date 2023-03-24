#include "HUData.h"
#include<iostream>
//using namespace std;

namespace TenTrans{

void HUBatch::Debug()
{
	if(!sentenceIds_.empty()) {
      std::cerr << "sentence ID: ";
      for(auto id : sentenceIds_) {
          std::cerr << id << " ";
      }
      std::cerr << std::endl;
    }

    std::cerr << "sentence words: ";
    const auto& vocab = this->vocab();
    for (size_t i = 0; i < this->batchSize(); i++) 
    {
        for (size_t j = 0; j < this->batchWidth(); j++)
        {
            size_t idx = i * this->batchWidth() + j;
            size_t w = this->data()[idx];
            if (vocab) {
                std::cerr << (*vocab)[w] << " ";
            }
            else {
                std::cerr << w << " "; // if not loaded then print numeric id instead
            }
            
        }
        std::cerr << std::endl;
    }

    std::cerr << "mask ids: ";
    for (size_t i = 0; i < this->batchSize(); i++)
    {
        for (size_t j = 0; j < this->batchWidth(); j++)
        {
            size_t idx = i * this->batchWidth() + j;
            std::cerr << this->mask()[idx] << " ";

        }
        std::cerr << std::endl;
    } 

    /*
	const auto& vocab = this->vocab();
    for(size_t i = 0; i < this->batchWidth(); i++) {
    	std::cerr << "\t w: ";
    	for(size_t j = 0; j < this->batchSize(); j++) {
        	size_t idx = i * this->batchSize() + j;
        	size_t w = this->data()[idx];
        	if (vocab)
        		std::cerr << (*vocab)[w] << " ";
          	else
            	std::cerr << w << " "; // if not loaded then print numeric id instead
        }
		std::cerr << "\n mask:";
		for(size_t j = 0; j < this->batchSize(); j++)
		{
			size_t idx = i * this->batchSize() + j;
            std::cerr << this->mask()[idx] << " ";
		}
    	std::cerr << std::endl;
    } 
    */
	
}

HUTextInput::HUTextInput(std::vector<std::string> sources, HUPtr<HUVocab> vocab)
{
	this->sources_ = sources;
	this->vocab_ = vocab;
}


/*
std::vector<HUSentence> HUTextInput::ToSents()
{
	std::vector<HUSentence> batchVector;
	for(int i=0; i < sources_.size(); i++)
	{
		std::vector<size_t> ids = this->vocab_->Encode(sources_[i]);
		batchVector.push_back(*(new HUSentence(i, ids)));
        // HUPtr<HUSentence> cur_sent = HUNew<HUSentence>(i, ids);  
        // batchVector.push_back(cur_sent);
	}
	return batchVector;
} 
*/

std::vector<HUPtr<HUSentence>> HUTextInput::ToSents()
{
    std::vector<HUPtr<HUSentence>> batchVector;
    for(int i=0; i < sources_.size(); i++)
    {
        std::vector<size_t> ids = this->vocab_->Encode(sources_[i]);
        // batchVector.push_back(*(new HUSentence(i, ids)));
        HUPtr<HUSentence> cur_sent = HUNew<HUSentence>(i, ids);  
        batchVector.push_back(cur_sent);
    }
    return batchVector;
} 


HUPtr<HUBatch> HUTextInput::ToBatch(const std::vector<HUPtr<HUSentence>>& batchVector)
{
    int maxDim = 0;
	size_t batchSize = batchVector.size(); 
    std::vector<size_t> sentenceIds; 
    for(auto& sent : batchVector)
    {   
        if(sent->Size() > maxDim) {
            //// maxDim = sent.Size();
            maxDim = sent->Size();
        }
        //// sentenceIds.push_back(sent.GetId());
        sentenceIds.push_back(sent->GetId());
    }   

    // avoid long sentences
    maxDim = maxDim > MAX_SOURCE_TOKENS ? MAX_SOURCE_TOKENS : maxDim;

    int count = 0;
    std::vector<float> lengths;
    HUPtr<HUBatch> batch = HUNew<HUBatch>(batchSize, maxDim, vocab_);
    for(int i = 0; i < batchSize; i++)
    {   
        float cur_length = 0.f;
        //// int traverse_length = batchVector[i].Size();
        int traverse_length = batchVector[i]->Size();
        if (traverse_length > maxDim) {
            traverse_length = maxDim;
        }

        for(int k = 0; k < traverse_length; k++)
        {   
            //// batch->data()[i*maxDim+k] = batchVector[i][k];
            batch->data()[i*maxDim+k] = (*(batchVector[i]))[k];
            batch->mask()[i*maxDim+k] = (TT_DATA_TYPE)1.f;
            cur_length += 1;
            count++;
        }   
        lengths.push_back(cur_length);
    }   
    batch->setWords(count);
    batch->setLengths(lengths);
    batch->setSentenceIds(sentenceIds);

    return batch;
}

}
