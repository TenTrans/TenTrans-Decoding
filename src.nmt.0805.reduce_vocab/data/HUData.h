#pragma once 
#include <iostream>
#include <map>
#include <vector>
#include "cnpy.h"
#include "HUGlobal.h"
#include "HUVocab.h"
#include "HUConfig.h"
//using namespace std;

namespace TenTrans{

class HUSentence{
private:
	size_t id_;
	std::vector<size_t> words_;

public:
	HUSentence(size_t id, std::vector<size_t> words)
	{
		this->id_ = id;
		this->words_ = words;
	}
	size_t GetId() const { return  id_; }
	size_t Size() const { return words_.size(); }
	const size_t operator[](size_t i) const { return words_[i]; }
};

class HUBatch{
private:
	std::vector<size_t> indices_;              // [batch_size * max_seq_len]
	std::vector<TT_DATA_TYPE> mask_;           // [batch_size * max_seq_len]
	size_t size_;                              // batch_size
	size_t width_;                             // max_seq_len
	size_t words_;                             // The total number of words in the batch, considering the mask.
	HUPtr<HUVocab> vocab_;
	HUPtr<HUConfig> config_;
	std::vector<size_t> sentenceIds_;
    std::vector<float> lengths_;               // batch_size, record length of each sentence.
    std::vector<size_t> newVocabIndices_;      // new indices for vocabulary selection.
	
public:
	HUBatch(size_t size, size_t width, const HUPtr<HUVocab>& vocab)
      : indices_(size * width, PAD_ID),
        mask_(size * width, (TT_DATA_TYPE)0.f),
        size_(size),
        width_(width),
        words_(0),
        vocab_(vocab) {}
	
	std::vector<size_t>& data() { return indices_; }
	std::vector<TT_DATA_TYPE>& mask() { return mask_; }
    std::vector<float>& lengths() { return lengths_; }
	const HUPtr<HUVocab>& vocab() const { return vocab_; }
    std::vector<size_t>& newVocab() { return newVocabIndices_; }

	size_t batchSize() { return size_; }
	size_t batchWidth() { return width_; };
	size_t batchWords() { return words_; }
	
	void setWords(size_t words) { words_ = words; }
	void setSentenceIds(const std::vector<size_t>& ids) { sentenceIds_ = ids; }
    void setLengths(const std::vector<float>& lengths) { lengths_ = lengths; }
	const std::vector<size_t>& getSentenceIds() const { return sentenceIds_; }

    void setNewVocabs(const std::vector<size_t>& ids) { newVocabIndices_ = ids; }

	void Debug();
};

class HUTextInput{
private:
	std::vector<std::string> sources_;
	HUPtr<HUVocab> vocab_;
    // cnpy::npz_t shortListNpz_;

public:
	HUTextInput(std::vector<std::string> sources, HUPtr<HUVocab> vocab/*, cnpy::npz_t shortListNpz*/);
	std::vector<HUPtr<HUSentence>> ToSents();
	HUPtr<HUBatch> ToBatch(const std::vector<HUPtr<HUSentence>>& batchVector, const std::map<size_t, std::vector<size_t>>& shortList);

};



}
