#pragma once
#include <iostream>
#include <cmath>
#include <queue>
#include "HUBeamCell.h"

namespace TenTrans
{

class HUHistory
{

private:
	/* one hypothesis of a full sentence (reference into search grid) */
	struct SentenceHypothesisCoord 
    {
    	bool operator<(const SentenceHypothesisCoord& hc) const { return normalizedPathScore < hc.normalizedPathScore; }

    	size_t i;                        // decoding time-steps
    	size_t j;                        // beamId
    	float normalizedPathScore;       // length-normalized sentence score
	};

public:
	HUHistory(const size_t lineNo, const size_t maxLen, const size_t beamSize, bool earlyStop, const float alpha=1.f, float wp=0.f, float worstScore=10000.f)
        :lineNo_(lineNo), maxLen_(maxLen), beamSize_(beamSize), earlyStop_(earlyStop), alpha_(alpha), wp_(wp), worstScore_(worstScore)  {}

	float LengthPenalty(size_t length) { return std::pow((float)length, alpha_); }   // length penalty
	float WordPenalty(size_t length) { return wp_ * (float)length; }                 // word penalty
	size_t size() const { return history_.size(); }                                  // current time-steps

    void AddFinished(int beamId, float pathScore) 
    {
        float normalizedPathScore = (pathScore - WordPenalty(history_.size())) / LengthPenalty(history_.size());
        if ((topHyps_.size() < beamSize_) || (normalizedPathScore > worstScore_))
        {
            topHyps_.push({history_.size(), beamId, normalizedPathScore});
            if (topHyps_.size() > beamSize_)                                         // keep beam_size finished hypo, update worstScore
            {
                std::vector<SentenceHypothesisCoord> tmpPriorityQueue;
                while(!topHyps_.empty()) 
                {
                    auto hypCoord = topHyps_.top();
                    tmpPriorityQueue.push_back({hypCoord.i, hypCoord.j, hypCoord.normalizedPathScore});
                    // std::cout << hypCoord.i << "\t" << hypCoord.j << "\t" << hypCoord.normalizedPathScore << std::endl;
                    topHyps_.pop();
                } 
                // std::cout << "[toHyps Size]: " << topHyps_.size() << std::endl;
                // std::cout << "[tmpPriorityQueue Size]: " << tmpPriorityQueue.size() << std::endl;

                worstScore_ = tmpPriorityQueue[beamSize_-1].normalizedPathScore;
                // std::cout << worstScore_ << "\t" << tmpPriorityQueue[beamSize_-1].normalizedPathScore << std::endl; 
                for (size_t i = 0; i < beamSize_; i++) {
                    topHyps_.push({tmpPriorityQueue[i].i, tmpPriorityQueue[i].j, tmpPriorityQueue[i].normalizedPathScore});
                    // std::cout << "copyFrom: " << tmpPriorityQueue[i].i << "\t" << tmpPriorityQueue[i].j << "\t" << tmpPriorityQueue[i].normalizedPathScore << std::endl;
                }
                // std::cout << "[PriorityQueue Size]: " << topHyps_.size() << std::endl;
            }
            else
            {
                // std::cout << "[go through] ..." << std::endl;
                if (normalizedPathScore < worstScore_) {
                    // std::cout << "[go through] ..." << std::endl;
                    worstScore_ = normalizedPathScore;
                }
            }
        }
        // std::cout << "normalizedPathScore: " << normalizedPathScore << "\tworstScore: " << worstScore_ << std::endl;
    }

    bool isDone(float curBestPathScore) 
    {
        // std::cout << "[Done]:\t" << "curScore: " << curBestPathScore << "\tworstScore: " << worstScore_ << std::endl;

        // early-stop strategy 
        if (this->earlyStop_) 
        { 
            if (topHyps_.size() >= 1) 
            { 
                auto hypCoord = topHyps_.top();
                if (hypCoord.j == 0) 
                { 
                    return true;
                }
            }
        }

        if (topHyps_.size() < beamSize_) {   // finished hypos < beam_size, continue BeamSearch
            return false;
        }

        // std::cout << "[Done]:\t" << "curScore: " << curBestNormalizedPathScore << "\tworstScore: " << worstScore_ << std::endl;  
        // if worstScore >= curBestNormalizedPathScore, stop BeamSearch
        float curBestNormalizedPathScore = curBestPathScore / LengthPenalty(maxLen_);
        // std::cout << "maxLen: " << maxLen_ << "\tcurNormalizedScore: " << curBestNormalizedPathScore << std::endl;
        return worstScore_ >= curBestNormalizedPathScore;
    }

	void Add(const Beam& beam, int trgEosId, bool last=false) { history_.push_back(beam); }

	NBestList NBest(size_t n) const 
    {
    	NBestList nbest;
        // std::cout << "[topHyps_ size]: " << topHyps_.size() << std::endl;
    	for (auto topHypsCopy = topHyps_; nbest.size() < n && !topHypsCopy.empty(); topHypsCopy.pop()) 
        {
      		auto bestHypCoord = topHypsCopy.top();

      		const size_t start = bestHypCoord.i;                // last time step of this hypothesis
      		const size_t j = bestHypCoord.j;                    // which beam entry
      		HUPtr<HUBeamCell> bestHyp = history_[start-1][j];   // start from 0
    		float c = bestHypCoord.normalizedPathScore;
    		// std::cerr << "setps: " << start << "\tbeam: " << j << "\tscore" << c << std::endl;

      		// trace back best path
      		std::vector<int> targetWords = bestHyp->TracebackWords();
            // std::cout << "words: " << targetWords.size() << std::endl;

      		// note: bestHyp->GetPathScore() is not normalized, while bestHypCoord.normalizedPathScore is
      		nbest.emplace_back(targetWords, bestHyp, bestHypCoord.normalizedPathScore);
    	}
    	return nbest;
	}
	
	Result Top() const { return NBest(beamSize_)[0]; }	
	size_t GetLineNum() const { return lineNo_; }	

private:
	std::vector<Beam> history_;                                   // [timeSteps, beamSize] of HUPtr<HUBeamCell>
	std::priority_queue<SentenceHypothesisCoord> topHyps_;        // all sentence hypotheses (those that reached eos), sorted by score
	const size_t lineNo_;                                         // sentenceId
    const size_t beamSize_;                                       // beam-size
	const float alpha_;                                           // for length penalty
	const float wp_;                                              // for word penalty
    const float maxLen_;                                          // (srcLen+50-1), for length penalty                                    
    float worstScore_;
    bool earlyStop_;                                              // early-stop for beam search
};

typedef std::vector<HUPtr<HUHistory>> HUHistories;

}
