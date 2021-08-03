#pragma once
#include "HUGlobal.h"
#include <vector>
#include <tuple>

namespace TenTrans
{
// This code draws on memory management part of Marian project
class HUBeamCell
{
public:
	HUBeamCell(): prevHyp_(nullptr), prevIndex_(0), word_(PAD_ID), pathScore_(0.0f) {}
    HUBeamCell(const HUPtr<HUBeamCell> prevHyp, const int prevIndex, const int word, const float pathScore): 
        prevHyp_(prevHyp), prevIndex_(prevIndex), word_(word), pathScore_(pathScore) {}

	const HUPtr<HUBeamCell> GetPrevHyp() const { return prevHyp_; }
    int GetPrevStateIndex() const { return prevIndex_; }
	int GetWord() const { return word_; }
	float GetPathScore() const { return pathScore_; }
	std::vector<float>& GetScoreBreakdown() { return scoreBreakdown_; }

	/* Trace back paths referenced from this hypothesis */
  	std::vector<int> TracebackWords()
  	{
      std::vector<int> targetWords;
      for (auto hyp = this; hyp->GetPrevHyp(); hyp = hyp->GetPrevHyp().get()) {
          targetWords.push_back(hyp->GetWord());
      }
      std::reverse(targetWords.begin(), targetWords.end());
      return targetWords;
    }

private:
	const HUPtr<HUBeamCell> prevHyp_;                                     // previous hypothesis
	const int prevIndex_;                                                 // for updating cache
	const int word_;                                                      // current word ID
	const float pathScore_;                                               // path score
	std::vector<float> scoreBreakdown_;

};

typedef std::vector<HUPtr<HUBeamCell>> Beam;                              // [beamSize] of HUPtr<HUBeamCell>  
typedef	std::vector<Beam> Beams;                                          // [batchSize, beamSize] of HUPtr<HUBeamCell>    
typedef std::tuple<std::vector<int>, HUPtr<HUBeamCell>, float> Result;    // <hyp.wordId, hyp, hyp.normalizedScore>
typedef std::vector<Result> NBestList;                                    // Nbest of Result

}
