model=$1
convertModel=$2
srcVocab=$3
convertSrcVocab=$4
tgtVocab=$5
convertTgtVocab=$6

# 1. extract model params as .npz file   note: python need install torch
CUDA_VISIBLE_DEVICES=0 python load_torch_model.py $model $convertModel

# 2. format source vocabulary as .yml file
python get_vocab.py $srcVocab $convertSrcVocab

# 3. format target vocabulary as .yml file
python get_vocab.py $tgtVocab $convertTgtVocab

