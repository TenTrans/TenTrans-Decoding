## Requirements

- CMake = 3.11

- Zlib

- CUDA >= 8.0 or newer version

- Python 3 is recommended because some features are not supported in python 2

- PyTorch >= 1.4.0

- gcc = 4.8.5

  

## Quick Start Guide

### 1. Training Transformer Model

- Using TenTrans Training Platform to get transformer model.

### 2. Convert Model

```shell
cd TenTrans-Decoding/tools
sh run_convert.sh
```

- run_convert.sh

```shell
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
```

### 3. Configuration

- TenTrans-Decoding/conf/config.yml

```yaml
models:
    - ../model/checkpoint_seq2seq_wmtende_mt_best.npz
vocabs:
    - ../model/vocab.src.yml
    - ../model/vocab.tgt.yml
enc-depth: 6
dec-depth: 6
dim-emb: 512
transformer-dim-ffn: 2048
transformer-heads: 8
share-all-embed: True                  ## whether share source embedding and target embedding
share-out-embed: True                  ## whether share target embedding and project embedding
transformer-ffn-activation: relu
normalize-before: True                 ## whether pre-norm or post-norm
learned-pos: True
max-seq-length: 512
use-emb-scale: False
transformer-ffn-depth: 1
normalize: 0.6
decode-length: 50
n-best: True

devices:
    - 0
mini-batch: 16
beam-size: 4

#trace - debug - info - warn - err(or) - critical - off
log-level: trace
```

#### 4. Compile & Run

```shell
cd TenTrans-Decoding/build
sh compile.sh
sh run.sh
```










