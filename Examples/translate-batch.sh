#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=2
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.1
# source activate pt11-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
source activate py13-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3

# ----- dir ------
# 1. BPE
model=models/en-de-v001
path_vocab_src=../lib/mustc-en-de-proc/vocab/en-bpe-30000/vocab
path_vocab_tgt=../lib/mustc-en-de-proc/vocab/de-bpe-30000/vocab
use_type='word'

# fname=eval_train_h1000
# ftst=../lib/mustc-en-de-proc/train/train_h1000.en.bpe50000
# seqlen=170
fname=eval_dev
ftst=../lib/mustc-en-de-proc/dev/dev.en.bpe30000
seqlen=120
# fname=eval_tst-COMMON
# ftst=../lib/mustc-en-de-proc/tst-COMMON/tst-COMMON.en.bpe50000
# seqlen=145
# fname=eval_tst-HE
# ftst=../lib/mustc-en-de-proc/tst-HE/tst-HE.en.bpe50000
# seqlen=100

# ----- models ------

for i in `seq 1 1 26`
do
    echo $i
    ckpt=$i

    $PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/nmt-transformer/translate.py \
        --test_path_src $ftst \
        --test_path_tgt $ftst \
        --seqrev False \
        --path_vocab_src $path_vocab_src \
        --path_vocab_tgt $path_vocab_tgt \
        --use_type $use_type \
        --load $model/checkpoints_epoch/$ckpt \
        --test_path_out $model/$fname/$ckpt/ \
        --max_seq_len $seqlen \
        --batch_size 50 \
        --use_gpu True \
        --beam_width 1 \
        --eval_mode 1
done
