#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD
echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=1
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.1
# source activate py13-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
source activate pt12-cuda10
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt12-cuda10/bin/python3

# ----- dir ------
path_vocab_src=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/vocab/clctotal+swbd.min-count4.en
path_vocab_tgt=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/vocab/clctotal+swbd.min-count4.en
use_type='word'

libbase=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib-bpe

# fname=test_fce_test
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/clc/test.src
# seqlen=100

# fname=test_clc
# ftst=$libbase/clc/nobpe/clc-test.src
# seqlen=125

# fname=test_nict
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-nict/nict.src
# seqlen=85

# fname=test_nict_new
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-nict-new/nict.src
# seqlen=85

# fname=test_dtal
# ftst=$libbase/dtal/nobpe/dtal.src
# seqlen=165

fname=test_dtal_asr # default seg manual
ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-dtal-asr/dtal-asr.src
seqlen=165

# ----- models ------
# export ckpt=$1
model=models/gec-v031
ckpt=19
beam_width=1
batch_size=50
use_gpu=True

$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/nmt-transformer/translate.py \
    --test_path_src $ftst \
    --seqrev False \
    --path_vocab_src $path_vocab_src \
    --path_vocab_tgt $path_vocab_tgt \
    --use_type $use_type \
    --load $model/checkpoints_epoch/$ckpt \
    --test_path_out $model/$fname/$ckpt/ \
    --max_seq_len $seqlen \
    --batch_size $batch_size \
    --use_gpu $use_gpu \
    --beam_width $beam_width \
    --eval_mode 1
