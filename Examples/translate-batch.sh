#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=1
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.1
source activate pt11-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
# source activate py13-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3

# ----- dir ------
# 1. fairseq BPE
model=models/en-de-v008
path_vocab_src=../lib/wmt17_en_de/wmt17_en_de/vocab.en
path_vocab_tgt=../lib/wmt17_en_de/wmt17_en_de/vocab.de
use_type='word'

# fname=eval_dev
# ftst=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.en
# seqlen=155
fname=eval_tst-COMMON
ftst=../lib/mustc-en-de-proc-fairseq/mustc/tst-COMMON/tst-COMMON.BPE.en
seqlen=175
# fname=eval_tst-HE
# ftst=../lib/mustc-en-de-proc-fairseq/mustc/tst-HE/tst-HE.BPE.en
# seqlen=120
# fname=eval_tst_wmt17
# ftst=../lib/wmt17_en_de/wmt17_en_de/test.en
# seqlen=105

# 2. [fairseq bpe on src; char on tgt]
# model=models/en-de-v007
# path_vocab_src=../lib/mustc-en-de-proc-fairseq/vocab.en
# path_vocab_tgt=../lib/mustc-en-de-proc-fairseq/vocab.de.char
# use_type='char'
#
# fname=eval_train_h1000
# ftst=../lib/mustc-en-de-proc-fairseq/mustc/train_h1000/train_h1000.BPE.en
# seqlen=855
# # fname=eval_dev
# # ftst=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.en
# # seqlen=665
# # fname=eval_tst-COMMON
# # ftst=../lib/mustc-en-de-proc-fairseq/mustc/tst-COMMON/tst-COMMON.BPE.en
# # seqlen=850
# # fname=eval_tst-HE
# # ftst=../lib/mustc-en-de-proc-fairseq/mustc/tst-HE/tst-HE.BPE.en
# # seqlen=570

# ----- models ------
batch_size=300

for i in `seq 1 1 19`
do
    echo $i
    ckpt=$i

    $PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/nmt-base/translate.py \
        --test_path_src $ftst \
        --test_path_tgt $ftst \
        --seqrev False \
        --path_vocab_src $path_vocab_src \
        --path_vocab_tgt $path_vocab_tgt \
        --use_type $use_type \
        --load $model/checkpoints_epoch/$ckpt \
        --test_path_out $model/$fname/$ckpt/ \
        --max_seq_len $seqlen \
        --batch_size $batch_size \
        --use_gpu True \
        --beam_width 1 \
        --eval_mode 1
done
