#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.3
source activate py13-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
# source activate pt11-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
# source activate pt12-cuda10
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt12-cuda10/bin/python3

# ------------------------ DIR --------------------------
# [mustc corpus]
# option 1 - huggingface bpe
# savedir=models/en-de-v001/
# train_path_src=../lib/mustc-en-de-proc/train/train.en.bpe30000
# train_path_tgt=../lib/mustc-en-de-proc/train/train.de.bpe30000
# # dev_path_src=../lib/mustc-en-de-proc/dev/dev.en.bpe30000
# # dev_path_tgt=../lib/mustc-en-de-proc/dev/dev.de.bpe30000
# dev_path_src=None
# dev_path_tgt=None
# path_vocab_src=../lib/mustc-en-de-proc/vocab/en-bpe-30000/vocab
# path_vocab_tgt=../lib/mustc-en-de-proc/vocab/de-bpe-30000/vocab
# use_type='word'

# option 2 - fairseq bpe
savedir=models/en-de-debug/
train_path_src=../lib/mustc-en-de-proc-fairseq/mustc/train/train.BPE.en
train_path_tgt=../lib/mustc-en-de-proc-fairseq/mustc/train/train.BPE.de
# dev_path_src=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.en
# dev_path_tgt=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.de
dev_path_src=None
dev_path_tgt=None
path_vocab_src=../lib/iwslt17_en_de/wmt17_en_de/vocab.en
path_vocab_tgt=../lib/iwslt17_en_de/wmt17_en_de/vocab.de
use_type='word'

# [iwslt2017 corpus]
# savedir=models/en-de-v011/
# train_path_src=../lib/iwslt17_en_de/wmt17_en_de/train.en
# train_path_tgt=../lib/iwslt17_en_de/wmt17_en_de/train.de
# # dev_path_src=../lib/iwslt17_en_de/wmt17_en_de/valid.en
# # dev_path_tgt=../lib/iwslt17_en_de/wmt17_en_de/valid.de
# dev_path_src=None
# dev_path_tgt=None
# path_vocab_src=../lib/iwslt17_en_de/wmt17_en_de/vocab.en
# path_vocab_tgt=../lib/iwslt17_en_de/wmt17_en_de/vocab.de
# use_type='word'

# ------------------------ MODEL --------------------------
embedding_size_enc=300
embedding_size_dec=300
num_heads=8
dim_model=512
dim_feedforward=1024
enc_layers=6
dec_layers=6
transformer_type='standard' # standard | universal

# ------------------------ TRAIN --------------------------
# checkpoint_every=5
# print_every=2
checkpoint_every=1000
print_every=200

batch_size=200
max_seq_len=50
num_epochs=50
random_seed=300
eval_with_mask=True
max_count_no_improve=5
max_count_num_rollback=2
keep_num=2
normalise_loss=True
learning_rate=0.0001

$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/nmt-transformer/train.py \
	--train_path_src $train_path_src \
	--train_path_tgt $train_path_tgt \
	--dev_path_src $dev_path_src \
	--dev_path_tgt $dev_path_tgt \
	--path_vocab_src $path_vocab_src \
	--path_vocab_tgt $path_vocab_tgt \
	--use_type $use_type \
	--save $savedir \
	--random_seed $random_seed \
	--embedding_size_enc $embedding_size_enc \
	--embedding_size_dec $embedding_size_dec \
	--num_heads $num_heads \
	--dim_model $dim_model \
	--dim_feedforward $dim_feedforward \
	--enc_layers $enc_layers \
	--dec_layers $dec_layers \
	--transformer_type $transformer_type \
	--max_seq_len $max_seq_len \
	--batch_size $batch_size \
	--seqrev False \
	--eval_with_mask $eval_with_mask \
	--dropout 0.2 \
	--embedding_dropout 0.0 \
	--num_epochs $num_epochs \
	--use_gpu True \
	--learning_rate $learning_rate \
	--max_grad_norm 1.0 \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--max_count_no_improve $max_count_no_improve \
	--max_count_num_rollback $max_count_num_rollback \
	--keep_num $keep_num \
	--normalise_loss $normalise_loss \
