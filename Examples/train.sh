#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH
# export PATH=/home/mifs/ytl28/anaconda3/bin/:/usr/bin
# echo $PATH

export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.3
# source activate pt11-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
# source activate pt12-cuda10
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt12-cuda10/bin/python3
source activate py13-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
# ------------------------ DIR --------------------------
# [new clc]
savedir=models/gec-debug/
train_path_src=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/clc/train.src.nodot
train_path_tgt=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/clc/train.tgt.nodot
# dev_path_src=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/clc/dev.src
# dev_path_tgt=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/clc/dev.tgt
dev_path_src=None
dev_path_tgt=None
path_vocab_src=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/vocab/clctotal+swbd.min-count4.en
path_vocab_tgt=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/vocab/clctotal+swbd.min-count4.en
use_type='word'
share_embedder=True

# ------------------------ MODEL --------------------------
embedding_size_enc=512
embedding_size_dec=512
# load_embedding_src=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/embeddings/glove.6B.200d.txt
# load_embedding_tgt=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/embeddings/glove.6B.200d.txt
load_embedding_src='None'
load_embedding_tgt='None'

num_heads=8
dim_model=512
dim_feedforward=2048
enc_layers=6
dec_layers=6
transformer_type='standard' # standard | universal

# ------------------------ TRAIN --------------------------
# checkpoint_every=5
# print_every=2
checkpoint_every=5000
print_every=1000

batch_size=256
minibatch_split=2
max_seq_len=32
num_epochs=20

learning_rate_init=0.0001
learning_rate=0.0001
lr_warmup_steps=0 # total~?k

random_seed=2020
eval_with_mask=True
normalise_loss=True
max_count_no_improve=5
max_count_num_rollback=2
keep_num=2


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
	--share_embedder $share_embedder \
	--embedding_size_enc $embedding_size_enc \
	--embedding_size_dec $embedding_size_dec \
	--load_embedding_src $load_embedding_src \
	--load_embedding_tgt $load_embedding_tgt \
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
	--learning_rate_init $learning_rate_init \
	--lr_warmup_steps $lr_warmup_steps \
	--max_grad_norm 1.0 \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--max_count_no_improve $max_count_no_improve \
	--max_count_num_rollback $max_count_num_rollback \
	--keep_num $keep_num \
	--normalise_loss $normalise_loss \
	--minibatch_split $minibatch_split \
