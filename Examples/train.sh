#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
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
# savedir=models/en-de-debug/
# train_path_src=../lib/mustc-en-de-proc-fairseq/mustc/train/train.BPE.en
# train_path_tgt=../lib/mustc-en-de-proc-fairseq/mustc/train/train.BPE.de
# # dev_path_src=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.en
# # dev_path_tgt=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.de
# dev_path_src=None
# dev_path_tgt=None
# path_vocab_src=../lib/iwslt17_en_de/wmt17_en_de/vocab.en
# path_vocab_tgt=../lib/iwslt17_en_de/wmt17_en_de/vocab.de
# use_type='word'

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

# option 4 - fairseq bpe on src; char on tgt
# loaddir=models/en-de-v004/checkpoints_epoch/16
loaddir='None'
savedir=models/en-de-v005/
train_path_src=../lib/mustc-en-de-proc-fairseq/mustc/train/train.BPE.en
train_path_tgt=../lib/mustc-en-de/train/txt/train.de
# dev_path_src=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.en
# dev_path_tgt=../lib/mustc-en-de/dev/txt/dev.de
dev_path_src=None
dev_path_tgt=None
path_vocab_src=../lib/mustc-en-de-proc-fairseq/vocab.en
path_vocab_tgt=../lib/mustc-en-de-proc-fairseq/vocab.de.char
load_embedding_src=None
load_embedding_tgt=None
use_type='char'

# ------------------------ MODEL --------------------------
embedding_size_enc=512
embedding_size_dec=512
num_heads=8
dim_model=512
dim_feedforward=2048
enc_layers=6
dec_layers=6
transformer_type='standard' # standard | universal

# ------------------------ TRAIN --------------------------
# checkpoint_every=5
# print_every=2
checkpoint_every=1000
print_every=200

learning_rate_init=0.0001
learning_rate=0.2
lr_warmup_steps=16000

batch_size=256
minibatch_split=4
max_seq_len=250
num_epochs=50

random_seed=300
eval_with_mask=True
max_count_no_improve=5
max_count_num_rollback=2
keep_num=2
normalise_loss=True

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
	--load $loaddir \
