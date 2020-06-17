#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD
echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.3
# source activate py13-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
source activate pt11-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3

# ------------------------ DIR --------------------------
# [mustc corpus]
# option 1 - fairseq bpe
# savedir=models/en-de-v008/
# train_path_src=../lib/mustc-en-de-proc-fairseq/mustc/train/train.BPE.en
# train_path_tgt=../lib/mustc-en-de-proc-fairseq/mustc/train/train.BPE.de
# # dev_path_src=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.en
# # dev_path_tgt=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.de
# dev_path_src=None
# dev_path_tgt=None
# path_vocab_src=../lib/wmt17_en_de/wmt17_en_de/vocab.en
# path_vocab_tgt=../lib/wmt17_en_de/wmt17_en_de/vocab.de
# load_embedding_src=None
# load_embedding_tgt=None
# use_type='word'

# option 2 - fairseq bpe on src; char on tgt
# savedir=models/en-de-v009/
# train_path_src=../lib/mustc-en-de-proc-fairseq/mustc/train/train.BPE.en
# train_path_tgt=../lib/mustc-en-de/train/txt/train.de
# # dev_path_src=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.en
# # dev_path_tgt=../lib/mustc-en-de/dev/txt/dev.de
# dev_path_src=None
# dev_path_tgt=None
# path_vocab_src=../lib/mustc-en-de-proc-fairseq/vocab.en
# path_vocab_tgt=../lib/mustc-en-de-proc-fairseq/vocab.de.char
# load_embedding_src=None
# load_embedding_tgt=None
# use_type='char'

# option 3 - fairseq bpe on src; tokenised + char on tgt
savedir=models/en-de-v011/
train_path_src=../lib/mustc-en-de-proc-fairseq/mustc/train/train.BPE.en
train_path_tgt=../lib/mustc-en-de-proc-fairseq/mustc-prep/train/train.de
# dev_path_src=../lib/mustc-en-de-proc-fairseq/mustc/dev/dev.BPE.en
# dev_path_tgt=../lib/mustc-en-de/dev/txt/dev.de
dev_path_src=None
dev_path_tgt=None
path_vocab_src=../lib/mustc-en-de-proc-fairseq/vocab.en
path_vocab_tgt=../lib/mustc-en-de-proc-fairseq/vocab.de.char
load_embedding_src=None
load_embedding_tgt=None
use_type='char'


# [wmt2017 corpus]
# savedir=models/en-de-v021/
# train_path_src=../lib/wmt17_en_de/wmt17_en_de/train.en
# train_path_tgt=../lib/wmt17_en_de/wmt17_en_de/train.de
# # dev_path_src=../lib/wmt17_en_de/wmt17_en_de/valid.en
# # dev_path_tgt=../lib/wmt17_en_de/wmt17_en_de/valid.de
# dev_path_src=None
# dev_path_tgt=None
# path_vocab_src=../lib/wmt17_en_de/wmt17_en_de/vocab.en
# path_vocab_tgt=../lib/wmt17_en_de/wmt17_en_de/vocab.de
# load_embedding_src=None
# load_embedding_tgt=None
# use_type='word'

# ------------------------ MODEL --------------------------
embedding_size_enc=300
embedding_size_dec=300
hidden_size_enc=300
hidden_size_dec=300
hidden_size_shared=300
num_bilstm_enc=3
num_unilstm_dec=4
att_mode=bilinear # bahdanau | bilinear

# ------------------------ TRAIN --------------------------
# checkpoint_every=5
# print_every=2
checkpoint_every=1000
print_every=200

batch_size=512
minibatch_split=4
max_seq_len=250
num_epochs=50

random_seed=300
eval_with_mask=True
max_count_no_improve=5
max_count_num_rollback=2
keep_num=2
normalise_loss=True

$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/nmt-base/train.py \
	--train_path_src $train_path_src \
	--train_path_tgt $train_path_tgt \
	--dev_path_src $dev_path_src \
	--dev_path_tgt $dev_path_tgt \
	--path_vocab_src $path_vocab_src \
	--path_vocab_tgt $path_vocab_tgt \
	--load_embedding_src $load_embedding_src \
	--load_embedding_tgt $load_embedding_tgt \
	--use_type $use_type \
	--save $savedir \
	--random_seed $random_seed \
	--embedding_size_enc $embedding_size_enc \
	--embedding_size_dec $embedding_size_dec \
	--hidden_size_enc $hidden_size_enc \
	--num_bilstm_enc $num_bilstm_enc \
	--num_unilstm_enc 0 \
	--hidden_size_dec $hidden_size_dec \
	--num_unilstm_dec $num_unilstm_dec \
	--hidden_size_att 10 \
	--att_mode $att_mode \
	--residual True \
	--hidden_size_shared $hidden_size_shared \
	--max_seq_len $max_seq_len \
	--batch_size $batch_size \
	--batch_first True \
	--seqrev False \
	--eval_with_mask $eval_with_mask \
	--scheduled_sampling False \
	--teacher_forcing_ratio 1.0 \
	--dropout 0.2 \
	--embedding_dropout 0.0 \
	--num_epochs $num_epochs \
	--use_gpu True \
	--learning_rate 0.001 \
	--max_grad_norm 1.0 \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--max_count_no_improve $max_count_no_improve \
	--max_count_num_rollback $max_count_num_rollback \
	--keep_num $keep_num \
	--normalise_loss $normalise_loss \
	--minibatch_split $minibatch_split \
