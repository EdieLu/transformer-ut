import torch
import random
import time
import os
import argparse
import sys
import numpy as np

from utils.misc import set_global_seeds, save_config, validate_config, check_device
from utils.dataset import Dataset
from models.Seq2seq import Seq2seq
from trainer.trainer import Trainer


def load_arguments(parser):

	""" Seq2seq model """

	# paths
	parser.add_argument('--train_path_src', type=str, required=True, help='train src dir')
	parser.add_argument('--train_path_tgt', type=str, required=True, help='train tgt dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, required=True, help='vocab tgt dir')
	parser.add_argument('--dev_path_src', type=str, default=None, help='dev src dir')
	parser.add_argument('--dev_path_tgt', type=str, default=None, help='dev tgt dir')
	parser.add_argument('--load_embedding_src', type=str, default=None, help='init src embedding')
	parser.add_argument('--load_embedding_tgt', type=str, default=None, help='init tgt embedding')
	parser.add_argument('--data_ratio', type=float, default=1.0, help='data partition being used')

	parser.add_argument('--save', type=str, required=True, help='model save dir')
	parser.add_argument('--load', type=str, default=None, help='model load dir')
	parser.add_argument('--load_mode', type=str, default='null', help='loading mode resume|restart|null')
	parser.add_argument('--use_type', type=str, default='word', help='word | char')

	# model
	parser.add_argument('--share_embedder', type=str, default='False', help='share embedder or not')
	parser.add_argument('--embedding_size_enc', type=int, default=200, help='encoder embedding size')
	parser.add_argument('--embedding_size_dec', type=int, default=200, help='decoder embedding size')
	parser.add_argument('--num_heads', type=int, default=8, help='multi head attention')
	parser.add_argument('--dim_model', type=int, default=512, help='dim_model')
	parser.add_argument('--dim_feedforward', type=int, default=1024, help='dim_feedforward')
	parser.add_argument('--hidden_size_dec', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--enc_layers', type=int, default=6, help='number of encoder layers')
	parser.add_argument('--dec_layers', type=int, default=6, help='number of decoder layers')
	parser.add_argument('--transformer_type', type=str, default='standard', help='universal | standard')
	parser.add_argument('--act', type=str, default='False', help='universal transformer, dynamic hault')

	# data
	parser.add_argument('--eval_with_mask', type=str, default='True', help='calc loss excluding padded words')
	parser.add_argument('--embedding_dropout', type=float, default=0.0, help='embedding dropout')
	parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
	parser.add_argument('--seqrev', type=str, default='False', help='reverse src, tgt sequence')

	# train
	parser.add_argument('--random_seed', type=int, default=666, help='random seed')
	parser.add_argument('--gpu_id', type=int, default=0, help='only used for memory reservation')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--minibatch_split', type=int, default=1, help='split the batch to avoid OOM')
	parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
	parser.add_argument('--learning_rate_init', type=float, default=0.0005, help='learning rate init')
	parser.add_argument('--lr_warmup_steps', type=int, default=12000, help='lr warmup steps')
	parser.add_argument('--normalise_loss', type=str, default='True', help='normalise loss or not')
	parser.add_argument('--max_grad_norm', type=float, default=1.0,
		help='optimiser gradient norm clipping: max grad norm')

	# save and print
	parser.add_argument('--grab_memory', type=str, default='True', help='grab full GPU memory')
	parser.add_argument('--checkpoint_every', type=int, default=10, help='save ckpt every n steps')
	parser.add_argument('--print_every', type=int, default=10, help='print every n steps')
	parser.add_argument('--eval_mode', type=str, default='tf',
		help='fr | tf (free running or teacher forcing)')
	parser.add_argument('--eval_metric', type=str, default='tokacc',
		help='tokacc | bleu (token-level accuracy or word-level bleu)')
	parser.add_argument('--max_count_no_improve', type=int, default=2,
		help='if meet max, operate roll back')
	parser.add_argument('--max_count_num_rollback', type=int, default=2,
		help='if meet max, reduce learning rate')
	parser.add_argument('--keep_num', type=int, default=1,
		help='number of models to keep')

	return parser


def main():

	# load config
	parser = argparse.ArgumentParser(description='Seq2seq Training')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# set random seed
	if config['random_seed'] is not None:
		set_global_seeds(config['random_seed'])

	# record config
	if not os.path.isabs(config['save']):
		config_save_dir = os.path.join(os.getcwd(), config['save'])
	if not os.path.exists(config['save']):
		os.makedirs(config['save'])

	# loading old models
	if config['load']:
		print('loading {} ...'.format(config['load']))
		config_save_dir = os.path.join(config['save'], 'model-cont.cfg')
	else:
		config_save_dir = os.path.join(config['save'], 'model.cfg')
	save_config(config, config_save_dir)

	# contruct trainer
	t = Trainer(expt_dir=config['save'],
					load_dir=config['load'],
					load_mode=config['load_mode'],
					batch_size=config['batch_size'],
					checkpoint_every=config['checkpoint_every'],
					print_every=config['print_every'],
					eval_mode=config['eval_mode'],
					eval_metric=config['eval_metric'],
					learning_rate=config['learning_rate'],
					learning_rate_init=config['learning_rate_init'],
					lr_warmup_steps=config['lr_warmup_steps'],
					eval_with_mask=config['eval_with_mask'],
					use_gpu=config['use_gpu'],
					gpu_id=config['gpu_id'],
					max_grad_norm=config['max_grad_norm'],
					max_count_no_improve=config['max_count_no_improve'],
					max_count_num_rollback=config['max_count_num_rollback'],
					keep_num=config['keep_num'],
					normalise_loss=config['normalise_loss'],
					minibatch_split=config['minibatch_split']
					)

	# load train set
	train_path_src = config['train_path_src']
	train_path_tgt = config['train_path_tgt']
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']
	train_set = Dataset(train_path_src, train_path_tgt,
		path_vocab_src=path_vocab_src, path_vocab_tgt=path_vocab_tgt,
		seqrev=config['seqrev'],
		max_seq_len=config['max_seq_len'],
		batch_size=config['batch_size'],
		data_ratio=config['data_ratio'],
		use_gpu=config['use_gpu'],
		logger=t.logger,
		use_type=config['use_type'])

	vocab_size_enc = len(train_set.vocab_src)
	vocab_size_dec = len(train_set.vocab_tgt)

	# load dev set
	if config['dev_path_src'] and config['dev_path_tgt']:
		dev_path_src = config['dev_path_src']
		dev_path_tgt = config['dev_path_tgt']
		dev_set = Dataset(dev_path_src, dev_path_tgt,
			path_vocab_src=path_vocab_src, path_vocab_tgt=path_vocab_tgt,
			seqrev=config['seqrev'],
			max_seq_len=config['max_seq_len'],
			batch_size=config['batch_size'],
			use_gpu=config['use_gpu'],
			logger=t.logger,
			use_type=config['use_type'])
	else:
		dev_set = None

	# construct model
	seq2seq = Seq2seq(vocab_size_enc, vocab_size_dec,
					share_embedder=config['share_embedder'],
					enc_embedding_size=config['embedding_size_enc'],
					dec_embedding_size=config['embedding_size_dec'],
					load_embedding_src=config['load_embedding_src'],
					load_embedding_tgt=config['load_embedding_tgt'],
					num_heads=config['num_heads'],
					dim_model=config['dim_model'],
					dim_feedforward=config['dim_feedforward'],
					enc_layers=config['enc_layers'],
					dec_layers=config['dec_layers'],
					embedding_dropout=config['embedding_dropout'],
					dropout=config['dropout'],
					max_seq_len=config['max_seq_len'],
					act=config['act'],
					enc_word2id=train_set.src_word2id,
					dec_word2id=train_set.tgt_word2id,
					enc_id2word=train_set.src_id2word,
					dec_id2word=train_set.tgt_id2word,
					transformer_type=config['transformer_type'])

	device = check_device(config['use_gpu'])
	t.logger.info('device: {}'.format(device))
	seq2seq = seq2seq.to(device=device)

	# run training
	seq2seq = t.train(train_set, seq2seq, num_epochs=config['num_epochs'],
		dev_set=dev_set, grab_memory=config['grab_memory'])


if __name__ == '__main__':
	main()
