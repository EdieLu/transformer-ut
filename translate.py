import torch
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

from utils.dataset import Dataset
from utils.misc import save_config, validate_config
from utils.misc import get_memory_alloc, log_ckpts
from utils.misc import plot_alignment, check_device, combine_weights
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor
from utils.config import PAD, EOS
from modules.checkpoint import Checkpoint
from models.Seq2seq import Seq2seq

logging.basicConfig(level=logging.INFO)


def load_arguments(parser):

	""" Seq2seq eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--path_vocab_src', type=str, default='None', help='vocab src dir, not needed')
	parser.add_argument('--path_vocab_tgt', type=str, default='None', help='vocab tgt dir, not needed')
	parser.add_argument('--load', type=str, required=True, help='model load dir')
	parser.add_argument('--combine_path', type=str, default='None', help='combine multiple ckpts if given dir')
	parser.add_argument('--test_path_out', type=str, required=True, help='test out dir')

	# others
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--beam_width', type=int, default=0, help='beam width; set to 0 to disable beam search')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_mode', type=int, default=2, help='which evaluation mode to use')
	parser.add_argument('--seqrev', type=str, default='False', help='whether or not to reverse sequence')
	parser.add_argument('--use_type', type=str, default='word', help='word | char')

	return parser


def translate_logp(test_set, model, test_path_out, use_gpu,
	max_seq_len, device, seqrev=False):

	"""
		run translation; record logp
	"""
	# import pdb; pdb.set_trace()

	# reset max_len
	model.max_seq_len = max_seq_len
	model.enc.expand_time(max_seq_len)
	model.dec.expand_time(max_seq_len)
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):

				print(idx+1, len(evaliter))
				batch_items = evaliter.next()

				# load data
				src_ids = batch_items['srcid'][0]
				src_lengths = batch_items['srclen']
				tgt_ids = batch_items['tgtid'][0]
				tgt_lengths = batch_items['tgtlen']
				src_len = max(src_lengths)
				tgt_len = max(tgt_lengths)
				src_ids = src_ids[:,:src_len].to(device=device)
				tgt_ids = tgt_ids.to(device=device)

				# import pdb; pdb.set_trace()
				# split minibatch to avoid OOM
				# if idx < 12: continue

				n_minibatch = int(tgt_len / 100 + tgt_len % 100 > 0)
				minibatch_size = int(src_ids.size(0) / n_minibatch)
				n_minibatch = int(src_ids.size(0) / minibatch_size) + \
					(src_ids.size(0) % minibatch_size > 0)

				for j in range(n_minibatch):

					st = j * minibatch_size
					ed = min((j+1) * minibatch_size, src_ids.size(0))
					src_ids_sub = src_ids[st:ed,:]

					print('minibatch: ', st, ed, src_ids.size(0))

					time1 = time.time()
					preds, logps, *_ = model.forward_eval(src=src_ids_sub, use_gpu=use_gpu)
					time2 = time.time()
					print('comp time: ', time2-time1)

					# import pdb; pdb.set_trace()

					# write to file
					seqlist = preds[:,1:]
					logps_max = torch.max(logps[:,1:,:], dim=2)[0] # b x len
					seqwords = _convert_to_words_batchfirst(seqlist, test_set.tgt_id2word)

					for i in range(len(seqwords)):
						if src_lengths[i] == 0:
							continue
						words = []
						logp_sum = 0
						for j in range(len(seqwords[i])):
							word = seqwords[i][j]
							logp = logps_max[i][j]
							if word == '<pad>':
								continue
							elif word == '<spc>':
								words.append(' ')
								logp_sum += logp
							elif word == '</s>':
								# average over sequence length
								logp_ave = 1. * logp_sum / len(words)
								break
							else:
								words.append(word)
								logp_sum += logp

						if 'logp_ave' not in locals():
								logp_ave = 1. * logp_sum / len(words)

						if len(words) == 0:
							outline = ''
						else:
							if seqrev:
								words = words[::-1]
							if test_set.use_type == 'word':
								outline = ' '.join(words)
							elif test_set.use_type == 'char':
								outline = ''.join(words)
						f.write('{:0.5f}\t{}\n'.format(logp_ave, outline))

					sys.stdout.flush()


def translate(test_set, model, test_path_out, use_gpu,
	max_seq_len, beam_width, device, seqrev=False):

	"""
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""
	# import pdb; pdb.set_trace()

	# reset max_len
	model.max_seq_len = max_seq_len
	model.enc.expand_time(max_seq_len)
	model.dec.expand_time(max_seq_len)
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):

				print(idx+1, len(evaliter))
				batch_items = evaliter.next()

				# load data
				src_ids = batch_items['srcid'][0]
				src_lengths = batch_items['srclen']
				tgt_ids = batch_items['tgtid'][0]
				tgt_lengths = batch_items['tgtlen']
				src_len = max(src_lengths)
				tgt_len = max(tgt_lengths)
				src_ids = src_ids[:,:src_len].to(device=device)
				tgt_ids = tgt_ids.to(device=device)

				# import pdb; pdb.set_trace()
				# split minibatch to avoid OOM
				# if idx < 12: continue

				n_minibatch = int(tgt_len / 100 + tgt_len % 100 > 0)
				minibatch_size = int(src_ids.size(0) / n_minibatch)
				n_minibatch = int(src_ids.size(0) / minibatch_size) + \
					(src_ids.size(0) % minibatch_size > 0)

				for j in range(n_minibatch):

					st = j * minibatch_size
					ed = min((j+1) * minibatch_size, src_ids.size(0))
					src_ids_sub = src_ids[st:ed,:]

					print('minibatch: ', st, ed, src_ids.size(0))

					time1 = time.time()
					if next(model.parameters()).is_cuda:
						preds = model.forward_translate(src=src_ids_sub,
								beam_width=beam_width, use_gpu=use_gpu)
					else:
						# preds = model.forward_translate_fast(src=src_ids_sub,
						# 			beam_width=beam_width, use_gpu=use_gpu)
						preds = model.forward_translate(src=src_ids_sub,
									beam_width=beam_width, use_gpu=use_gpu)
					time2 = time.time()
					print('comp time: ', time2-time1)

					# preds2, logps, *_ = model.forward_eval(src=src_ids_sub, use_gpu=use_gpu)

					# write to file
					seqlist = preds[:,1:]
					seqwords = _convert_to_words_batchfirst(seqlist, test_set.tgt_id2word)

					# import pdb; pdb.set_trace()

					for i in range(len(seqwords)):
						if src_lengths[i] == 0:
							continue
						words = []
						for word in seqwords[i]:
							if word == '<pad>':
								continue
							elif word == '<spc>':
								words.append(' ')
							elif word == '</s>':
								break
							else:
								words.append(word)
						if len(words) == 0:
							outline = ''
						else:
							if seqrev:
								words = words[::-1]
							if test_set.use_type == 'word':
								outline = ' '.join(words)
							elif test_set.use_type == 'char':
								outline = ''.join(words)
						f.write('{}\n'.format(outline))

					sys.stdout.flush()


def main():

	# load config
	parser = argparse.ArgumentParser(description='Seq2seq Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# load src-tgt pair
	test_path_src = config['test_path_src']
	test_path_tgt = test_path_src
	test_path_out = config['test_path_out']
	load_dir = config['load']
	max_seq_len = config['max_seq_len']
	batch_size = config['batch_size']
	beam_width = config['beam_width']
	use_gpu = config['use_gpu']
	seqrev = config['seqrev']
	use_type = config['use_type']

	# set test mode: 1 = translate; 2 = plot; 3 = save comb ckpt
	MODE = config['eval_mode']
	if MODE != 3:
		if not os.path.exists(test_path_out):
			os.makedirs(test_path_out)
		config_save_dir = os.path.join(test_path_out, 'eval.cfg')
		save_config(config, config_save_dir)

	# check device:
	device = check_device(use_gpu)
	print('device: {}'.format(device))

	# load model
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model = resume_checkpoint.model.to(device)
	vocab_src = resume_checkpoint.input_vocab
	vocab_tgt = resume_checkpoint.output_vocab
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# combine model
	if type(config['combine_path']) != type(None):
		model = combine_weights(config['combine_path'])

	# load test_set
	test_set = Dataset(test_path_src, test_path_tgt,
						vocab_src_list=vocab_src, vocab_tgt_list=vocab_tgt,
						seqrev=seqrev,
						max_seq_len=900,
						batch_size=batch_size,
						use_gpu=use_gpu,
						use_type=use_type)
	print('Test dir: {}'.format(test_path_src))
	print('Testset loaded')
	sys.stdout.flush()

	# run eval
	if MODE == 1:
		translate(test_set, model, test_path_out, use_gpu,
			max_seq_len, beam_width, device, seqrev=seqrev)

	elif MODE == 2: # output posterior
		translate_logp(test_set, model, test_path_out, use_gpu,
			max_seq_len, device, seqrev=seqrev)

	elif MODE == 3: # save combined model
		ckpt = Checkpoint(model=model,
				   optimizer=None, epoch=0, step=0,
				   input_vocab=test_set.vocab_src,
				   output_vocab=test_set.vocab_tgt)
		saved_path = ckpt.save_customise(
			os.path.join(config['combine_path'].strip('/')+'-combine','combine'))
		log_ckpts(config['combine_path'], config['combine_path'].strip('/')+'-combine')
		print('saving at {} ... '.format(saved_path))



if __name__ == '__main__':
	main()
