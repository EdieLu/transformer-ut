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
from utils.misc import get_memory_alloc, plot_alignment, check_device
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor
from utils.config import PAD, EOS
from modules.loss import NLLLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.Seq2seq import Seq2seq

import logging
logging.basicConfig(level=logging.INFO)


def load_arguments(parser):

	""" Seq2seq eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--test_path_tgt', type=str, required=True, help='test tgt dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, required=True, help='vocab tgt dir')
	parser.add_argument('--load', type=str, required=True, help='model load dir')
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


def translate(test_set, load_dir, test_path_out, use_gpu,
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

	# load model
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model.to(device)
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.max_seq_len = max_seq_len
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
				time1 = time.time()
				if next(model.parameters()).is_cuda:
					preds = model.forward_translate(src=src_ids,
							beam_width=beam_width, use_gpu=use_gpu)
				else:
					preds = model.forward_translate_fast(src=src_ids,
							beam_width=beam_width, use_gpu=use_gpu)
				time2 = time.time()
				print(time2-time1)

				# memory usage
				mem_kb, mem_mb, mem_gb = get_memory_alloc()
				mem_mb = round(mem_mb, 2)
				print('Memory used: {0:.2f} MB'.format(mem_mb))
				print(idx, len(evaliter))
				torch.cuda.empty_cache()

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
						outline = ' '.join(words)
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
	test_path_tgt = config['test_path_tgt']
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']
	test_path_out = config['test_path_out']
	load_dir = config['load']
	max_seq_len = config['max_seq_len']
	batch_size = config['batch_size']
	beam_width = config['beam_width']
	use_gpu = config['use_gpu']
	seqrev = config['seqrev']
	use_type = config['use_type']

	if not os.path.exists(test_path_out):
		os.makedirs(test_path_out)
	config_save_dir = os.path.join(test_path_out, 'eval.cfg')
	save_config(config, config_save_dir)

	# set test mode: 1 = translate; 2 = plot
	MODE = config['eval_mode']

	# check device:
	device = check_device(use_gpu)
	print('device: {}'.format(device))

	# load test_set
	test_set = Dataset(test_path_src, test_path_tgt,
						path_vocab_src, path_vocab_tgt,
						seqrev=seqrev,
						max_seq_len=max_seq_len,
						batch_size=batch_size,
						use_gpu=use_gpu,
						use_type=use_type)
	print('Testset loaded')
	sys.stdout.flush()

	# run eval
	if MODE == 1:
		translate(test_set, load_dir, test_path_out, use_gpu,
			max_seq_len, beam_width, device, seqrev=seqrev)


if __name__ == '__main__':
	main()
