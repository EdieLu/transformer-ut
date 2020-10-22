import random
import numpy as np
import psutil
import os
import torch
import torch.nn as nn

# for plot
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modules.checkpoint import Checkpoint


def reserve_memory(device_id=0):

	# import pdb; pdb.set_trace()
	total, used = os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used \
		--format=csv,nounits,noheader').read().split('\n')[device_id].split(",")

	total = int(total)
	used = int(used)

	max_mem = int(total * 0.85)
	block_mem = max_mem - used

	x = torch.rand((256,1024,block_mem)).cuda()
	x = torch.rand((2,2)).cuda()


def add2corpus(hyp, ref, dict, hyp_corpus, ref_corpus, type='char'):

	""" map list of hyp/ref id to word lis; add to corpus """

	hyp_toks = _convert_to_words_batchfirst(hyp, dict)
	ref_toks = _convert_to_words_batchfirst(ref[:,1:], dict)

	for i in range(len(hyp_toks)):
		# loop over each line
		# hyp
		words = []
		for word in hyp_toks[i]:
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
			if type == 'word':
				outline = ' '.join(words)
			elif type == 'char':
				outline = ''.join(words)
		hypline = outline

		# ref
		words = []
		for word in ref_toks[i]:
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
			if type == 'word':
				outline = ' '.join(words)
			elif type == 'char':
				outline = ''.join(words)
		refline = outline

		# accumulate lines
		# import pdb; pdb.set_trace()
		hyp = hypline.split()
		ref = refline.split()
		hyp_corpus.append(hyp)
		ref_corpus.append([ref])

	return hyp_corpus, ref_corpus


def combine_weights(path):

	"""
	 	reference - qd212
		average ckpt weights under the given path
	"""

	ckpt_path_list = [os.path.join(path, ep) for ep in os.listdir(path)]
	ckpt_state_dict_list = [Checkpoint.load(ckpt_path).model.state_dict()
		for ckpt_path in ckpt_path_list]

	model = Checkpoint.load(ckpt_path_list[0]).model
	mean_state_dict = model.state_dict()
	for key in mean_state_dict.keys():
		mean_state_dict[key] = 1. * (sum(d[key] for d in ckpt_state_dict_list)
			/ len(ckpt_state_dict_list))

	model.load_state_dict(mean_state_dict)

	return model


def log_ckpts(ckpt_path, out_path):

	f = open(os.path.join(out_path,'ckpts.log'), 'w')
	for ckpt in sorted(os.listdir(ckpt_path)):
		f.write('{}\n'.format(ckpt))
	f.close()

	
def check_device(use_gpu):

	""" set device """
	# import pdb; pdb.set_trace()
	# assert torch.cuda.is_available()
	if use_gpu and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	return device


def get_memory_alloc():

	""" get memory used by current process """

	process = psutil.Process(os.getpid())
	mem_byte = process.memory_info().rss  # in bytes
	mem_kb = float(mem_byte) / (1024.0)
	mem_mb = float(mem_kb) / (1024.0)
	mem_gb = float(mem_mb) / (1024.0)

	return mem_kb, mem_mb, mem_gb


def get_device_memory():

	""" get total memory on current device """

	device = torch.cuda.current_device()
	mem_byte = torch.cuda.get_device_properties(device).total_memory
	mem_kb = float(mem_byte) / (1024.0)
	mem_mb = float(mem_kb) / (1024.0)
	mem_gb = float(mem_mb) / (1024.0)

	return mem_kb, mem_mb, mem_gb


def set_global_seeds(i):

	try:
		import torch
	except ImportError:
		pass
	else:
		torch.manual_seed(i)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(i)
	np.random.seed(i)
	random.seed(i)


def write_config(path, config):

	with open(path, 'w') as file:
		for x in config:
			file.write('{}={}\n'.format(x, config[x]))


def read_config(path):

	config = {}
	with open(path, 'r') as file:
		for line in file:
			x = line.strip().split('=')
			key = x[0]
			if x[1].isdigit():
				val = int(x[1])
			elif isfloat(x[1]):
				val = float(x[1])
			elif x[1].lower() == "true" or x[1].lower() == "false":
				if x[1].lower() == "true":
					val = True
				else:
					val = False
			else: # string
				val = x[1]

			config[key] = val

	return config


def print_config(config):

	print('\n-------- Config --------')
	for key, val in config.items():
		print("{}:{}".format(key, val))


def save_config(config, save_dir):

	f = open(save_dir, 'w')
	for key, val in config.items():
		f.write("{}:{}\n".format(key, val))
	f.close()


def validate_config(config):

	for key, val in config.items():
		if isinstance(val,str):
			if val.lower() == 'true':
				config[key] = True
			if val.lower() == 'false':
				config[key] = False
			if val.lower() == 'none':
				config[key] = None

	return config



def get_base_hidden(hidden):

	""" strip the nested tuple, get the last hidden state """

	tuple_dim = []
	while isinstance(hidden, tuple):
		tuple_dim.append(len(hidden))
		hidden = hidden[-1]
	return hidden, tuple_dim


def init_hidden():

	""" TODO """
	pass

	if hidden is None:
		num_layers = getattr(self.cell, 'num_layers', 1)
		zero = inputs.data.new(1).zero_()
		h0 = zero.view(1, 1, 1).expand(num_layers, batch_size, hidden_size)
		hidden = h0


def check_srctgt(src_ids, tgt_ids, src_id2word, tgt_id2word):

	""" check src(2dlist) tgt(2dlist) pairs """

	msgs = []
	assert len(src_ids) == len(tgt_ids), 'Mismatch src tgt length'

	for i in range(min(3,len(src_ids))):
		srcseq = []
		tgtseq = []
		# print(src_ids[i])
		# print(tgt_ids[i])
		for j in range(len(src_ids[0])):
			srcseq.append(src_id2word[src_ids[i][j]])
			tgtseq.append(src_id2word[tgt_ids[i][j]])
		msgs.append('{} - src: {}\n'.format(i, ' '.join(srcseq)).encode('utf-8'))
		msgs.append('{} - tgt: {}\n'.format(i, ' '.join(tgtseq)).encode('utf-8'))

	return msgs


def _convert_to_words(seqlist, tgt_id2word):

	"""
		convert sequences of word_ids to words
		Args:
			seqlist: ids of predicted sentences [seq_len x num_batch]
			tgt_id2word: id2word dictionary
		Returns:
			a sequence[batch] of sequence[time] of words
	"""

	seq_len = len(seqlist)
	num_batch = len(seqlist[0])
	words = []

	for i in range(num_batch):
		seqwords = []
		for j in range(seq_len):
			seqwords.append(tgt_id2word[int(seqlist[j][i].data)])
		words.append(seqwords)

	return words


def _convert_to_words_batchfirst(seqlist, tgt_id2word):

	"""
		convert sequences of word_ids to words
		Args:
			seqlist: ids of predicted sentences [num_batch x seq_len]
			tgt_id2word: id2word dictionary
		Returns:
			a sequence[batch] of sequence[time] of words
	"""

	num_batch = len(seqlist)
	seq_len = len(seqlist[0])
	words = []

	for i in range(num_batch):
		seqwords = []
		for j in range(seq_len):
			seqwords.append(tgt_id2word[int(seqlist[i][j].data)])
		words.append(seqwords)

	return words


def _convert_to_tensor(variable, use_gpu):

	""" convert variable to torch tensor """

	if type(variable) == type(None):
		return None

	variable = torch.tensor(variable)
	if use_gpu:
		variable = variable.cuda()

	return variable


def _del_var(model):

	""" delete var to free up memory """

	for name, param in model.named_parameters():
		del param
	torch.cuda.empty_cache()


def plot_alignment(alignment, path, src, hyp, ref=None):

	"""
		plot att alignment -
		adapted from: https://gitlab.com/Josh-ES/tacotron/blob/master/tacotron/utils/plot.py
	"""

	fig, ax = plt.subplots(figsize=(12, 10))
	im = ax.imshow(
		alignment,
		aspect='auto',
		cmap='hot',
		origin='lower',
		interpolation='none',
		vmin=0, vmax=1)
	fig.colorbar(im, ax=ax)

	plt.xticks(np.arange(len(src)), src, rotation=40)
	plt.yticks(np.arange(len(hyp)), hyp, rotation=20)

	xlabel = 'Src'
	if ref is not None:
		xlabel += '\n\nRef: ' + ' '.join(ref)

	plt.xlabel(xlabel)
	plt.ylabel('Hyp')
	plt.tight_layout()

	# save the alignment to disk
	plt.savefig(path, format='png')


def _inflate(tensor, times, dim):

	"""
		Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)
		Args:
			tensor: A :class:`Tensor` to inflate
			times: number of repetitions
			dim: axis for inflation (default=0)
		Returns:
			A :class:`Tensor`
		Examples::
			>> a = torch.LongTensor([[1, 2], [3, 4]])
			>> a
			1   2
			3   4
			[torch.LongTensor of size 2x2]
			>> b = ._inflate(a, 2, dim=1)
			>> b
			1   2   1   2
			3   4   3   4
			[torch.LongTensor of size 2x4]
			>> c = _inflate(a, 2, dim=0)
			>> c
			1   2
			3   4
			1   2
			3   4
			[torch.LongTensor of size 4x2]
	"""

	repeat_dims = [1] * tensor.dim()
	repeat_dims[dim] = times
	return tensor.repeat(*repeat_dims)


def inflat_hidden_state(hidden_state, k):

	if hidden_state is None:
		hidden = None
	else:
		if isinstance(hidden_state, tuple):
			hidden = tuple([_inflate(h, k, 1) for h in hidden_state])
		else:
			hidden = _inflate(hidden_state, k, 1)
	return hidden


def convert_dd_att_ref(labs):

	"""
		labs: b * l
		out: b * l
		conversion of each row:
		in	0 0 0 1 1 0 0 0 0 0 ...
		out	0 1 2 5 6 7 8 9 ...
	"""
	b = labs.size()[0]
	l = labs.size()[1]
	dummy = torch.autograd.Variable(torch.LongTensor(l).fill_(0), requires_grad=False)

	outs = [(labs[i,:] == 0).nonzero().view(-1) for i in range(b)]
	outs.append(dummy) # ensure 32 out
	outs_pad = torch.nn.utils.rnn.pad_sequence(outs, batch_first=True)
	res = outs_pad[:-1,:]

	# print(labs)
	# print(outs)
	# print(outs_pad.size())
	# print(outs_pad[:-1,:].size())

	# print(res)
	# input('...')

	return res
