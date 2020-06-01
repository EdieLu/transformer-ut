import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.layers import TransformerDecoderLayer
from modules.layers import _get_pad_mask, _get_zero_mask, _get_subsequent_mask
from utils.config import PAD, EOS, BOS, UNK
from utils.dataset import load_pretrained_embedding
from utils.misc import check_device
from .Enc import Encoder
from .Dec import Decoder

import warnings
warnings.filterwarnings("ignore")


class Seq2seq(nn.Module):

	""" transformer enc-dec model """

	def __init__(self,
		enc_vocab_size,
		dec_vocab_size,
		enc_embedding_size = 200,
		dec_embedding_size = 200,
		load_embedding_src = None,
		load_embedding_tgt = None,
		max_seq_len = 32,
		num_heads = 8,
		dim_model = 512,
		dim_feedforward = 1024,
		enc_layers = 6,
		dec_layers = 6,
		embedding_dropout=0.0,
		dropout=0.2,
		act=False,
		enc_word2id=None,
		enc_id2word=None,
		dec_word2id=None,
		dec_id2word=None,
		transformer_type='standard',
		use_gpu=False
		):

		super(Seq2seq, self).__init__()

		# define var
		self.enc_vocab_size = enc_vocab_size
		self.dec_vocab_size = dec_vocab_size
		self.enc_embedding_size = enc_embedding_size
		self.dec_embedding_size = dec_embedding_size
		self.load_embedding_src = load_embedding_src
		self.load_embedding_tgt = load_embedding_tgt
		self.max_seq_len = max_seq_len
		self.num_heads = num_heads
		self.dim_model = dim_model
		self.dim_feedforward = dim_feedforward

		self.enc_layers = enc_layers
		self.dec_layers = dec_layers

		self.embedding_dropout = nn.Dropout(embedding_dropout)
		self.dropout = nn.Dropout(dropout)
		self.act = act

		self.enc_word2id = enc_word2id
		self.enc_id2word = enc_id2word
		self.dec_word2id = dec_word2id
		self.dec_id2word = dec_id2word
		self.transformer_type = transformer_type

		# ------------- define embedder -------------
		if self.load_embedding_src:
			embedding_matrix = np.random.rand(self.enc_vocab_size, self.enc_embedding_size)
			embedding_matrix = torch.FloatTensor(load_pretrained_embedding(
				self.enc_word2id, embedding_matrix, self.load_embedding_src))
			self.enc_embedder = nn.Embedding.from_pretrained(embedding_matrix,
				freeze=False, sparse=False, padding_idx=PAD)
		else:
			self.enc_embedder = nn.Embedding(self.enc_vocab_size,
				self.enc_embedding_size, sparse=False, padding_idx=PAD)

		if self.load_embedding_tgt:
			embedding_matrix = np.random.rand(self.dec_vocab_size, self.dec_embedding_size)
			embedding_matrix = torch.FloatTensor(load_pretrained_embedding(
				self.dec_word2id, embedding_matrix, self.load_embedding_tgt))
			self.dec_embedder = nn.Embedding.from_pretrained(embedding_matrix,
				freeze=False, sparse=False, padding_idx=PAD)
		else:
			self.dec_embedder = nn.Embedding(self.dec_vocab_size,
				self.dec_embedding_size, sparse=False, padding_idx=PAD)

		self.enc_emb_proj_flag = False
		if self.enc_embedding_size != self.dim_model:
			self.enc_emb_proj = nn.Linear(self.enc_embedding_size,
				self.dim_model, bias=False) # embedding -> hidden
			self.enc_emb_proj_flag = True
		self.dec_emb_proj_flag = False
		if self.dec_embedding_size != self.dim_model:
			self.dec_emb_proj = nn.Linear(self.dec_embedding_size,
				self.dim_model, bias=False) # embedding -> hidden
			self.dec_emb_proj_flag = True

		# ------------- construct enc, dec  -------------------
		if self.enc_layers > 0:
			enc_params = (self.dim_model, self.dim_feedforward, self.num_heads,
				self.enc_layers, self.act, dropout, self.transformer_type)
			self.enc = Encoder(*enc_params)
		if self.dec_layers > 0:
			dec_params = (self.dim_model, self.dim_feedforward, self.num_heads,
				self.dec_layers, self.act, dropout, self.transformer_type)
			self.dec = Decoder(*dec_params)

		# ------------- define out ffn -------------------
		self.out = nn.Linear(self.dim_model, self.dec_vocab_size, bias=False)


	def forward_train(self, src, tgt, debug_flag=False, use_gpu=True):

		"""
			train disfluency detection enc + dec
			note: all output useful up to the second last element i.e. b x (len-1)
					e.g. [b,:-1] for preds -
						src: 		w1 w2 w3 <EOS> <PAD> <PAD> <PAD>
						ref: 	BOS	w1 w2 w3 <EOS> <PAD> <PAD>
						tgt:		w1 w2 w3 <EOS> <PAD> <PAD> dummy
						ref start with BOS, the last elem does not have ref!
		"""

		# import pdb; pdb.set_trace()

		# check gpu
		global device
		device = check_device(use_gpu)

		# run transformer
		src_mask = _get_pad_mask(src).to(device=device) 	# b x len
		tgt_mask = ((_get_pad_mask(tgt).to(device=device)
			& _get_subsequent_mask(self.max_seq_len).to(device=device)))

		# b x len x dim_model
		if self.enc_emb_proj_flag:
			emb_src = self.enc_emb_proj(self.embedding_dropout(self.enc_embedder(src)))
		else:
			emb_src = self.embedding_dropout(self.enc_embedder(src))
		if self.dec_emb_proj_flag:
			emb_tgt = self.dec_emb_proj(self.embedding_dropout(self.dec_embedder(tgt)))
		else:
			emb_tgt = self.embedding_dropout(self.dec_embedder(tgt))

		enc_outputs, *_ = self.enc(emb_src, src_mask=src_mask)	# b x len x dim_model
		dec_outputs, *_ = self.dec(emb_tgt, enc_outputs, tgt_mask=tgt_mask, src_mask=src_mask)

		logits = self.out(dec_outputs)	# b x len x vocab_size
		logps = torch.log_softmax(logits, dim=2)
		preds = logps.data.topk(1)[1]

		return preds, logps, dec_outputs


	def forward_eval(self, src, debug_flag=False, use_gpu=True):

		"""
			eval disfluency detection enc + dec (beam_width = 1)
			all outputs following:
				tgt:	<BOS> w1 w2 w3 <EOS> <PAD>
				gen:		  w1 w2 w3 <EOS> <PAD> <PAD>
				shift by 1, i.e.
					used input = <BOS>	w1 		<PAD> 	<PAD>
					gen output = dummy	w2 		dummy
					update prediction: assign w2(output[1]) to be input[2]
		"""

		# import pdb; pdb.set_trace()

		# check gpu
		global device
		device = check_device(use_gpu)

		batch = src.size(0)
		length_out = self.max_seq_len

		# run enc dec
		src_mask = _get_pad_mask(src).to(device=device)
		if self.enc_emb_proj_flag:
			emb_src = self.enc_emb_proj(self.embedding_dropout(self.enc_embedder(src)))
		else:
			emb_src = self.embedding_dropout(self.enc_embedder(src))
		enc_outputs, enc_var = self.enc(emb_src, src_mask=src_mask)

		logps = torch.Tensor([-1e-4]).repeat(batch,length_out,self.dec_vocab_size).type(
			torch.FloatTensor).to(device=device)
		dec_outputs = torch.Tensor([0]).repeat(batch,length_out,self.dim_model).type(
			torch.FloatTensor).to(device=device)

		preds = torch.Tensor([PAD]).repeat(batch,length_out).type(
			torch.LongTensor).to(device=device)
		preds[:, 0] =  torch.Tensor([BOS]).type(
			torch.LongTensor).to(device=device)	# b x len - [<BOS> <PAD> <PAD> ...]
		preds_save = torch.Tensor([PAD]).repeat(batch,length_out).type(
			torch.LongTensor).to(device=device) 	# used to update pred history
		preds_save[:, 0] =  torch.Tensor([BOS]).type(
			torch.LongTensor).to(device=device)


		for i in range(1, self.max_seq_len):

			# gen: 0-30; ref: 1-31
			# import pdb; pdb.set_trace()

			tgt_mask = ((_get_pad_mask(preds).to(device=device)
				& _get_subsequent_mask(self.max_seq_len).to(device=device)))
			if self.dec_emb_proj_flag:
				emb_tgt = self.dec_emb_proj(self.dec_embedder(preds))
			else:
				emb_tgt = self.dec_embedder(preds)
			dec_output, dec_var, *_ = self.dec(emb_tgt, enc_outputs,
				tgt_mask=tgt_mask, src_mask=src_mask)
			logit = self.out(dec_output)
			logp = torch.log_softmax(logit, dim=2)
			pred = logp.data.topk(1)[1]

			# b x len x dim_model - [:,0,:] is dummy 0's
			dec_outputs[:, i, :] = dec_output[:, i-1]
			# b x len x vocab_size - [:,0,:] is dummy -1e-4's # individual logps
			logps[:, i, :] = logp[:, i-1, :]
			preds_save[:, i] = pred[:, i-1].view(-1)
			preds = preds_save.clone()


		if not debug_flag:
			return preds, logps, dec_outputs
		else:
			return preds, logps, dec_outputs, enc_var, dec_var


	def forward_translate(self, src, beam_width=1, penalty_factor=1, use_gpu=True):

		"""
			run disfluency detection enc + dec inference - with beam search
		"""

		# import pdb; pdb.set_trace()

		# check gpu
		global device
		device = check_device(use_gpu)

		# run dd
		src_mask = _get_pad_mask(src).to(device=device) # b x 1 x len
		if self.enc_emb_proj_flag:
			emb_src = self.enc_emb_proj(self.enc_embedder(src))
		else:
			emb_src = self.enc_embedder(src)
		enc_outputs, *_ = self.enc(emb_src, src_mask=src_mask) # b x len x dim_model

		batch = src.size(0)
		length_in = src.size(1)
		length_out = self.max_seq_len

		eos_mask = torch.BoolTensor([False]).repeat(batch * beam_width).to(device=device)
		len_map = torch.Tensor([1]).repeat(batch * beam_width).to(device=device)

		preds = torch.Tensor([PAD]).repeat(batch, self.max_seq_len).type(
			torch.LongTensor).to(device=device)
		preds[:, 0] =  torch.Tensor([BOS]).type(
			torch.LongTensor).to(device=device)	# b x len - [<BOS> <PAD> <PAD> ...]

		# repeat for beam_width times
		# a b c d -> aaa bbb ccc ddd

		# b x 1 x len -> (b x beam_width) x 1 x len
		src_mask_expand = src_mask.repeat(1, beam_width, 1).view(-1, 1, length_in)
		# b x len x dim_model -> (b x beam_width) x len x dim_model
		enc_outputs_expand = enc_outputs.repeat(1, beam_width, 1).view(-1, length_in, self.dim_model)
		# (b x beam_width) x len
		preds_expand = preds.repeat(1, beam_width).view(-1, self.max_seq_len)
		# (b x beam_width)
		scores_expand = torch.Tensor([0]).repeat(batch * beam_width).type(
			torch.FloatTensor).to(device=device)

		# loop over sequence length
		for i in range(1, self.max_seq_len):

			# gen: 0-30; ref: 1-31
			# import pdb; pdb.set_trace()

			# Get k candidates for each beam, k^2 candidates in total (k=beam_width)
			tgt_mask_expand = ((_get_pad_mask(preds_expand).to(device=device)
				& _get_subsequent_mask(self.max_seq_len).to(device=device)))
			if self.dec_emb_proj_flag:
				emb_tgt_expand = self.dec_emb_proj(self.dec_embedder(preds_expand))
			else:
				emb_tgt_expand = self.dec_embedder(preds_expand)
			dec_output_expand, *_ = self.dec(emb_tgt_expand, enc_outputs_expand,
				tgt_mask=tgt_mask_expand, src_mask=src_mask_expand)

			logit_expand = self.out(dec_output_expand)
			# (b x beam_width) x len x vocab_size
			logp_expand = torch.log_softmax(logit_expand, dim=2)
			# (b x beam_width) x len x beam_width
			score_expand, pred_expand = logp_expand.data.topk(beam_width)

			# select current slice
			dec_output = dec_output_expand[:, i-1]	# (b x beam_width) x dim_model - nouse
			logp = logp_expand[:, i-1, :] 	# (b x beam_width) x vocab_size - nouse
			pred = pred_expand[:, i-1] 		# (b x beam_width) x beam_width
			score = score_expand[:, i-1]		# (b x beam_width) x beam_width

			# select k candidates from k^2 candidates
			if i == 1:
				# inital state, keep first k candidates
				# b x (beam_width x beam_width) -> b x (beam_width) -> (b x beam_width) x 1
				score_select = scores_expand + score.reshape(batch, -1)[:,:beam_width]\
					.contiguous().view(-1)
				scores_expand = score_select
				pred_select = pred.reshape(batch, -1)[:, :beam_width].contiguous().view(-1)
				preds_expand[:, i] = pred_select

			else:
				# keep only 1 candidate when hitting eos
				# (b x beam_width) x beam_width
				eos_mask_expand = eos_mask.reshape(-1,1).repeat(1, beam_width)
				eos_mask_expand[:,0] = False
				# (b x beam_width) x beam_width
				score_temp = scores_expand.reshape(-1,1) + score.masked_fill(
					eos_mask.reshape(-1,1), 0).masked_fill(eos_mask_expand, -1e9)
				# length penalty
				score_temp = score_temp / (len_map.reshape(-1,1) ** penalty_factor)
				# select top k from k^2
				# (b x beam_width^2 -> b x beam_width)
				score_select, pos = score_temp.reshape(batch, -1).topk(beam_width)
				scores_expand = score_select.view(-1)
				# select correct elements according to pos
				pos = (pos + torch.range(0, (batch - 1) * (beam_width**2), (beam_width**2)).to(
					device=device).reshape(batch, 1)).long()
				r_idxs, c_idxs = pos // beam_width, pos % beam_width # b x beam_width
				pred_select = pred[r_idxs, c_idxs].view(-1) # b x beam_width -> (b x beam_width)
				# Copy the corresponding previous tokens.
				preds_expand[:, :i] = preds_expand[r_idxs.view(-1), :i] # (b x beam_width) x i
				# Set the best tokens in this beam search step
				preds_expand[:, i] = pred_select # (b x beam_width)

			# locate the eos in the generated sequences
			eos_mask = (pred_select == EOS) + eos_mask
			len_map = len_map + torch.Tensor([1]).repeat(batch * beam_width).to(
				device=device).masked_fill(eos_mask, 0)

		# select the best candidate
		preds = preds_expand.reshape(batch, -1)[:, :self.max_seq_len].contiguous() # b x len
		scores = scores_expand.reshape(batch, -1)[:, 0].contiguous() # b

		# select the worst candidate
		# preds = preds_expand.reshape(batch, -1)
		# [:, (beam_width - 1)*length : (beam_width)*length].contiguous() # b x len
		# scores = scores_expand.reshape(batch, -1)[:, -1].contiguous() # b

		return preds


	def check_var(self, var_name, var_val_set=None):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			var_val = var_val_set if type(var_val_set) != type(None) else None

			# set class attribute to default value
			setattr(self, var_name, var_val)
