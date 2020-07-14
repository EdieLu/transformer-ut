import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

from utils.config import PAD, EOS, BOS, UNK


"""
	modified from -

	[1]https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
	[2]https://github.com/andreamad8/Universal-Transformer-Pytorch/
	[3]https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/
"""

# ---------------------------------------------------------------------------
# layers: transformer encoder + decoder
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):

	"""
		Args:
			dim_model: k,q,v dimension (len = k_len = q_len = v_len)
			nhead: number of attention heads
			dim_feedforward: ff layer hidden size
			d_k, d_v: internal key, value dimension
			dropout
		NOTE: seqlen x batch x d
	"""

	def __init__(self, dim_model, nhead, dim_feedforward, d_k, d_v, dropout=0.1):

		super(TransformerEncoderLayer, self).__init__()

		self.slf_attn = MultiheadAttention(nhead, dim_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(dim_model, dim_feedforward, dropout=dropout)


	def forward(self, src, slf_attn_mask=None, prior_weight=None):

		"""
			Args:
				src: input seq 	[b x len x dim_model]
				slf_attn_mask:
							[1] causal attention,
								where the mask prevents the attention
								from looking forward in time
								[q_len x k_len] (invariant across batches)
							[2] mask on certain keys to exclude <PAD>
								[b x k_len]
		"""

		# import pdb; pdb.set_trace()

		x = src
		y, att = self.slf_attn(x, x, x, mask=slf_attn_mask, prior_weight=prior_weight)
		y = self.pos_ffn(y)

		return y, att


class TransformerDecoderLayer(nn.Module):

	"""
		Args:
			dim_model: k,q,v dimension
			nhead: number of attention heads
			dim_feedforward: ff layer hidden size
			d_k, d_v: internal key, value dimension
			dropout
		NOTE: seqlen x batch x d
	"""

	def __init__(self, dim_model, nhead, dim_feedforward, d_k, d_v, dropout=0.1):

		super(TransformerDecoderLayer, self).__init__()

		self.decslf_attn = MultiheadAttention(nhead, dim_model, d_k, d_v, dropout=dropout)
		self.encdec_attn = MultiheadAttention(nhead, dim_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(dim_model, dim_feedforward, dropout=dropout)


	def forward(self, dec_input, enc_output,
		decslf_attn_mask=None, encdec_attn_mask=None,
		decode_speedup=False, cache_decslf=None, cache_encdec=None):

		"""
			Args:
				dec_input: target seq 				[b x len x dim_model]
				enc_output: encoder outputs			[b x len x dim_model]
				*attn_mask: attnetion mask - same as in encoder
		"""

		# import pdb; pdb.set_trace()

		x = dec_input
		y, att_decslf = self.decslf_attn(x, x, x, mask=decslf_attn_mask,
			decode_speedup=decode_speedup, cache=cache_decslf)
		cache_decslf = y.detach()[:,-1:,:]
		y, att_encdec = self.encdec_attn(y, enc_output, enc_output, mask=encdec_attn_mask,
			decode_speedup=decode_speedup, cache=cache_encdec)
		cache_encdec = y.detach()[:,-1:,:]
		y = self.pos_ffn(y)

		if decode_speedup:
			return y, att_decslf, att_encdec, cache_decslf, cache_encdec
		else:
			return y, att_decslf, att_encdec

# ----------------------------------------------------------------
# sub-layers:
# 	[1]replicate nn.MultiheadAttention w/ batch_first
# 	[2]PositionwiseFeedForward
# ----------------------------------------------------------------

class MultiheadAttention(nn.Module):

	''' Multi-Head Attention module '''

	def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
		super().__init__()

		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v

		self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
		self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
		self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
		self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

		self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


	def forward(self, q, k, v, mask=None, prior_weight=None,
		decode_speedup=False, cache=None):

		"""
			decode_speedup: only care about last query position in decoding
		"""
		# import pdb; pdb.set_trace()

		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
		sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

		residual = q
		q = self.layer_norm(q)

		# Pass through the pre-attention projection: b x lq x (n*dv)
		# Separate different heads: b x lq x n x dv
		q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
		k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
		v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

		# Transpose for attention dot product: b x n x lq x dv
		q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

		if mask is not None:
			# For head axis broadcasting. b x 1 x lq x lv
			mask = mask.unsqueeze(1)
		if prior_weight is not None:
			# For head axis broadcasting. b x 1 x lq x lv
			prior_weight = prior_weight.unsqueeze(1)

		if decode_speedup:
			# speedup with cache
			q = q[:, :, -1:, :] # b x n x 1 x dv
			if mask is not None: mask = mask[:, :, -1:, :] # b x 1 x 1 x lv
			if prior_weight is not None: prior_weight = prior_weight[:, :, -1:, :]

			q, attn = self.attention(q, k, v, mask=mask, prior_weight=prior_weight)
			# Transpose to move the head dimension back: b x 1 x n x dv
			q = q.transpose(1, 2).contiguous().view(sz_b, 1, -1)
			# Combine the last two dimensions to concatenate all the heads together:
			q = self.dropout(self.fc(q)) # b x 1 x (n*dv)
			q += residual[:,-1:,:]

			if len_q != 1:
				assert type(cache) != type(None)
				q = torch.cat((cache, q), dim=1) # b x lq x n*dv

		else:
			q, attn = self.attention(q, k, v, mask=mask, prior_weight=prior_weight)
			# Transpose to move the head dimension back: b x lq x n x dv
			q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
			# Combine the last two dimensions to concatenate all the heads together:
			q = self.dropout(self.fc(q)) # b x lq x (n*dv)
			q += residual

		return q, attn


class ScaledDotProductAttention(nn.Module):

	'''
		Scaled Dot-Product Attention
		mask: fill False(0) with -1e9 i.e. ignore elem == PAD
	'''

	def __init__(self, temperature, attn_dropout=0.1):

		super().__init__()
		self.temperature = temperature
		self.dropout = nn.Dropout(attn_dropout)

	def forward(self, q, k, v, mask=None, prior_weight=None):

		# import pdb; pdb.set_trace()
		attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

		if prior_weight is not None:
			# prior_weight: b x 1 x 1 x lq
			attn = attn * prior_weight

		if mask is not None:
			# mask: b x 1 x 1 x lq; attn: b x n x lq x lq
			attn = attn.masked_fill(mask == 0, -1e9)

		attn = self.dropout(F.softmax(attn, dim=-1))
		output = torch.matmul(attn, v)

		return output, attn


class PositionwiseFeedForward(nn.Module):

	''' A two-feed-forward-layer module '''

	def __init__(self, d_in, d_hid, dropout=0.1):
		super().__init__()
		self.w_1 = nn.Linear(d_in, d_hid) # position-wise
		self.w_2 = nn.Linear(d_hid, d_in) # position-wise
		self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):

		residual = x
		x = self.layer_norm(x)

		x = self.w_2(F.relu(self.w_1(x)))
		x = self.dropout(x)
		x += residual

		return x


# -------------------------------------------------------------
# internal utils
# -------------------------------------------------------------


def _get_zero_mask(seq):

	""" For masking out the zeros in the sequence. """

	padding_mask = (seq != 0).unsqueeze(-2) # b x len -> b x 1 x len

	return padding_mask


def _get_pad_mask(seq):

	""" For masking out the padding part of the sequence. """

	padding_mask = (seq != PAD).unsqueeze(-2) # b x len -> b x 1 x len

	return padding_mask


def _get_subsequent_mask(max_length):

	""" For masking out future timesteps during attention. """

	# import pdb; pdb.set_trace()

	# pt>=1.3
	# torch_mask = (1 - torch.triu(torch.ones((1, max_length, max_length)), diagonal=1)).bool()

	# pt>=1.1
	torch_mask = (1 - torch.triu(torch.ones((1, max_length, max_length)), diagonal=1)).type(torch.bool)

	return torch_mask


def _gen_position_signal(max_len, d_model):

	"""
		Generates a [1, max_len, d_model] position signal consisting of sinusoids
		Adapted from:
		https://github.com/pytorch/examples/blob/master/word_language_model/model.py
	"""

	# import pdb; pdb.set_trace()
	pe = torch.zeros(max_len, d_model)
	position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
	div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
	pe[:, 0::2] = torch.sin(position * div_term)
	pe[:, 1::2] = torch.cos(position * div_term)
	pe = pe.unsqueeze(0)

	return pe.clone().detach()
