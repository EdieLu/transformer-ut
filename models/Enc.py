import random
import numpy as np

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.layers import TransformerEncoderLayer
from modules.layers import _gen_position_signal
from .Act import ACT

import warnings
warnings.filterwarnings("ignore")


class Encoder(nn.Module):

	""" transformer encoder	"""

	def __init__(self,
		dim_model = 200,
		dim_feedforward=512,
		num_heads = 8,
		num_layers = 6,
		act=False,
		dropout=0.2,
		transformer_type='standard'
		):

		super(Encoder, self).__init__()

		upperbound_seq_len = 500 # upper boundary of seq_len for both train and eval

		# layer [1 x num_layers x dim_model]
		self.layer_signal = _gen_position_signal(num_layers, dim_model)
		# time [1 x max_seq_len x dim_model]
		self.time_signal = _gen_position_signal(upperbound_seq_len, dim_model)

		self.dim_model = dim_model
		self.dim_feedforward = dim_feedforward
		self.d_k = int(dim_model / num_heads)
		self.d_v = int(dim_model / num_heads)
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.act = act
		self.transformer_type = transformer_type

		self.enc = TransformerEncoderLayer(self.dim_model, self.num_heads,
			self.dim_feedforward, self.d_k, self.d_v, dropout)
		if self.transformer_type == 'universal':
			self.enc_layers = nn.ModuleList([self.enc for _ in range(num_layers)])
			if self.act:
				self.act_fn = ACT(self.dim_model)
		elif self.transformer_type == 'standard':
			self.enc_layers = _get_clones(self.enc, num_layers)
		else: assert False, 'not implemented transformer type'

		self.norm = nn.LayerNorm(self.dim_model, eps=1e-6)


	def forward(self, src, src_mask=None):

		"""
			add time/layer positional encoding; then run encoding
			Args:
				src: [b x seq_len x dim_model]
		"""

		# import pdb; pdb.set_trace()

		x = src[:]
		if not self.act:
			for layer in range(self.num_layers):
				x = x + self.time_signal[:, :src.shape[1], :].type_as(
					src.data).clone().detach()
				if self.transformer_type == 'universal':
					x = x + self.layer_signal[:, layer, :].unsqueeze(1).repeat(
						1,src.shape[1],1).type_as(src.data).clone().detach()
				x, att = self.enc_layers[layer](x, slf_attn_mask=src_mask)
			x = self.norm(x)
			return x, att
		else:
			x, layer_map = self.act_fn.forward_enc(x, src_mask, self.enc,
				self.time_signal, self.layer_signal, self.num_layers)
			x = self.norm(x)
			return x, layer_map


def _get_clones(module, N):

	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
