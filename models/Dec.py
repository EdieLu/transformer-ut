import random
import numpy as np

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.layers import TransformerDecoderLayer
from modules.layers import _gen_position_signal
from .Act import ACT

import warnings
warnings.filterwarnings("ignore")


class Decoder(nn.Module):

	""" transformer decoder """

	def __init__(self,
		dim_model = 200,
		dim_feedforward=512,
		num_heads = 8,
		num_layers = 6,
		act=False,
		dropout=0.2,
		transformer_type='standard'
		):

		super(Decoder, self).__init__()

		upperbound_seq_len = 500
		self.layer_signal = _gen_position_signal(num_layers, dim_model) # layer
		self.time_signal = _gen_position_signal(upperbound_seq_len, dim_model) # time

		self.dim_model = dim_model
		self.dim_feedforward = dim_feedforward
		self.d_k = int(dim_model / num_heads)
		self.d_v = int(dim_model / num_heads)
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.act = act
		self.transformer_type = transformer_type

		self.dec = TransformerDecoderLayer(self.dim_model, self.num_heads,
			self.dim_feedforward, self.d_k, self.d_v, dropout)
		if self.transformer_type == 'universal':
			self.dec_layers = nn.ModuleList([self.dec for _ in range(num_layers)])
			if self.act:
				self.act_fn = ACT(self.dim_model)
		elif self.transformer_type == 'standard':
			self.dec_layers = _get_clones(self.dec, num_layers) # deep copy
		else: assert False, 'not implemented transformer type'

		self.norm = nn.LayerNorm(self.dim_model)


	def expand_time(self, max_seq_len):

		self.time_signal = _gen_position_signal(max_seq_len, self.dim_model)


	def forward(self, tgt, memory,
		tgt_mask=None, src_mask=None,
		decode_speedup=False, cache_decslf=None, cache_encdec=None):

		"""
			add time/layer positional encoding; then run decoding
			Args:
				tgt: [b x seq_len x dim_model]
				memory: encoder outputs
		"""

		# import pdb; pdb.set_trace()

		x = tgt
		b = x.size(0)
		q_len = x.size(1)

		if not self.act:
			for layer in range(self.num_layers):
				x = x + self.time_signal[:, :tgt.shape[1], :].type_as(
					tgt.data).clone().detach()
				if self.transformer_type == 'universal':
					x = x + self.layer_signal[:, layer, :].unsqueeze(1).repeat(
						1,tgt.shape[1],1).type_as(tgt.data).clone().detach()
				if decode_speedup:
					# for speedy decoding
					if q_len == 1:
						cache_decslf_in = None
						cache_encdec_in = None
					else:
						# 1 x b x q_len-1 x dim
						cache_decslf_in = cache_decslf[layer, :, :q_len-1, :]
						cache_encdec_in = cache_encdec[layer, :, :q_len-1, :]
					x, att_decslf, att_encdec, cache_decslf_out, cache_encdec_out = \
						self.dec_layers[layer](x, memory,
							decslf_attn_mask=tgt_mask, encdec_attn_mask=src_mask,
							decode_speedup=decode_speedup,
							cache_decslf=cache_decslf_in, cache_encdec=cache_encdec_in
						)
					if layer == 0:
						# init cache new
						if q_len == 1:
							# layers x b x 1 x dim:512
							cache_decslf = cache_decslf_out.unsqueeze(0).repeat(
								self.num_layers,1,1,1)
							cache_encdec = cache_encdec_out.unsqueeze(0).repeat(
								self.num_layers,1,1,1)
						else:
							# layers x b x q_len x dim
							cache_decslf = torch.cat((cache_decslf, cache_decslf_out\
								.unsqueeze(0).repeat(self.num_layers,1,1,1)), dim=2)
							cache_encdec = torch.cat((cache_encdec, cache_encdec_out\
								.unsqueeze(0).repeat(self.num_layers,1,1,1)), dim=2)
					else:
						# update cache
						cache_decslf[layer,:,-1,:] = cache_decslf_out.squeeze()
						cache_encdec[layer,:,-1,:] = cache_encdec_out.squeeze()
				else:
					x, att_decslf, att_encdec = self.dec_layers[layer](x, memory,
						decslf_attn_mask=tgt_mask, encdec_attn_mask=src_mask)
			x = self.norm(x)
			if decode_speedup:
				return x, att_decslf, att_encdec, cache_decslf, cache_encdec
			else:
				return x, att_decslf, att_encdec
		else:
			x, layer_map = self.act_fn.forward_dec(x, memory, tgt_mask, src_mask,
				self.dec, self.time_signal, self.layer_signal, self.num_layers)
			x = self.norm(x)
			return x, layer_map


def _get_clones(module, N):

	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
