import random
import numpy as np

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F


class ACT(nn.Module):

	def __init__(self, hidden_size):

		super(ACT, self).__init__()
		self.sigma = nn.Sigmoid()
		self.p = nn.Linear(hidden_size,1)
		self.p.bias.data.fill_(1)
		self.threshold = 1 - 0.1
		self.hidden_size = hidden_size

	def forward_enc(self, state, src_mask, fn, time_signal, layer_signal, max_hop):

		# import pdb; pdb.set_trace()

		# state - [B, S, self.hidden_size]
		batch = state.size(0)
		length = state.size(1)
		layer_map = torch.zeros(batch, length).to(device=device)

		# initial vars
		halting_probability = torch.zeros(batch, length).to(device=device)
		remainders = torch.zeros(batch, length).to(device=device)
		n_updates = torch.zeros(batch, length).to(device=device)
		previous_state = torch.zeros(batch, length, self.hidden_size).to(device=device)
		step = 0

		# for l in range(self.num_layers):
		while ( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any() ):

			# Add timing signal
			state = state + time_signal[:, :length, :].type_as(state.data).clone().detach()
			state = state + layer_signal[:, step, :].unsqueeze(1).repeat(1,length,1).type_as(state.data).clone().detach()

			# ----------------- ACT ------------------
			# FF layer for halting probablity
			p = self.sigma(self.p(state)).squeeze(-1)

			# Mask for state which have not halted yet
			still_running = (halting_probability < 1.0).float()

			# ------------
			# Mask of state which halted at this step
			new_halted = ((halting_probability + p * still_running) > self.threshold).float() * still_running

			# Mask of state which haven't halted, and didn't halt this step
			still_running = ((halting_probability + p * still_running) <= self.threshold).float() * still_running
			# ------------

			# Add the halting probability for this step to the halting
			# probabilities for those input which haven't halted yet
			# note: only update still_running elements
			halting_probability = halting_probability + p * still_running

			# Compute probability remainders for the state which halted at this step
			# note: remainders stay 0 for still_running elements
			remainders = remainders + new_halted * (1 - halting_probability)

			# Add the remainders to those state which halted at this step
			# note: make sure prob cap at 1 for halting elements -> new_halted * (1 - halting_probability + halting_probability)
			halting_probability = halting_probability + new_halted * remainders

			# Increment n_updates for all state which are still running (track #layers)
			n_updates = n_updates + still_running + new_halted

			# Compute the weight to be applied to the new state and output
			# 0 when the input has already halted
			# p when the input hasn't halted yet
			# the remainders when it halted this step
			update_weights = p * still_running + remainders * new_halted
			# ------------------------------------------------

			# apply transformation on the state - enc/dec
			state, att = fn(state, slf_attn_mask=src_mask)

			# update running part in the weighted state and keep the rest
			previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
			## previous_state is actually the new_state at end of hte loop
			## to save a line I assigned to previous_state so in the next
			## iteration is correct. Notice that indeed we return previous_state
			step = step + 1

			# update layer map to record effective #layer
			layer_map = layer_map + new_halted * step

		layer_map = layer_map + still_running * max_hop

		return previous_state, layer_map


	def forward_dec(self, state, memory, tgt_mask, src_mask, fn, time_signal, layer_signal, max_hop):

		# import pdb; pdb.set_trace()

		# state - [B, S, self.hidden_size]
		batch = state.size(0)
		length = state.size(1)
		layer_map = torch.zeros(batch, length).to(device=device)

		# initial vars
		halting_probability = torch.zeros(batch, length).to(device=device)
		remainders = torch.zeros(batch, length).to(device=device)
		n_updates = torch.zeros(batch, length).to(device=device)
		previous_state = torch.zeros(batch, length, self.hidden_size).to(device=device)
		step = 0

		# for l in range(self.num_layers):
		while ( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any() ):

			# Add timing signal
			state = state + time_signal[:, :length, :].type_as(state.data).clone().detach()
			state = state + layer_signal[:, step, :].unsqueeze(1).repeat(1,length,1).type_as(state.data).clone().detach()

			# ----------------- ACT ------------------
			# FF layer for halting probablity
			p = self.sigma(self.p(state)).squeeze(-1)

			# Mask for state which have not halted yet
			still_running = (halting_probability < 1.0).float()

			# Mask of state which halted at this step
			new_halted = ((halting_probability + p * still_running) > self.threshold).float() * still_running

			# Mask of state which haven't halted, and didn't halt this step
			still_running = ((halting_probability + p * still_running) <= self.threshold).float() * still_running

			# Add the halting probability for this step to the halting
			# probabilities for those input which haven't halted yet
			# note: prob must be <= self.threshold
			halting_probability = halting_probability + p * still_running

			# Compute remainders for the state which halted at this step
			remainders = remainders + new_halted * (1 - halting_probability)

			# Add the remainders to those state which halted at this step
			halting_probability = halting_probability + new_halted * remainders

			# Increment n_updates for all state which are still running
			n_updates = n_updates + still_running + new_halted

			# Compute the weight to be applied to the new state and output
			# 0 when the input has already halted
			# p when the input hasn't halted yet
			# the remainders when it halted this step
			update_weights = p * still_running + new_halted * remainders
			# ------------------------------------------------

			# apply transformation on the state - enc/dec
			state, att_decslf, att_encdec = fn(state, memory, decslf_attn_mask=tgt_mask, encdec_attn_mask=src_mask)

			# update running part in the weighted state and keep the rest
			previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
			## previous_state is actually the new_state at end of hte loop
			## to save a line I assigned to previous_state so in the next
			## iteration is correct. Notice that indeed we return previous_state
			step = step + 1

			# update layer map to record effective #layer
			layer_map = layer_map + new_halted * step

		layer_map = layer_map + still_running * max_hop

		return previous_state, layer_map
