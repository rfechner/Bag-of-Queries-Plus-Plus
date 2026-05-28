# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import numpy as np
from typing import Literal

class BoQPlusPlusBlock(torch.nn.Module):
    def __init__(self, in_dim, seq_dim, seer_k, nheads=8, max_capacity=100):
        super(BoQPlusPlusBlock, self).__init__()
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        self.max_capacity = max_capacity
        self.M = torch.zeros((max_capacity, seq_dim * in_dim))
        self.M_index = 0
        self.seer_k = seer_k
        self.similarity_t = 0.8

    def forward(self, x, mode):
        x = self.encoder(x)
        xd = torch.nn.functional.normalize(x.flatten(1), p=2, dim=-1) # [B, S * C]
        if mode == 'train':
            # check whether x is in M, if not include in M.    
            for xx in xd: # [S * C,]
                if not np.any((self.M @ xx) > self.similarity_t):
                    self.M[self.M_index, :] = xx
            out = None
        else:
            # build attention scores
            out = (self.M @ xd.T).T # [B, M]

            # threshold output like SEER
            top_idx = torch.topk(out, k=self.seer_k, dim=1).indices
            out[~top_idx] = 0

        return x, out


class BoQPlusPlus(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_layers=2):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.LayerNorm(proj_channels)
        
        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList([
            BoQPlusPlusBlock(in_dim, nheads=in_dim//64) for _ in range(num_layers)])

    def prepare(self, xs): # gather exemplars before evaluation
        for x in xs:
            _ = self.forward(x, mode='train')

    def forward(self, x, mode='test'):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        
        outs = []
        for i in range(len(self.boqs)):
            x, out = self.boqs[i](x, mode=mode)
            outs.append(out)

        out = torch.cat(outs, dim=1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out