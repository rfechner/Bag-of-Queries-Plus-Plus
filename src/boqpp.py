# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

"""
    TODO: 
    - think about how to include exemplars.
    - can we random-project the large vectors into smaller ones?

"""
import torch

class BoQPlusPlusBlock(torch.nn.Module):
    def __init__(self, in_dim, seer_k, nheads=8, max_capacity=1000):
        super(BoQPlusPlusBlock, self).__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.max_capacity = max_capacity
        self.seer_k = seer_k
        self.similarity_t = 0.8
        self.register_buffer('M', None)  # lazy initialized in forward
        self.register_buffer('M_index', torch.tensor(0, dtype=torch.long))

    def forward(self, x, mode):
        x = self.encoder(x)
        B, S, C = x.shape
        if self.M is None:
            self.M = torch.zeros((self.max_capacity, S * C), device=x.device, dtype=x.dtype)

        xd = torch.nn.functional.normalize(x.flatten(1), p=2, dim=-1) # [B, S * C]
        if mode == 'train':
            with torch.no_grad():
                # check whether x is in M, if not include in M.
                for xx in xd.detach(): # [S * C,]
                    if self.M_index >= self.max_capacity:
                        break
                    if not torch.any((self.M @ xx) > self.similarity_t).item():
                        self.M[self.M_index] = xx
                        self.M_index += 1
            out = None
        elif mode=='test':
            # build attention scores
            out = (self.M @ xd.T).T # [B, max_capacity]

            # threshold output like SEER — keep only top-k along the memory dim
            k = min(self.seer_k, int(self.M_index))
            if k > 0:
                top_idx = torch.topk(out, k=k, dim=-1).indices
                mask = torch.zeros_like(out, dtype=torch.bool)
                mask.scatter_(-1, top_idx, True)
                out = out.masked_fill(~mask, 0)
            else:
                out = torch.zeros_like(out)
        else:
            raise ValueError(f"Illegal forward mode: {mode}")
        
        return x, out


class BoQPlusPlus(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, max_capacity=1000, num_layers=2):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.LayerNorm(proj_channels)
        
        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList([
            BoQPlusPlusBlock(in_dim, seer_k=200, nheads=in_dim//64, max_capacity=max_capacity) for _ in range(num_layers)])

    def forward(self, x, mode='test'):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        
        outs = []
        for i in range(len(self.boqs)):
            x, out = self.boqs[i](x, mode=mode)
            if out is not None:
                outs.append(out)
        if mode=='train':
            return None
        out = torch.cat(outs, dim=1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out