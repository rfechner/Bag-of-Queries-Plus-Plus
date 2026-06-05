# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

"""
Bag-of-Queries++ aggregator.

A merger of BoQ's transformer-encoder front-end with SEER's greedy
*sparse exemplar* bank (Neubert & Schubert, RSS 2022). Each block:

    tokens -> TransformerEncoderLayer
           -> flatten + L2-normalize    (descriptor x in R^{d_X})
           -> SEER bank                 (greedy build / top-(lambda*k) query)

All SEER hyperparameters are exposed at construction time so you can sweep
(d_M, k, lambda, sampling, ...) from a config or training script. Passing
``d_M=None`` recovers the original dense behaviour.
"""

from typing import Optional, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Sparse exemplar sampling
# ----------------------------------------------------------------------------

def sample_sparse_exemplar(
    x: torch.Tensor,
    d_M: int,
    strategy: str = 'proportional',
) -> torch.Tensor:
    """Return a sparse vector with x's values at d_M chosen indices, 0 elsewhere.

    strategy:
        - 'proportional': sample without replacement, P[i] proportional to |x_i|
          (SEER default).
        - 'topk': deterministically take the d_M largest-magnitude indices.
        - 'uniform': sample d_M indices uniformly without replacement.
    """
    d_X = x.numel()
    if d_M >= d_X:
        return x.clone()

    if strategy == 'proportional':
        # multinomial wants float32 with a strictly positive sum
        weights = x.abs().to(torch.float32) + 1e-12
        idx = torch.multinomial(weights, num_samples=d_M, replacement=False)
    elif strategy == 'topk':
        idx = torch.topk(x.abs(), k=d_M).indices
    elif strategy == 'uniform':
        idx = torch.randperm(d_X, device=x.device)[:d_M]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy!r}")

    m = torch.zeros_like(x)
    m[idx] = x[idx]
    return m


# ----------------------------------------------------------------------------
# SEER bank
# ----------------------------------------------------------------------------

class SEERBank(nn.Module):
    """Greedy bank of (optionally sparse) exemplars — SEER's createSEER as a module.

    Parameters
    ----------
    d_M : int | None
        Non-zero entries per exemplar. ``None`` => dense (d_M = d_X), matching
        the original boqpp.py behaviour.
    k : int
        Minimum number of similar exemplars per input descriptor. Controls how
        aggressively the bank grows and the output's effective density via
        lambda * k.
    lambda_ : float
        Output sparsity factor. The forward pass keeps the top ``lambda_ * k``
        similarities and zeros out the rest.
    max_capacity : int
        Hard cap on the bank size.
    similarity_threshold : float | str
        Cutoff for deciding when an input is "already represented" by the bank.
        ``'auto'`` uses d_M/d_X in sparse mode (SEER default) and 0.8 in dense
        mode.
    sampling : str
        Index-selection strategy for new exemplars. See ``sample_sparse_exemplar``.
    """

    def __init__(
        self,
        d_M: Optional[int] = None,
        k: int = 50,
        lambda_: float = 2.0,
        max_capacity: int = 1000,
        similarity_threshold: Union[float, str] = 'auto',
        sampling: str = 'proportional', **kwargs
    ):
        super().__init__(**kwargs)
        self.d_M = d_M
        self.k = k
        self.lambda_ = lambda_
        self.max_capacity = max_capacity
        self.similarity_threshold = similarity_threshold
        self.sampling = sampling

        # Bank buffer is lazily allocated once we know d_X.
        self.register_buffer('M', None)
        self.register_buffer('M_index', torch.tensor(0, dtype=torch.long))

    def _resolved_threshold(self, d_X: int) -> float:
        if self.similarity_threshold == 'auto':
            d_M = self.d_M if self.d_M is not None else d_X
            return d_M / d_X if d_M < d_X else 0.8
        return float(self.similarity_threshold)

    def _ensure_bank(self, d_X: int, device, dtype):
        # Start empty; the bank grows on demand (see ``_grow``) so we never
        # preallocate the full ``max_capacity`` rows, which would waste memory
        # whenever the bank ends up much smaller than its hard cap.
        if self.M is None or self.M.shape[1] != d_X:
            self.M = torch.zeros((0, d_X), device=device, dtype=dtype)
            self.M_index.zero_()

    def _grow(self, d_X: int, device, dtype):
        """Ensure M has room for one more exemplar, doubling capacity as needed."""
        capacity = self.M.shape[0]
        if int(self.M_index) < capacity:
            return
        new_capacity = min(max(1, 2 * capacity), self.max_capacity)
        grown = torch.zeros((new_capacity, d_X), device=device, dtype=dtype)
        grown[:capacity] = self.M
        self.M = grown

    @torch.no_grad()
    def _greedy_fill(self, x: torch.Tensor, threshold: float):
        """SEER createSEER inner loop for a single input x in R^{d_X}."""
        d_X = x.shape[0]
        d_M = self.d_M if self.d_M is not None else d_X

        cur = int(self.M_index)
        if cur > 0:
            sims = self.M[:cur] @ x
            c = int((sims >= threshold).sum())
        else:
            c = 0

        for _ in range(max(0, self.k - c)):
            if int(self.M_index) >= self.max_capacity:
                return
            self._grow(d_X, x.device, x.dtype)
            self.M[int(self.M_index)] = sample_sparse_exemplar(x, d_M, self.sampling)
            self.M_index += 1

    @torch.no_grad()
    def forward(
        self,
        xd: torch.Tensor,
        update: bool,
        produce_output: bool,
    ) -> Optional[torch.Tensor]:
        """xd: [B, d_X] L2-normalized descriptors."""
        _, d_X = xd.shape
        self._ensure_bank(d_X, xd.device, xd.dtype)
        threshold = self._resolved_threshold(d_X)

        if update:
            for x_i in xd.detach():
                if int(self.M_index) >= self.max_capacity:
                    break
                self._greedy_fill(x_i, threshold)

        if not produce_output:
            return None

        # Similarity of every input to every (valid) exemplar in the bank.
        # M may hold trailing zero-padding rows from geometric growth, so we
        # slice to M_index; the dropped columns would be zero for every input
        # anyway and so do not affect the L2-normalized descriptor.
        cur = int(self.M_index)
        out = xd @ self.M[:cur].T  # [B, cur]

        # SEER's top-(lambda * k) nonlinearity.
        n_keep = min(int(self.lambda_ * self.k), cur)
        if n_keep > 0:
            top_idx = torch.topk(out, k=n_keep, dim=-1).indices
            mask = torch.zeros_like(out, dtype=torch.bool)
            mask.scatter_(-1, top_idx, True)
            out = out.masked_fill(~mask, 0)
        else:
            out = torch.zeros_like(out)

        return out


# ----------------------------------------------------------------------------
# BoQ++ block / aggregator
# ----------------------------------------------------------------------------

# Maps the original 'train' / 'test' mode strings to (update, produce_output)
# flags. 'simultaneous' is SEER's online mode: build and query in one pass.
_MODE_FLAGS = {
    'train':        (True,  False),
    'test':         (False, True),
    'simultaneous': (True,  True),
}


class BoQPlusPlusBlock(nn.Module):
    """TransformerEncoderLayer followed by a SEER exemplar bank."""

    def __init__(
        self,
        in_dim: int,
        nheads: int = 8,
        d_M: Optional[int] = None,
        k: int = 50,
        lambda_: float = 2.0,
        max_capacity: int = 1000,
        similarity_threshold: Union[float, str] = 'auto',
        sampling: str = 'proportional', **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=nheads, dim_feedforward=4 * in_dim,
            batch_first=True, dropout=0.0,
        )
        self.bank = SEERBank(
            d_M=d_M,
            k=k,
            lambda_=lambda_,
            max_capacity=max_capacity,
            similarity_threshold=similarity_threshold,
            sampling=sampling, **kwargs
        )

    def forward(self, x: torch.Tensor, mode: str):
        if mode not in _MODE_FLAGS:
            raise ValueError(
                f"Illegal forward mode: {mode!r} (use one of {list(_MODE_FLAGS)})"
            )
        update, produce_output = _MODE_FLAGS[mode]

        x = self.encoder(x)
        xd = F.normalize(x.flatten(1), p=2, dim=-1)
        out = self.bank(xd, update=update, produce_output=produce_output)
        return x, out

class BoQPlusPlusBlock_BF(nn.Module):
    def __init__(
        self,
        in_dim: int,
        nheads: int = 8,
        d_M: Optional[int] = 200,
        k: int = 50,
        lambda_: float = 2.0,
        max_capacity: int = 1000,
        similarity_threshold = 'auto',
        sampling: str = 'proportional', **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=nheads, dim_feedforward=4 * in_dim,
            batch_first=True, dropout=0.0,
        )

        # need to define here in order to load weights, not fully used.
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)

        self.Mk, self.Mv = None, None # lazy initialized
        self.qproj, self.kproj, self.vproj = None, None, None
        self.d_M = d_M
        self.k = k
        self.lambda_ = lambda_
        self.max_capacity = max_capacity

        # when sampling from the down-projected vectorspace, the expected similarity changes
        assert self.cross_attn.kdim >= self.d_M, "Require number of non-zero indices per exemplar to be less than key-embedding dimension."
        self.similarity_threshold = self.d_M / self.cross_attn.kdim if similarity_threshold == 'auto' else similarity_threshold
        self.sampling = sampling
        
    def load_qkv_projs(self):
        in_proj_weight = self.cross_attn.in_proj_weight
        in_proj_bias = self.cross_attn.in_proj_bias
        self.qproj_w, self.kproj_w, self.vproj_w = torch.chunk(in_proj_weight, chunks=3, dim=0)
        self.qproj_b, self.kproj_b, self.vproj_b = torch.chunk(in_proj_bias, chunks=3, dim=0)
        return
    
    def sample(self, ks, vs):

        def sample_helper(x, difference : int):
            """samples `difference` sparse exemplars from x, probability of sampling index i is proportional to |x_i|.
            Each exemplar keeps `dM` coordinates drawn WITHOUT replacement; the remaining coordinates are zeroed."""
            d = x.shape[-1]
            weights = x.abs()
            idx = torch.multinomial(weights.expand(difference, -1), self.d_M)  # [difference, nnz], replacement=False
            exemplars = x.new_zeros(difference, d)
            rows = torch.arange(difference, device=x.device).unsqueeze(1)
            exemplars[rows, idx] = x[idx]
            return exemplars

        # iterate over batch dim
        for kk, vv in zip(ks, vs):
            # kk.shape [S, dk]
            for s, (k, v) in enumerate(zip(kk, vv)):
                # how many KEY exemplars already in bank s match this key above threshold
                # Mk[s] is a plain list of tensors that does not move with .to();
                # align it with the current key's device before matching.
                n = 0 if len(self.Mk[s]) == 0 else int(((torch.stack(self.Mk[s]).to(k.device) @ k) > self.similarity_threshold).sum())
                difference = self.k - n
                if difference > 0:
                    # top up with `difference` new sparse candidates from this token
                    self.Mk[s].extend(sample_helper(k, difference))
                    self.Mv[s].extend(sample_helper(v, difference))

    def forward_train(self, x : torch.Tensor):
        k, v = x @ self.kproj_w.T + self.kproj_b, x @ self.vproj_w.T + self.vproj_b # [B, S, dk], [B, S, dv]
        if not self.Mk:
            B, S, dk = k.shape
            self.Mk = [[] for _ in range(S)]
            self.Mv = [[] for _ in range(S)]

        self.sample(k, v)
        return
    
    def forward_test(self, x : torch.Tensor):
        q = x @ self.qproj_w.T + self.qproj_b # [B, S, dk]
        outs = []

        for qi in q: # iterate over batch dimension
            accu = []

            # build SEER representations on the fly
            for s in range(len(self.Mk)): # iterate over spatial dimension
                
                # Banks are plain lists (not buffers), so pin them to the query's
                # device to stay correct on CPU and GPU.
                Mks = torch.stack(self.Mk[s]).to(q.device) # shape [Lks, dk]
                Mvs = torch.stack(self.Mv[s]).to(q.device) # shape [Lks, dv]

                """
                    Unlike in BoQ where we build a Nq \times S score matrix which relates Query i to every
                    spatial position s, here we're not relating positions s to one another, just building per-spatial representations.
                """
                score = F.softmax(qi[s] @ Mks.T, dim=-1) # [Lks]
                out = score @ Mvs # [dv]
                accu.append(out)

            # TODO: is summing correct here? This likely shifts the distribution center s.t. we cannot use an out_proj afterwards.
            seer_repr = torch.stack(accu).sum(0) # [S, dv] -> [dv,]
            outs.append(seer_repr)

        outs = torch.stack(outs) # [B, dv]
        return outs
    
    def forward(self, x: torch.Tensor, mode: str):
        if mode not in _MODE_FLAGS:
            raise ValueError(
                f"Illegal forward mode: {mode!r} (use one of {list(_MODE_FLAGS)})"
            )
        outs = []
        x = self.encoder(x)
        if not self.kproj:
            self.load_qkv_projs()
        if mode == 'train':
            outs = self.forward_train(x)
        elif mode == 'test':
            outs = self.forward_test(x)
        else:
            raise NotImplementedError
        return x, outs

    
class BoQPlusPlus(nn.Module):
    """Stack of BoQ++ blocks. SEER hyperparameters apply to every block.

    Modes
    -----
    'train'        : fill the per-block banks; returns None.
    'test'         : query the banks; returns the L2-normalized concatenated
                     descriptor.
    'simultaneous' : do both in one pass (SEER's online setting).
    """

    def __init__(
        self,
        in_channels: int = 1024,
        proj_channels: int = 512,
        n_layers: int = 2,
        block_type : Literal['simple', 'brute_force'] = 'simple',

        # ---- SEER bank knobs (applied identically to every block) ----
        d_M: Optional[int] = None,
        k: int = 50,
        lambda_: float = 2.0,
        max_capacity: int = 10000,
        similarity_threshold: Union[float, str] = 'auto',
        sampling: str = 'proportional', **kwargs
    ):
        super().__init__(**kwargs)
        self.proj_c = nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = nn.LayerNorm(proj_channels)

        BoQModule = {
            'simple' : BoQPlusPlusBlock,
            'brute_force' : BoQPlusPlusBlock_BF
        }[block_type]

        in_dim = proj_channels
        self.boqs = nn.ModuleList([
            BoQModule(
                in_dim=in_dim,
                nheads=in_dim // 64,
                d_M=d_M,
                k=k,
                lambda_=lambda_,
                max_capacity=max_capacity,
                similarity_threshold=similarity_threshold,
                sampling=sampling,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mode: str = 'test') -> Optional[torch.Tensor]:
        # Follow the aggregator's own weights so the bank/buffers are built on
        # the same device as the input, on CPU or GPU alike.
        x = x.to(self.proj_c.weight.device)
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)

        outs = []
        for block in self.boqs:
            x, out = block(x, mode=mode)
            if out is not None:
                outs.append(out)

        if not outs:
            return None
        out = torch.cat(outs, dim=1)
        out = F.normalize(out, p=2, dim=-1)
        return out
    

