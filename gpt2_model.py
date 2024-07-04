from transformers import GPT2LMHeadModel, GPT2Config
from dataclasses import dataclass
import torch
from torch import nn
import math
import torch.nn.functional as F
import tiktoken

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # some quirk of gpt
        self.bias = torch.tril(torch.ones(config.block_size, config.block_size))\
                         .view(1, 1, config.block_size, config.block_size)\
                         .to('mps')

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y
 

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, model_type='gpt2'):
        super().__init__()
        self.model_type = model_type
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        # TODO: Turning off to mimic model with no LM head.
        #   Getting a wte shape issue with GPT2Model class directly from HF, fix later
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, till_layer):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h[:till_layer]:
            x = block(x)

        self.dump_partial_state(x)

        for block in self.transformer.h[till_layer:]:
            x = block(x)

        # TODO: shouldn't be off
        # x = self.transformer.ln_f(x)
        return x

    def dump_partial_state(self, x):
        with open(f'./ml-assets/{self.model_type}/partial_state.bin', 'wb') as f:
            to_write = x.cpu().detach().numpy().tolist()
            import struct
            import itertools

            # Pack into bytes
            to_write = list(itertools.chain(*to_write))
            to_write = list(itertools.chain(*to_write))
            f.write(struct.pack('%sf' % len(to_write), *to_write))

        with open(f'./ml-assets/{self.model_type}/partial_state_metadata.txt', 'w') as f:
            f.write(str(list(x.shape)))
