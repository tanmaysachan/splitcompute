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

def partial_forward(model, tokens, till_layer):
    x = tokens.to('mps')
    full_fwd_out = model.forward(x, till_layer)
    return full_fwd_out

def process_input_text(text, enc):
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    return tokens

def load_gpt2_model(model_type='gpt2', layers_to_offload=3):
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    # create a from-scratch initialized minGPT model
    config = GPTConfig(**config_args)
    model = GPT(config, model_type)
    model = model.to('mps')
    sd = model.state_dict()

    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    enc = tiktoken.get_encoding('gpt2')

    # Dump layers locally for server to find
    layers_to_save = []
    for i in range(config.n_layer - 1, config.n_layer - 1 - layers_to_offload, -1):
        layers_to_save.extend([
            f'transformer.h.{i}.ln_1.weight',
            f'transformer.h.{i}.ln_1.bias',
            f'transformer.h.{i}.attn.c_attn.weight',
            f'transformer.h.{i}.attn.c_attn.bias',
            f'transformer.h.{i}.attn.c_proj.weight',
            f'transformer.h.{i}.attn.c_proj.bias',
            f'transformer.h.{i}.ln_2.weight',
            f'transformer.h.{i}.ln_2.bias',
            f'transformer.h.{i}.mlp.c_fc.weight',
            f'transformer.h.{i}.mlp.c_fc.bias',
            f'transformer.h.{i}.mlp.c_proj.weight',
            f'transformer.h.{i}.mlp.c_proj.bias',
        ])

    import itertools
    import struct
    import os

    for key, tensor in sd_hf.items():
        # Convert to numpy array and then to list
        if key in layers_to_save:
            weights = tensor.cpu().detach().numpy().tolist()
            try:
                flattened = list(itertools.chain(*weights))
            except:
                flattened = weights
            filename = f'./ml-assets/{model_type}/{key}.weights'

            if os.path.exists(filename):
                continue

            with open(filename, 'wb') as f:
                f.write(struct.pack('%sf' % len(flattened), *flattened))

    return model, sd, enc, config.__dict__

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
