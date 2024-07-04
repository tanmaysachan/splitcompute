from transformers import GPT2LMHeadModel, GPT2Config
from dataclasses import dataclass
import torch
from torch import nn
import math
import torch.nn.functional as F
import tiktoken
from gpt2_model import GPT, GPTConfig

SD = None
MODEL = None
ENC = None


def process_input_text(text, till_layer):
    tokens = ENC.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    tokens = tokens.to('mps')
    full_fwd_out = MODEL.forward(tokens, till_layer=till_layer)
    return full_fwd_out


def load_gpt2_model(model_type='gpt2', layers_to_offload=3, copy_weights=True):
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

    config = GPTConfig(**config_args)
    if copy_weights:
        model = GPT(config, model_type)
        model = model.to('mps')
        sd = model.state_dict()

        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # Load model from HF for original weights
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
        
        global SD
        SD = sd_hf

        global MODEL
        MODEL = model

        global ENC
        ENC = tiktoken.get_encoding('gpt2')

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

    for key, tensor in SD.items():
        # Convert to numpy array and then to list
        if key in layers_to_save:
            filename = f'./ml-assets/{model_type}/{key}.weights'

            if os.path.exists(filename):
                continue

            weights = tensor.cpu().detach().numpy().tolist()
            try:
                flattened = list(itertools.chain(*weights))
            except:
                flattened = weights
            with open(filename, 'wb') as f:
                f.write(struct.pack('%sf' % len(flattened), *flattened))

    # Convert config to dict for passing around in API calls
    config = config.__dict__
    config['layers_to_offload'] = layers_to_offload

    return config
