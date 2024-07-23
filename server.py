from flask import Flask, render_template, request, abort
from gpt2_model import GPT, GPTConfig, DEBUG_STATE
from split_model import SplitModel
import tiktoken
import base64
import json
import torch
import struct
import os
import zlib

app = Flask(__name__)

split_model = None

model_type = 'gpt2-xl'

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        assert split_model is not None

        input_text = request.form['input_text']
        # Onus of JS to load this many layers
        offline_layers = int(request.form['offline_layers'])
        hash_value, _, partial_state = split_model.process_input(input_text, layers_offloaded=offline_layers)

        # Start to mid execution
        partial_state_elems = partial_state.cpu().detach().numpy().flatten().tolist()
        bstring = struct.pack('%sf' % len(partial_state_elems), *partial_state_elems)

        b64_encoded = base64.b64encode(bstring).decode('utf-8')

        # Hash value to track this query
        return {
            'hash': hash_value,
            'partial_state': b64_encoded,
            'partial_state_shape': list(partial_state.shape),
            'decoded_out': ''
        }

    return render_template('index.html', input_text='', encoded_text='')


### Routes for serving model assets ###
# runner.js's GPT-2 runner expects the following routes to be defined
@app.route('/get_model_config', methods=['GET'])
def get_model_config():
    return config.__dict__


@app.route('/get_model_weights/<layer>', methods=['GET'])
def get_model_weights(layer):
    assert split_model is not None

    weights_path = split_model.weights_path
    layer_path = os.path.join(weights_path, layer)

    if os.path.exists(layer_path):
        with open(layer_path, 'rb') as f:
            layer = f.read()
            encoded_layer = base64.b64encode(layer).decode('utf-8')

            return {'layer': encoded_layer}

    return abort(404)


if __name__ == '__main__':

    model_type = 'gpt2-xl'
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
    }[model_type]
    config_args['vocab_size'] = 50257
    config_args['block_size'] = 1024

    config = GPTConfig(**config_args)
    model = GPT(config, model_type)
    model = model.to('mps')

    # Init split model
    split_model = SplitModel(
        model_type,
        model,
        config,
        tiktoken.get_encoding('gpt2'),
    )

    app.run(debug=True)
