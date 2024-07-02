from flask import Flask, render_template, request
from load_ml_assets_gpt2 import load_gpt2_model, process_input_text, partial_forward
import base64
import ast

app = Flask(__name__)

model_type = 'gpt2-xl'
model, sd, enc, config = load_gpt2_model(model_type)
layers_to_offload = 2
config['layers_to_offload'] = layers_to_offload

def encode_text(text):
    tokens = process_input_text(text, enc)
    logits = partial_forward(model, tokens, till_layer=config['n_layer'] - config['layers_to_offload'])
    return logits

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_text = request.form['input_text']
        final_state = encode_text(input_text).cpu()
        # Printing final state for comparison
        return render_template('index.html', input_text=input_text,
                               encoded_text=final_state.__repr__() + ', shape=' + str(final_state.shape))
    return render_template('index.html', input_text='', encoded_text='')

@app.route('/get_gpt2_metadata', methods=['GET'])
def get_gpt2_assets():
    return config

@app.route('/get_gpt2_weights/<layer>', methods=['GET'])
def get_gpt2_weights(layer):
    # Return layer num
    with open(f'./ml-assets/{model_type}/{layer}', 'rb') as f:
        layer = f.read()
        encoded_layer = base64.b64encode(layer).decode('utf-8')
    return {'layer': encoded_layer}

@app.route('/get_gpt2_partial_state', methods=['GET'])
def get_gpt2_partial_state():
    with open(f'./ml-assets/{model_type}/partial_state.bin', 'rb') as f:
        partial_state = f.read()
        partial_state = base64.b64encode(partial_state).decode('utf-8')
    with open(f'./ml-assets/{model_type}/partial_state_metadata.txt', 'r') as f:
        # Stored as a python list
        partial_state_metadata = ast.literal_eval(f.read())
    return {'layer': partial_state, 'metadata': partial_state_metadata}

if __name__ == '__main__':
    app.run(debug=True)
