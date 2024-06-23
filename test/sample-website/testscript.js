// Inject tag with id "inject-js"

// Function to await for tensor
async function awaitTensorf32(path, expected_shape) {
    response = await fetch(path);
    data = await response.blob();
    buffer = await data.arrayBuffer();
    vals = new Float32Array(buffer);
    return torch.tensor({ data: Array.from(vals)})
                .reshape(expected_shape);
}

class GPT2LayerNorm {
    constructor() {

    }
}

class GPT2Block {
    constructor(layer_num) {
        this.layer_num = layer_num;
        this.layers_metadata = {
            'transformer.h.0.ln_1.weight.weights': [768],
            'transformer.h.0.ln_1.bias.weights': [768],
            'transformer.h.0.attn.c_attn.weight.weights': [768, 2304],
            'transformer.h.0.attn.c_attn.bias.weights': [2304],
            'transformer.h.0.attn.c_proj.weight.weights': [768, 768],
            'transformer.h.0.attn.c_proj.bias.weights': [768],
            'transformer.h.0.mlp.c_fc.weight.weights': [768, 3072],
            'transformer.h.0.mlp.c_fc.bias.weights': [3072],
            'transformer.h.0.mlp.c_proj.weight.weights': [3072, 768],
            'transformer.h.0.mlp.c_proj.bias.weights': [768],
            'transformer.h.0.ln_2.weight.weights': [768],
            'transformer.h.0.ln_2.bias.weights': [768],
        };
        this.layer_weights = {};
    }

    async load_weights() {
        for (const [key, value] of Object.entries(this.layers_metadata)) {
            this.layer_weights[key] = await awaitTensorf32(
                `http://localhost:8000/test/sample-website/ml-assets/GPT2/${key}`,
                value
            );
            console.log("Loaded weights for layer " + this.layer_num + " " + key);
        }
    }

    forward(x) {
        const attn = this.attention(x);
        const mlp = this.mlp(attn);
        return mlp;
    }
}

window.onload = async () => {
    if (!await torch.initWebGPUAsync()) {
        console.warn(`WebGPU is not supported.`);
    }
    console.log('WebGPU is supported.');

    // Load partial state
    const partial_state = await awaitTensorf32("http://localhost:8000/test/sample-website/ml-assets/GPT2/partial.bin",
                                               [5, 8, 768]);
    console.log(partial_state.mean(2, true))
    // Load layer 1
    GPT2Block0 = new GPT2Block(0);
    await GPT2Block0.load_weights();
}
