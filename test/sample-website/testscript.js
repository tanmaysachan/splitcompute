async function awaitTensorf32(path, expected_shape) {
    response = await fetch(path);
    data = await response.blob();
    buffer = await data.arrayBuffer();
    vals = new Float32Array(buffer);
    return torch.tensor({ data: Array.from(vals)})
                .reshape(expected_shape);
}

class GPT2MLP {
    constructor(c_fc, c_fc_bias, c_proj, c_proj_bias) {
        this.c_fc = c_fc;
        this.c_fc_bias = c_fc_bias;
        this.c_proj = c_proj;
        this.c_proj_bias = c_proj_bias;
    }

    forward(x) {
        let h = torch.matmul(x, this.c_fc).add(this.c_fc_bias);
        h = torch.gelu(h);
        h = torch.matmul(h, this.c_proj).add(this.c_proj_bias);
        return h;
    }
}

class GPT2Attention {
    constructor(c_attn, c_attn_bias, c_proj, c_proj_bias) {
        this.c_attn = c_attn;
        this.c_attn_bias = c_attn_bias;
        this.c_proj = c_proj;
        this.c_proj_bias = c_proj_bias;
    }

    forward(x) {
        let qkv = torch.matmul(x, this.c_attn).add(this.c_attn_bias);
        
        let qkvs = torch.split(qkv, 1600, 2);

        let q = qkvs[0];
        let k = qkvs[1];
        let v = qkvs[2];
        
        // Real torch is likely doing implicit contiguous here
        k = k.reshape([5, 8, 25, 64]).transpose(1, 2).transpose(2, 3).contiguous();
        q = q.reshape([5, 8, 25, 64]).transpose(1, 2).contiguous();
        v = v.reshape([5, 8, 25, 64]).transpose(1, 2).contiguous();

        let att = torch.matmul(q, k).div(8);
        att = torch.softmax(att, 3);
        v = torch.contiguous(v);

        let y = torch.matmul(att, v);

        y = y.transpose(1, 2).contiguous().reshape([5, 8, 1600]);

        y = torch.matmul(y, this.c_proj).add(this.c_proj_bias);

        return y
    }
}

class GPT2LayerNorm {
    constructor(gamma, beta) {
        this.gamma = gamma;
        this.beta = beta;
    }

    forward(x) {
        const mean = x.mean(2, true);
        const std = x.std(2, true);
        return this.gamma.reshape([1, 1, 1600]).mul(x.sub(mean).div(std)).add(this.beta);
    }
}

class GPT2Block {
    constructor(layer_num) {
        this.layer_num = layer_num;
        this.layers_metadata = {
            'transformer.h.$.ln_1.weight.weights': [1600],
            'transformer.h.$.ln_1.bias.weights': [1600],
            'transformer.h.$.attn.c_attn.weight.weights': [1600, 4800],
            'transformer.h.$.attn.c_attn.bias.weights': [4800],
            'transformer.h.$.attn.c_proj.weight.weights': [1600, 1600],
            'transformer.h.$.attn.c_proj.bias.weights': [1600],
            'transformer.h.$.mlp.c_fc.weight.weights': [1600, 6400],
            'transformer.h.$.mlp.c_fc.bias.weights': [6400],
            'transformer.h.$.mlp.c_proj.weight.weights': [6400, 1600],
            'transformer.h.$.mlp.c_proj.bias.weights': [1600],
            'transformer.h.$.ln_2.weight.weights': [1600],
            'transformer.h.$.ln_2.bias.weights': [1600],
        };

        this.layers_config = {}

        for (const [key, value] of Object.entries(this.layers_metadata)) {
            var new_key = key.replace('$', this.layer_num);
            this.layers_config[new_key] = value;
        }

        this.layer_weights = {};
    }

    async load_weights() {
        for (const [key, value] of Object.entries(this.layers_config)) {
            this.layer_weights[key] = await awaitTensorf32(
                `http://localhost:8000/test/sample-website/ml-assets/GPT2-XL/${key}`,
                value
            );
            console.log("Loaded weights for layer " + this.layer_num + " " + key);
        }
    }

    forward(x) {
        var ln_1 = new GPT2LayerNorm(this.layer_weights[`transformer.h.${this.layer_num}.ln_1.weight.weights`],
                                  this.layer_weights[`transformer.h.${this.layer_num}.ln_1.bias.weights`]);
        var attn = new GPT2Attention(this.layer_weights[`transformer.h.${this.layer_num}.attn.c_attn.weight.weights`],
                                     this.layer_weights[`transformer.h.${this.layer_num}.attn.c_attn.bias.weights`],
                                     this.layer_weights[`transformer.h.${this.layer_num}.attn.c_proj.weight.weights`],
                                     this.layer_weights[`transformer.h.${this.layer_num}.attn.c_proj.bias.weights`]);

        var ln_2 = new GPT2LayerNorm(this.layer_weights[`transformer.h.${this.layer_num}.ln_2.weight.weights`],
                                  this.layer_weights[`transformer.h.${this.layer_num}.ln_2.bias.weights`]);

        var mlp = new GPT2MLP(this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_fc.weight.weights`],
                              this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_fc.bias.weights`],
                              this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_proj.weight.weights`],
                              this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_proj.bias.weights`]);

        var x_ = ln_1.forward(x);
        var attn_out = attn.forward(x_);
        var x = x.add(attn_out);
        var x_ = ln_2.forward(x);
        var x = mlp.forward(x_).add(x);
        return x;
    }
}

window.onload = async () => {
    if (!await torch.initWebGPUAsync()) {
        console.warn(`WebGPU is not supported.`);
    }
    console.log('WebGPU is supported.');

    // Load partial state
    const partial_state = await awaitTensorf32("http://localhost:8000/test/sample-website/ml-assets/GPT2-XL/partial_xl.bin",
                                               [5, 8, 1600]);

    // Load layer 1
    let out = partial_state;

    let numLayers = 3;
    let allBlocks = [];
    for (let i = 0; i < numLayers; i++) {
        let block = new GPT2Block(i);
        await block.load_weights();
        allBlocks.push(block);
    }

    for (let i = 0; i < numLayers; i++) {
        out = allBlocks[i].forward(out);
        console.log(out.sum().toArrayAsync());
        console.log(out.toArrayAsync());
    }

}
