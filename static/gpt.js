let BACKEND_URL = "http://127.0.0.1:5000"

async function awaitTensorf32(path, expected_shape) {
    response = await fetch(path);
    data = await response.json();
    let weights = atob(data['layer']);
    let buffer = new Uint8Array(weights.length);
    for (let i = 0; i < weights.length; i++) {
        buffer[i] = weights.charCodeAt(i);
    }
    let view = new Float32Array(buffer.buffer);

    if (expected_shape.length == 0) {
        let t = torch.tensor({ data: Array.from(view)});
        metadata = data['metadata'];
        if (metadata === undefined) {
            throw new Error("Metadata not found in response");
        }
        return t.reshape(metadata);
    }

    return torch.tensor({ data: Array.from(view)})
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
    constructor(c_attn, c_attn_bias, c_proj, c_proj_bias, batch_size, sequence_length, n_embd, n_head) {
        this.c_attn = c_attn;
        this.c_attn_bias = c_attn_bias;
        this.c_proj = c_proj;
        this.c_proj_bias = c_proj_bias;

        this.batch_size = batch_size;
        this.sequence_length = sequence_length;
        this.n_embd = n_embd;
        this.n_head = n_head;
    }

    forward(x) {
        let qkv = torch.matmul(x, this.c_attn).add(this.c_attn_bias);
        
        let qkvs = torch.split(qkv, this.n_embd, 2);

        let q = qkvs[0];
        let k = qkvs[1];
        let v = qkvs[2];
        
        // Real torch is likely doing implicit contiguous here
        k = k.reshape([this.batch_size, this.sequence_length, this.n_head, this.n_embd / this.n_head]).transpose(1, 2).transpose(2, 3).contiguous();
        q = q.reshape([this.batch_size, this.sequence_length, this.n_head, this.n_embd / this.n_head]).transpose(1, 2).contiguous();
        v = v.reshape([this.batch_size, this.sequence_length, this.n_head, this.n_embd / this.n_head]).transpose(1, 2).contiguous();

        let att = torch.matmul(q, k).div(8);
        
        att = torch.masked_fill(att, torch.triu(torch.ones([this.sequence_length, this.sequence_length]), 1), -1000000000);

        att = torch.softmax(att, 3);
        v = torch.contiguous(v);

        let y = torch.matmul(att, v);

        y = y.transpose(1, 2).contiguous().reshape([this.batch_size, this.sequence_length, this.n_embd]);

        y = torch.matmul(y, this.c_proj).add(this.c_proj_bias);

        return y
    }
}

class GPT2LayerNorm {
    constructor(gamma, beta, n_embd) {
        this.gamma = gamma;
        this.beta = beta;
        this.n_embd = n_embd;
    }

    forward(x) {
        const mean = x.mean(2, true);
        const std = x.std(2, true);
        return this.gamma.reshape([1, 1, this.n_embd]).mul(x.sub(mean).div(std)).add(this.beta);
    }
}

class GPT2Block {
    constructor(layer_num, batch_size, sequence_length, n_embd, n_head) {
        this.layer_num = layer_num;
        this.batch_size = batch_size;
        this.sequence_length = sequence_length;
        this.n_embd = n_embd;
        this.n_head = n_head;
        this.layers_metadata = {
            'transformer.h.$.ln_1.weight.weights': [n_embd],
            'transformer.h.$.ln_1.bias.weights': [n_embd],
            'transformer.h.$.attn.c_attn.weight.weights': [n_embd, n_embd * 3],
            'transformer.h.$.attn.c_attn.bias.weights': [n_embd * 3],
            'transformer.h.$.attn.c_proj.weight.weights': [n_embd, n_embd],
            'transformer.h.$.attn.c_proj.bias.weights': [n_embd],
            'transformer.h.$.mlp.c_fc.weight.weights': [n_embd, n_embd * 4],
            'transformer.h.$.mlp.c_fc.bias.weights': [n_embd * 4],
            'transformer.h.$.mlp.c_proj.weight.weights': [n_embd * 4, n_embd],
            'transformer.h.$.mlp.c_proj.bias.weights': [n_embd],
            'transformer.h.$.ln_2.weight.weights': [n_embd],
            'transformer.h.$.ln_2.bias.weights': [n_embd],
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
                `${BACKEND_URL}/get_gpt2_weights/${key}`,
                value
            );
            console.log("Loaded weights for layer " + this.layer_num + " " + key);
        }
    }

    forward(x) {
        var ln_1 = new GPT2LayerNorm(this.layer_weights[`transformer.h.${this.layer_num}.ln_1.weight.weights`],
                                     this.layer_weights[`transformer.h.${this.layer_num}.ln_1.bias.weights`],
                                     this.n_embd);
        var attn = new GPT2Attention(this.layer_weights[`transformer.h.${this.layer_num}.attn.c_attn.weight.weights`],
                                     this.layer_weights[`transformer.h.${this.layer_num}.attn.c_attn.bias.weights`],
                                     this.layer_weights[`transformer.h.${this.layer_num}.attn.c_proj.weight.weights`],
                                     this.layer_weights[`transformer.h.${this.layer_num}.attn.c_proj.bias.weights`],
                                     this.batch_size,
                                     this.sequence_length,
                                     this.n_embd,
                                     this.n_head);

        var ln_2 = new GPT2LayerNorm(this.layer_weights[`transformer.h.${this.layer_num}.ln_2.weight.weights`],
                                     this.layer_weights[`transformer.h.${this.layer_num}.ln_2.bias.weights`],
                                     this.n_embd);

        var mlp = new GPT2MLP(this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_fc.weight.weights`],
                              this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_fc.bias.weights`],
                              this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_proj.weight.weights`],
                              this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_proj.bias.weights`]);

        var x_ = ln_1.forward(x);
        var attn_out = attn.forward(x_);
        x = x.add(attn_out);
        x_ = ln_2.forward(x);
        var mlp_out = mlp.forward(x_);
        x = x.add(mlp_out);
        return x;
    }
}

window.onload = async () => {
    let resp = await fetch(BACKEND_URL + "/get_gpt2_metadata");
    let config = await resp.json();

    if (!await torch.initWebGPUAsync()) {
        console.warn(`WebGPU is not supported.`);
    }
    console.log('WebGPU is supported.');

    // Load partial state
    const partial_state = await awaitTensorf32(`${BACKEND_URL}/get_gpt2_partial_state`,
                                               []);
    
    // Load layer 1
    let out = partial_state;
    let batch_size = out.shape[0];
    let sequence_length = out.shape[1];

    let numLayers = 48;
    let allBlocks = [];
    for (let i = numLayers - 1; i > numLayers - 1 - config.layers_to_offload; i--) {
        let block = new GPT2Block(i, batch_size, sequence_length, config.n_embd, config.n_head);
        await block.load_weights();
        allBlocks.push(block);
    }

    allBlocks.reverse();

    for (let i = 0; i < allBlocks.length; i++) {
        out = allBlocks[i].forward(out);
    }

    // If there exists a div with id "encoded_text", then inject js into div with id "inject-js"
    if (document.getElementById("encoded_text")) {
        let data = await out.toArrayAsync();
        data = data.toString();
        // Get first 10 and last 10 elements
        let first10 = data.slice(0, 100);
        let last10 = data.slice(data.length - 100, data.length);

        let prettyprintedTensor = `tensor([${first10}, ..., ${last10}], shape=[${out.shape}])`;
        document.getElementById("inject-js").innerHTML = `
    <h2> Splitcompute Out (${config.layers_to_offload} layer(s) running on your browser!): </h2>
    <p class="form-text">
` + prettyprintedTensor + `</p>`
    }

}
