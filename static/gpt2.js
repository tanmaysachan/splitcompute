class GPT2MLP {
    constructor(c_fc, c_fc_bias, c_proj, c_proj_bias) {
        this.c_fc = c_fc;
        this.c_fc_bias = c_fc_bias;
        this.c_proj = c_proj;
        this.c_proj_bias = c_proj_bias;
    }

    forward(x) {
        let h = torch.matmul(x, this.c_fc.t()).add(this.c_fc_bias);
        h = torch.gelu(h);
        h = torch.matmul(h, this.c_proj.t()).add(this.c_proj_bias);
        return h;
    }
}

class GPT2Attention {
    constructor(c_attn, c_attn_bias, c_proj, c_proj_bias, n_embd, n_head) {
        this.c_attn = c_attn;
        this.c_attn_bias = c_attn_bias;
        this.c_proj = c_proj;
        this.c_proj_bias = c_proj_bias;

        this.n_embd = n_embd;
        this.n_head = n_head;
    }

    forward(x, b, t) {
        let qkv = torch.matmul(x, this.c_attn.t()).add(this.c_attn_bias);
        let qkvs = torch.split(qkv, this.n_embd, 2);

        let q = qkvs[0];
        let k = qkvs[1];
        let v = qkvs[2];

        // Real torch is likely doing implicit contiguous here
        k = k.reshape([b, t, this.n_head, this.n_embd / this.n_head]).transpose(1, 2).transpose(2, 3).contiguous();
        q = q.reshape([b, t, this.n_head, this.n_embd / this.n_head]).transpose(1, 2).contiguous();
        v = v.reshape([b, t, this.n_head, this.n_embd / this.n_head]).transpose(1, 2).contiguous();

        let att = torch.matmul(q, k).div(Math.sqrt(this.n_embd / this.n_head));

        att = torch.masked_fill(att, torch.triu(torch.ones([t, t]), 1), -1000000000);

        att = torch.softmax(att, 3);
        v = torch.contiguous(v);

        let y = torch.matmul(att, v);

        y = y.transpose(1, 2).contiguous().reshape([b, t, this.n_embd]);

        y = torch.matmul(y, this.c_proj.t()).add(this.c_proj_bias);

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
    constructor(layer_num, n_embd, n_head) {
        this.layer_num = layer_num;
        this.n_embd = n_embd;
        this.n_head = n_head;
        this.layers_metadata = {
            'transformer.h.$.ln_1.weight': [n_embd],
            'transformer.h.$.ln_1.bias': [n_embd],
            'transformer.h.$.attn.c_attn.weight': [n_embd * 3, n_embd ],
            'transformer.h.$.attn.c_attn.bias': [n_embd * 3],
            'transformer.h.$.attn.c_proj.weight': [n_embd, n_embd],
            'transformer.h.$.attn.c_proj.bias': [n_embd],
            'transformer.h.$.mlp.c_fc.weight': [n_embd * 4, n_embd],
            'transformer.h.$.mlp.c_fc.bias': [n_embd * 4],
            'transformer.h.$.mlp.c_proj.weight': [n_embd, n_embd * 4],
            'transformer.h.$.mlp.c_proj.bias': [n_embd],
            'transformer.h.$.ln_2.weight': [n_embd],
            'transformer.h.$.ln_2.bias': [n_embd],
        };

        this.layers_config = {}

        for (const [key, value] of Object.entries(this.layers_metadata)) {
            var new_key = key.replace('$', this.layer_num);
            this.layers_config[new_key] = value;
        }

        this.layer_weights = {};
    }

    async loadWeights() {
        for (const [key, value] of Object.entries(this.layers_config)) {
            this.layer_weights[key] = await awaitTensorf32(
                `${BACKEND_URL}/get_model_weights/${key}`,
                value
            );
            console.log("Loaded weights for layer " + this.layer_num + " " + key);
        }

        this.ln_1 = new GPT2LayerNorm(this.layer_weights[`transformer.h.${this.layer_num}.ln_1.weight`],
                                      this.layer_weights[`transformer.h.${this.layer_num}.ln_1.bias`],
                                      this.n_embd);
        this.attn = new GPT2Attention(this.layer_weights[`transformer.h.${this.layer_num}.attn.c_attn.weight`],
                                      this.layer_weights[`transformer.h.${this.layer_num}.attn.c_attn.bias`],
                                      this.layer_weights[`transformer.h.${this.layer_num}.attn.c_proj.weight`],
                                      this.layer_weights[`transformer.h.${this.layer_num}.attn.c_proj.bias`],
                                      this.n_embd,
                                      this.n_head);

        this.ln_2 = new GPT2LayerNorm(this.layer_weights[`transformer.h.${this.layer_num}.ln_2.weight`],
                                      this.layer_weights[`transformer.h.${this.layer_num}.ln_2.bias`],
                                      this.n_embd);

        this.mlp = new GPT2MLP(this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_fc.weight`],
                               this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_fc.bias`],
                               this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_proj.weight`],
                               this.layer_weights[`transformer.h.${this.layer_num}.mlp.c_proj.bias`]);
    }

    forward(b, t, x) {
        var x_ = this.ln_1.forward(x);
        var attn_out = this.attn.forward(x_, b, t);
        x = x.add(attn_out);
        x_ = this.ln_2.forward(x);
        var mlp_out = this.mlp.forward(x_);
        x = x.add(mlp_out);
        return x;
    }
}

class GPT2AsyncLoader {
    constructor(layer_start, layer_end, n_embd, n_head) {
        this.layer_start = layer_start;
        this.layer_end = layer_end;
        this.n_embd = n_embd;
        this.n_head = n_head;

        this.loaded_layers = [];
        this.start_loading();
    }

    async start_loading() {
        for (let layer = this.layer_end; layer >= this.layer_start; layer--) {
            let block = new GPT2Block(layer, this.n_embd, this.n_head);
            await block.loadWeights();
            this.loaded_layers.unshift(block);
        }
    }

    layers_loaded() {
        return this.loaded_layers.length;
    }

    forward_from(b, t, x, layer) {
        let index = layer - this.layer_start;

        for (let i = index; i < this.loaded_layers.length; i++) {
            x = this.loaded_layers[i].forward(b, t, x);
        }

        return x;
    }
}
