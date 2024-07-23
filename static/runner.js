$(document).on('submit','#encode',function(e)
{
    e.preventDefault();
    $.ajax({
        type:'POST',
        url:'/',
        data:{
            offline_layers:$("#offline_layers").val(),
            input_text:$("#input_text").val(),
        },
        success:function(response)
        {
            runner('gpt2', parseInt($("#offline_layers").val()), response)
        }
    })
});

/* Parity check */
$(document).on('submit','#encode',function(e)
{
    e.preventDefault();
    $.ajax({
        type:'POST',
        url:'/',
        data:{
            offline_layers:0,
            input_text:$("#input_text").val(), },
        success:function(response)
        {
            runner('gpt2', 0, response, true)
        }
    })
});

async function parityPrintTensor(t) {
    text = await prettyPrintTensor(t);
    innerHTML = `
<h2>Embedding Generated (parity check with PyTorch):</h2>
<p class="form-text">${text}</p>
`;
    document.getElementById("parity-check").innerHTML = innerHTML;
}

async function embeddingPrintTensor(t, layers_to_offload) {
    text = await prettyPrintTensor(t);
    innerHTML = `
<h2>Embedding Generated (embedding generated, ${layers_to_offload} layers running locally!):</h2>
<p class="form-text">${text}</p>
`;
    document.getElementById("embedding").innerHTML = innerHTML;
}

async function prettyPrintTensor(t) {
    data = await t.toArrayAsync();
    data = data.flat(2);
    let text = "Tensor = [";
    // First few, then last few + shape
    for (let i = 0; i < 5; i++) {
        text += data[i] + ", ";
    }
    text += "..., ";
    for (let i = data.length - 5; i < data.length; i++) {
        text += data[i] + ", ";
    }
    text += `; shape=[${t.shape}]]`;

    return text;
}

function prettyPrintBenchmark(time, layers_to_offload) {
    let innerHTML = `
<h5>Time taken to run ${layers_to_offload} layers:</h5>
<p class="form-text">${time} ms</p>
<h5> Time per layer:</h5>
<p class="form-text">${time / layers_to_offload} ms</p>
`;
    document.getElementById("benchmarking").innerHTML = innerHTML;
}

let loaderGlobal = undefined;

async function runner(model, layers_to_offload, response, parity_run=false) {
    // Benchmark
    let start = performance.now();
    if (model === "gpt2") {
        await gpt2_runner(layers_to_offload, response, parity_run);
    } else {
        throw new Error("Model not yet supported!");
    }
    let end = performance.now();
    
    if (!parity_run) prettyPrintBenchmark(end - start, layers_to_offload);
}


async function gpt2_runner(layers_to_offload, response, parity_run=false) {
    let resp = await fetch(BACKEND_URL + "/get_model_config");
    let config = await resp.json();

    if (!await torch.initWebGPUAsync()) {
        console.warn(`WebGPU is not supported.`);
    }
    console.log('WebGPU is supported.');
    let numLayers = config.n_layer;

    let layer_start = numLayers - 1 - layers_to_offload + 1;
    let layer_end = numLayers - 1;

    if (loaderGlobal === undefined || loaderGlobal.layer_start > layer_start) {
        // TODO: can cache so much better, just dont reload from scratch everytime
        let loader = new GPT2AsyncLoader(layer_start,
                                         layer_end,
                                         config.n_embd,
                                         config.n_head);
        loaderGlobal = loader;
    }

    let loader = loaderGlobal;

    while (loader.layers_loaded() < layers_to_offload) {
        console.log("Waiting for layers to load...");
        await new Promise(r => setTimeout(r, 1000));
    }

    let partial_state = response['partial_state'];
    let partial_state_shape = response['partial_state_shape'];
    let partial_state_tensor = loadTensorf32(partial_state, partial_state_shape);

    let out = loader.forward_from(partial_state_shape[0], partial_state_shape[1], partial_state_tensor, layer_start)

    if (parity_run) {
        parityPrintTensor(out);
    } else {
        embeddingPrintTensor(out, layers_to_offload);
    }
}
