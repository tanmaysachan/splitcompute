async function gpt2_runner() {

    let time_start = performance.now();
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

    let layer_start = numLayers - 1 - config.layers_to_offload + 1;

    let layer_end = numLayers - 1;

    let loader = new GPT2AsyncLoader(layer_start,
                                     layer_end,
                                     batch_size,
                                     sequence_length,
                                     config.n_embd,
                                     config.n_head);


    while (loader.layersLoaded() < config.layers_to_offload) {
        // Wait for some time
        console.log("Waiting for layers to load...");
        await new Promise(r => setTimeout(r, 1000));
    }

    let time_processing = performance.now();
    out = loader.forward_from(out, layer_start);

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
    <p class="form-text" style="background-color:Tomato;">
    <b>Time taken for loading weights (One-time op): </b> ${time_processing - time_start} ms <br>
    <b>Time taken for processing: </b> ${performance.now() - time_processing} ms <br>
    <b>Time taken for total: </b> ${performance.now() - time_start} ms <br>
    <p class="form-text">
` + prettyprintedTensor + `</p>`
    }
}