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
        // Pass the shape as metadata, if it needs to be extracted from the response
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

