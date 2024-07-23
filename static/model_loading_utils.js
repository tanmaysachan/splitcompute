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
        return t.reshape(metadata['shape']);
    }

    return torch.tensor({ data: Array.from(view)})
           .reshape(expected_shape);
}

function loadTensorf32(data, expected_shape) {
    data = atob(data);
    let buffer = new Uint8Array(data.length);
    for (let i = 0; i < data.length; i++) {
        buffer[i] = data.charCodeAt(i);
    }
    let view = new Float32Array(buffer.buffer);
    return torch.tensor({ data: Array.from(view) })
           .reshape(expected_shape);
}

function floatArrayToBase64(floatArray) {
    // Create a buffer to hold the float values
    const buffer = new ArrayBuffer(floatArray.length * 4); // 4 bytes per float
    const floatView = new Float32Array(buffer);

    // Fill the buffer with float values
    floatArray.forEach((value, index) => {
        floatView[index] = value;
    });

    // Convert the buffer to a byte array
    const byteArray = new Uint8Array(buffer);

    // Convert the byte array to a string in chunks
    const chunkSize = 0x8000; // arbitrary chunk size (32K)
    let binary = '';
    for (let i = 0; i < byteArray.length; i += chunkSize) {
        binary += String.fromCharCode.apply(null, byteArray.subarray(i, i + chunkSize));
    }

    // Base64 encode the byte string
    b64_data = btoa(binary);
    return b64_data;
}
