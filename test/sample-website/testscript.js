// Inject tag with id "inject-js"

window.onload = async () => {
    if (!await torch.initWebGPUAsync()) {
        console.warn(`WebGPU is not supported.`);
    }
    console.log('WebGPU is supported.');

    const a = torch.tensor([1.2, 2, 3]);
    const b = torch.tensor([4, 5, 6]);
    const c = a.add(b);
    
    const result = await c.toArrayAsync();
    console.log(result);

    // Try out matmul
    // const x = torch.tensor([[1, 2], [3, 4]]);
    // const y = torch.tensor([[1, 0], [0, 1]]);
    const x = torch.zeros([3, 4]);
    const y = torch.ones([4, 5]);
    console.log(x);
    console.log(y);
    const z = torch.matmul(x, y);
    const z_result = await z.toArrayAsync();
    console.log(z_result);

    fetch("http://localhost:8000/test/sample-website/ml-assets/GPT2/transformer.wte.weight.weights", {
        method: "GET",
    })
    .then(response => response.blob())
    .then(data => {
        data.arrayBuffer().then(buffer => {
            const vals = new Float32Array(buffer);

            loaded_tensor = torch.tensor({ data: Array.from(vals.slice(100)), shape: [50257, 768] });

            loaded_tensor2 = torch.t(torch.tensor({ data: Array.from(vals), shape: [50257, 768] }));

            loaded_tensor3 = torch.matmul(loaded_tensor, loaded_tensor2);

            loaded_tensor3.eager()

        })
    })
}
