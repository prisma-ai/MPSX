# ONNX inference engine based on MPSGraph

![MPSX](logo.svg)

✅ This framework is our production solution at ([Prisma Labs](https://prisma-ai.com)).

⚠️ It's metal based, so you should be familiar with Metal API (+ metal performance shaders).

🔎 See [tests](/Sources/MPSXTests/OnnxTests.swift) for realworld examples.

To run your ONNX model with MPSX you need to perform several easy steps:

# Graph initialization

### 1) Create instance of your model:

``` swift
let model = try OnnxModel(data: <protobuf bytes>)
```

### 2) Create graph configuration:

``` swift
let config = OnnxGraphConfig(
    inputs: [
        "your_input1_name": .init(
            dims: [2: 512, 3: 512], // 0,1,2,3 -> NCHW, so here we specify Height and Width
            valuesRange: .init(-1, 1) // we assume runtime input has value range 0-1, but our model requires -1-1 range, so passing required range, MPSX automatically denormalize your input values
        )
    ],
    outputs: [
        "your_output_1_name": .init(valuesRange: .init(0, 255)), // output_1 will be normalized to 0-1 range, using actual range 0-255
        "your_output_2_name": .init(valuesRange: .init(-1, 1)), // output_2 will be normalized to 0-1 range, using actual range -1-1
    ],
    tensorsDataType: .fp16 // or .fp32
)
```

> this is a complete graph configuration example - each argument is either optional or has a default value.

### 3) Create graph instance passing onnx model, metal device and configuration:

``` swift
let graph = try OnnxGraph(
    model: model,
    device: <metal device>,
    config: config
)
```

# Inference (graph encoding)

### 1) Raw inputs/outputs:

``` swift
let outputs: [MPSGraphTensorData] = graph(
    <[String: MPSGraphTensorData]> // String key is a model corresponding input name
    in: <MPSCommandBuffer>
)
```

This method requires manual data transformation from/to MPSGraphTensorData. For example:

#### texture conversion to MPSGraphTensorData

``` swift
let input: MPSGraphTensorData = .NCHW(
    texture: <MTLTexture>,
    matching: <MPSGraphTensor>,
    in: <MPSCommandBuffer>
)
```

#### MPSGraphTensorData conversion to MPSTemporaryImage

``` swift
let image: MPSTemporaryImage = <MPSGraphTensorData>
    .nhwc(in: <MPSCommandBuffer>)
    .temporaryImage(in: <MPSCommandBuffer>)
```

#### raw floats from MPSGraphTensorData

``` swift
let array: MPSNDArray = <MPSGraphTensorData>.synchronizedNDArray(on: <MPSCommandBuffer>)

... // finish GPU work to read floats on CPU side

let floats = array.floats
```

### 2) Convenient texture-to-texture call

For image-to-image neural networks MPSX provides convenient API:

``` swift
let texture: MTLTexture = graph.texture2DFrom(
    inputTextures: [model.inputs[0].name: <MTLTexture>],
    pixelFormat: .rgba8Unorm,
    converter: <MPSImageConversion>,
    in: <MPSCommandBuffer>
)
```

# MPSGraph DSL

In addition to ONNX graphs, MPSX provides a convenient API for building [custom computational graphs](/Sources/MPSXTests/FoundationTests.swift#L16) similar to NumPy/PyTorch.

# Links

[MPSCommandBuffer explanation](https://geor.blog/mpscommandbuffer/)

# Optimizations

1) Use [ONNX simplifier](https://github.com/daquexian/onnx-simplifier)
2) Use [ONNX2MPSX.py](ONNX2MPSX.py) script for weights conversion (FP 16/32) and specific MPSGraph optimizations.

``` console
for f in $(find $1 -name "*.onnx"); do
    onnxsim $f $f
    python ONNX2MPSX.py --half --input $f --output $f
done;
```

# Limitations

MPSX...

1) supports limited set of ONNX operators
2) is PyTorch oriented - TF models converted to ONNX may not be supported
3) is available only on iOS 15+/macOS 12+

# Installation

Use SPM:

``` swift
dependencies: [
    .package(url: "https://github.com/prisma-ai/MPSX.git", .upToNextMajor(from: "1.3.0"))
]
```

# Authors

* [Geor Kasapidi](https://github.com/geor-kasapidi)
