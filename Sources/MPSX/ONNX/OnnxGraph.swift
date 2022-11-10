import MetalPerformanceShadersGraph

public final class OnnxGraph {
    // MARK: Lifecycle

    /// Initialize graph instance
    /// - Parameters:
    ///   - model: onnx model
    ///   - device: metal device for graph compilation
    ///   - config: graph building configuration
    public init(
        model: OnnxModel,
        device: MTLDevice,
        config: OnnxGraphConfig,
        options: MPSGraphOptions = .none,
        compilationDescriptor: MPSGraphCompilationDescriptor? = nil
    ) throws {
        compiledGraph = try MPSCompiledGraph(
            options: options,
            compilationDescriptor: compilationDescriptor,
            device: device
        ) { mpsGraph in
            let onnxGraph = model.proto.graph

            let tensorsDataType: MPSDataType

            switch config.tensorsDataType {
            case .fp16: tensorsDataType = .float16
            case .fp32: tensorsDataType = .float32
            }

            var tensors: [String: MPSGraphTensor] = try model.initializer.mapValues {
                try mpsGraph.constant($0, targetDataType: tensorsDataType)
            }

            model.inputs.forEach { input in
                let options = config.inputs[input.name]

                let shape = input.shape.enumerated().map {
                    options?.dims?[$0.offset] ?? Int($0.element)
                }

                tensors[input.name] = mpsGraph.input(
                    shape: shape.nsnumbers,
                    dataType: tensorsDataType,
                    valuesRange: options?.valuesRange,
                    name: input.name
                )
            }

            var constants: [String: Onnx_TensorProto] = model.initializer

            for node in onnxGraph.node {
                let output: MPSGraphTensor

                switch node.opType {
                case "Add":
                    output = try mpsGraph.arithmetic(op: .add, node, tensors)
                case "Sub":
                    output = try mpsGraph.arithmetic(op: .sub, node, tensors)
                case "Mul":
                    output = try mpsGraph.arithmetic(op: .mul, node, tensors)
                case "Div":
                    output = try mpsGraph.arithmetic(op: .div, node, tensors)
                case "BatchNormalization":
                    output = try mpsGraph.batchNorm(node, tensors)
                case "InstanceNormalization":
                    output = try mpsGraph.instanceNorm(node, tensors)
                case "custom_group_norm": // onnx does not support group norm out of the box
                    output = try mpsGraph.groupNorm(node, tensors, constants)
                case "Concat":
                    output = try mpsGraph.concat(node, tensors)
                case "Conv":
                    output = try mpsGraph.conv(node, tensors)
                case "FusedConv":
                    output = try mpsGraph.fusedConv(node, tensors)
                case "ConvTranspose":
                    output = try mpsGraph.convTranspose(node, tensors)
                case "Gemm",
                     "MatMul":
                    output = try mpsGraph.gemm(node, tensors)
                case "GlobalAveragePool":
                    output = try mpsGraph.globalPool(.avg, node, tensors)
                case "AveragePool":
                    output = try mpsGraph.pool(.avg, node, tensors)
                case "MaxPool":
                    output = try mpsGraph.pool(.max, node, tensors)
                case "Pad":
                    output = try mpsGraph.pad(node, tensors, constants)
                case "Reshape":
                    output = try mpsGraph.reshape(node, tensors, constants)
                case "Squeeze":
                    output = try mpsGraph.squeeze(node, tensors, constants)
                case "Unsqueeze":
                    output = try mpsGraph.unsqueeze(node, tensors, constants)
                case "Shape":
                    output = try mpsGraph.shape(node, tensors)
                case "Relu":
                    output = try mpsGraph.relu(node, tensors)
                case "PRelu":
                    output = try mpsGraph.prelu(node, tensors, constants)
                case "Elu":
                    output = try mpsGraph.elu(node, tensors)
                case "Sigmoid":
                    output = try mpsGraph.sigmoid(node, tensors)
                case "HardSigmoid":
                    output = try mpsGraph.hardSigmoid(node, tensors)
                case "Upsample",
                     "Resize":
                    output = try mpsGraph.resize(node, tensors, constants)
                case "Tanh":
                    output = try mpsGraph.tanh(node, tensors)
                case "Softmax":
                    output = try mpsGraph.softmax(node, tensors)
                case "Flatten":
                    output = try mpsGraph.flatten(node, tensors)
                case "Transpose":
                    output = try mpsGraph.permute(node, tensors)
                case "Slice":
                    output = try mpsGraph.slice(node, tensors, constants)
                case "ReduceMean":
                    output = try mpsGraph.reduceMean(node, tensors, constants)
                case "Dropout":
                    output = try mpsGraph.dropout(node, tensors, constants)
                case "Constant":
                    guard let value = node.attr("value") else {
                        throw OnnxError.invalidInput(node.name)
                    }
                    node.output.forEach {
                        constants[$0] = value.t
                    }
                    output = try mpsGraph.constant(value.t, targetDataType: tensorsDataType)
                case "Cast",
                     "Clip":
                    output = try mpsGraph.passthrough(node, tensors)
                case "Split":
                    try mpsGraph.split(node, tensors).forEach {
                        tensors[$0.0] = $0.1
                    }
                    continue
                default:
                    throw OnnxError.unsupportedOperator(node.opType)
                }

                node.output.forEach {
                    tensors[$0] = output
                }
            }

            let outputs: [MPSGraphTensor] = try model.outputs.map {
                guard let tensor = tensors[$0] else {
                    throw OnnxError.invalidModel
                }

                return mpsGraph.output(
                    tensor: tensor,
                    valuesRange: config.outputs[$0]?.valuesRange
                )
            }

            return outputs
        }

        let feedTensors = (compiledGraph.executable.feedTensors ?? []).reduce(into: [:]) { $0[$1.operation.name] = $1 }
        let targetTensors = (compiledGraph.executable.targetTensors ?? []).reduce(into: [:]) { $0[$1.operation.name] = $1 }

        outputShapes = targetTensors.compactMapValues(\.shape).mapValues { $0.map(\.intValue) }
        inputShapes = feedTensors.compactMapValues(\.shape).mapValues { $0.map(\.intValue) }
        inputDataTypes = feedTensors.mapValues(\.dataType)
    }

    // MARK: Public

    public let outputShapes: [String: [Int]]
    public let inputShapes: [String: [Int]]
    public let inputDataTypes: [String: MPSDataType]

    public func encode(to commandBuffer: MPSCommandBuffer, inputs: [MPSGraphTensorData]) -> [MPSGraphTensorData] {
        compiledGraph.executable.encode(to: commandBuffer, inputs: inputs)
    }

    public func encode(to commandBuffer: MPSCommandBuffer, inputs: [String: MPSGraphTensorData]) -> [MPSGraphTensorData] {
        compiledGraph.executable.encode(to: commandBuffer, inputs: inputs)
    }

    // MARK: Private

    private let compiledGraph: MPSCompiledGraph
}

public extension OnnxGraph {
    func warmUp(in commandBuffer: MPSCommandBuffer) {
        _ = compiledGraph.executable.encode(
            to: commandBuffer,
            inputs: MPSCompiledGraph(device: commandBuffer.device) { g in
                (compiledGraph.executable.feedTensors ?? []).map { t in
                    g.randomUniformTensor(withShape: t.shape ?? [], name: nil).cast(to: t.dataType)
                }
            }.executable.encode(to: commandBuffer, inputs: [])
        )
    }

    func imageFrom(
        inputTextures: [String: MTLTexture],
        in commandBuffer: MPSCommandBuffer
    ) -> MPSTemporaryImage {
        encode(to: commandBuffer, inputs: inputTextures.reduce(into: [:]) {
            $0[$1.key] = .NCHW(
                texture: $1.value,
                tensorShape: inputShapes[$1.key]!, tensorDataType: inputDataTypes[$1.key]!, in: commandBuffer
            )
        })[0].transposeNHWC(in: commandBuffer).temporaryImage(in: commandBuffer)
    }

    func texture2DFrom(
        inputTextures: [String: MTLTexture],
        pixelFormat: MTLPixelFormat = .bgra8Unorm,
        converter: MPSImageConversion,
        in commandBuffer: MPSCommandBuffer
    ) -> MTLTexture {
        encode(to: commandBuffer, inputs: inputTextures.reduce(into: [:]) {
            $0[$1.key] = .NCHW(
                texture: $1.value,
                tensorShape: inputShapes[$1.key]!, tensorDataType: inputDataTypes[$1.key]!, in: commandBuffer
            )
        })[0].transposeNHWC(in: commandBuffer).texture2D(pixelFormat: pixelFormat, converter: converter, in: commandBuffer)
    }
}
