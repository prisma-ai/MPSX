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
        config: OnnxGraphConfig = .init()
    ) throws {
        executable = try MPSCompiledGraph(device: device) { mpsGraph in
            let onnxGraph = model.proto.graph

            let tensorsDataType = config.tensorsDataType.mpsDataType

            var constants: [String: Onnx_TensorProto] = model.initializer

            var tensors: [String: MPSGraphTensor] = try constants.mapValues {
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

            for node in onnxGraph.node {
                let success = try mpsGraph.onnx(
                    node: node,
                    optimizedForMPS: model.optimizedForMPS,
                    tensorsDataType: tensorsDataType,
                    tensors: &tensors,
                    constants: &constants
                )

                if success {
                    continue
                }

                throw OnnxError.unsupportedOperator(node.opType)
            }

            return try model.outputs.reduce(into: [:]) {
                guard let tensor = tensors[$1] else {
                    throw OnnxError.invalidModel(reason: "tensor named \($1) not found")
                }

                $0[$1] = mpsGraph.output(
                    tensor: tensor,
                    valuesRange: config.outputs[$1]?.valuesRange
                )
            }
        }
    }

    // MARK: Public

    public var inputs: [String: MPSGraphTensor] {
        executable.inputs
    }

    public var outputs: [String: MPSGraphTensor] {
        executable.outputs
    }

    public var input: MPSGraphTensor {
        executable.input
    }

    public var output: MPSGraphTensor {
        executable.output
    }

    /// single input -> single output
    public func callAsFunction(
        _ input: MPSGraphTensorData,
        in commandBuffer: MPSCommandBuffer
    ) throws -> MPSGraphTensorData {
        executable(input, in: commandBuffer)
    }

    /// multiple inputs -> single output
    public func callAsFunction(
        _ inputs: [String: MPSGraphTensorData],
        in commandBuffer: MPSCommandBuffer
    ) throws -> MPSGraphTensorData {
        executable(inputs, in: commandBuffer)
    }

    /// single input  -> multiple outputs
    public func callAsFunction(
        _ input: MPSGraphTensorData,
        in commandBuffer: MPSCommandBuffer
    ) throws -> [String: MPSGraphTensorData] {
        executable(input, in: commandBuffer)
    }

    /// multiple inputs -> multiple outputs
    public func callAsFunction(
        _ inputs: [String: MPSGraphTensorData],
        in commandBuffer: MPSCommandBuffer
    ) throws -> [String: MPSGraphTensorData] {
        executable(inputs, in: commandBuffer)
    }

    // MARK: Private

    private let executable: MPSCompiledGraph
}

public extension OnnxGraph {
    func warmUp(in commandBuffer: MPSCommandBuffer) {
        let randomInputs: [String: MPSGraphTensorData] = MPSCompiledGraph(device: commandBuffer.device) { g in
            executable.inputs.mapValues { t in
                g.randomUniformTensor(withShape: t.shape ?? [], name: nil).cast(to: t.dataType)
            }
        }([:], in: commandBuffer)

        let _: [String: MPSGraphTensorData] = executable(randomInputs, in: commandBuffer)
    }

    func inputsFromTextures(_ dict: [String: MTLTexture], in commandBuffer: MPSCommandBuffer) -> [String: MPSGraphTensorData] {
        dict.reduce(into: [:]) {
            if let input = self.inputs[$1.key] {
                $0[$1.key] = .NCHW(texture: $1.value, matching: input, in: commandBuffer)
            } else {
                assertionFailure("no input with key \($1.key)")
            }
        }
    }

    func imageFrom(
        _ inputTexture: MTLTexture,
        pixelFormat: MTLPixelFormat? = nil,
        converter: MPSImageConversion? = nil,
        in commandBuffer: MPSCommandBuffer
    ) throws -> MPSTemporaryImage {
        try self(
            .NCHW(texture: inputTexture, matching: input, in: commandBuffer),
            in: commandBuffer
        ).nhwc(in: commandBuffer).temporaryImage2D(
            pixelFormat: pixelFormat,
            converter: converter,
            in: commandBuffer
        )
    }

    func texture2DFrom(
        _ inputTexture: MTLTexture,
        pixelFormat: MTLPixelFormat = .bgra8Unorm,
        converter: MPSImageConversion,
        in commandBuffer: MPSCommandBuffer
    ) throws -> MTLTexture {
        try self(
            .NCHW(texture: inputTexture, matching: input, in: commandBuffer),
            in: commandBuffer
        ).nhwc(in: commandBuffer).texture2D(
            pixelFormat: pixelFormat,
            converter: converter,
            in: commandBuffer
        )
    }

    func arrayFrom(
        _ inputTexture: MTLTexture,
        in commandBuffer: MPSCommandBuffer
    ) throws -> MPSNDArray {
        try self(
            .NCHW(texture: inputTexture, matching: input, in: commandBuffer),
            in: commandBuffer
        ).synchronizedNDArray(in: commandBuffer)
    }
}
