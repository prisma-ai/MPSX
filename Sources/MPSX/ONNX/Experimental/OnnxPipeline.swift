import MetalPerformanceShadersGraph

extension Onnx_NodeProto: OnnxAttributesProvider {}

/// Experimental version of OnnxGraph with support for custom implementations for unknown operators
public final class OnnxPipeline {
    // MARK: Lifecycle

    /// Initialize pipeline instance
    /// - Parameters:
    ///   - model: onnx model
    ///   - device: metal device for graph compilation
    ///   - customNodes: user-provided implementations for unknown operations
    ///   - config: graph building configuration
    public init(
        model: OnnxModel,
        device: MTLDevice,
        customNodes: [String: OnnxCustomNode],
        config: OnnxGraphConfig = .init()
    ) throws {
        let onnxGraph = model.proto.graph

        // required for custom operations

        let valueInfo: [String: Onnx_TypeProto.Tensor] = onnxGraph.valueInfo.reduce(into: [:]) {
            if case let .tensorType(tensor) = $1.type.value {
                $0[$1.name] = tensor // this object contains information about shape, datatype, etc.
            }
        }

        let splitResult = customNodes.isEmpty ? [:] : model.split(by: Set(customNodes.keys))

        let populateTensorsOnTheFly = !splitResult.isEmpty

        let graphOptions: MPSGraphOptions = .none
        let compilationDescriptor: MPSGraphCompilationDescriptor? = nil
        let tensorsDataType = config.tensorsDataType.mpsDataType // fp16 or fp32

        var constants = model.initializer // weights, biases, etc.

        var pipeline: [PipelineStep] = []

        var mpsGraph = MPSGraph(options: graphOptions) // current mpsgraph -> will be overwritten if split result is not empty
        var mpsTensors: [String: MPSGraphTensor] // table of current mpsgraph tensors for faster lookup by name

        if populateTensorsOnTheFly {
            mpsTensors = [:]
        } else {
            mpsTensors = try model.initializer.mapValues {
                try mpsGraph.constant($0, targetDataType: tensorsDataType)
            }
        }

        // create placeholders

        for input in model.inputs {
            let options = config.inputs[input.name]

            let shape = input.shape.enumerated().map {
                options?.dims?[$0.offset] ?? Int($0.element)
            }

            mpsTensors[input.name] = mpsGraph.input(
                shape: shape.nsnumbers,
                dataType: tensorsDataType,
                valuesRange: options?.valuesRange,
                name: input.name
            )
        }

        // iterate over graph: onnx guarantees graph nodes are sorted topologically

        for node in onnxGraph.node {
            if populateTensorsOnTheFly {
                for input in node.input {
                    if let tensor = model.initializer[input], mpsTensors[input] == nil {
                        mpsTensors[input] = try mpsGraph.constant(tensor, targetDataType: tensorsDataType)
                    }
                }
            }

            let success = try mpsGraph.onnx(
                node: node,
                optimizedForMPS: model.optimizedForMPS,
                tensorsDataType: tensorsDataType,
                tensors: &mpsTensors,
                constants: &constants
            )

            if success {
                continue // known operation
            }

            // lookup for a user-provided implementation

            guard let customNode = customNodes[node.opType],
                  let subgraphOutputs = splitResult[node.name]
            else {
                throw OnnxError.unsupportedOperator(node.opType) // custom implementation not found
            }

            // generate node inputs using current mpsgraph instance and tensor table

            try node.input.forEach {
                guard let inputTensor = mpsTensors[$0] else {
                    throw OnnxError.invalidModel(reason: "Tensor named \($0) not found")
                }

                mpsTensors[$0] = customNode.preprocess(
                    inputTensor: inputTensor,
                    inputName: $0,
                    graph: mpsGraph
                )
            }

            // finalize current mps graph

            let outputTensors = subgraphOutputs.reduce(into: [:]) {
                $0[$1] = mpsTensors[$1]
            }

            let subgraph = MPSCompiledGraph(
                compilationDescriptor: compilationDescriptor,
                device: device,
                graph: mpsGraph,
                outputTensors: outputTensors
            )

            // create next mpsgraph instance and tensor table

            mpsGraph = .init(options: graphOptions)
            mpsTensors = [:]

            var customNames: [String: String] = [:]

            outputTensors.forEach {
                let placeholder = mpsGraph.placeholder(shape: $1.shape, dataType: $1.dataType, name: $0)

                mpsTensors[$0] = placeholder

                // ⚠️ mps can change the user-defined name: ex. dots (.) will be replaced with underscores (_)

                if placeholder.operation.name != $0 {
                    customNames[$0] = placeholder.operation.name
                }
            }

            // generate node outputs and placeholders

            let outputShapes = try node.output.map {
                guard let tensorInfo = valueInfo[$0], tensorInfo.hasShape else {
                    throw OnnxError.invalidModel(reason: "Shaped tensor named \($0) not found")
                }

                let shape = tensorInfo.shape.dim.map { Int($0.dimValue) }

                let (placeholder, tensor) = customNode.postprocess(
                    outputName: $0,
                    outputShape: shape,
                    requiredDataType: tensorsDataType,
                    graph: mpsGraph
                )

                if placeholder.operation.name != $0 {
                    customNames[$0] = placeholder.operation.name
                }

                guard (tensor.shape ?? []).map(\.intValue) == shape else {
                    throw OnnxError.incorrectCustomNodeImplementation(
                        opType: node.opType,
                        reason: "Shape of tensor named \($0) does not match the required \(shape)"
                    )
                }

                mpsTensors[$0] = tensor

                return shape
            }

            // pipeline++

            pipeline.append(.graph(subgraph))
            pipeline.append(.custom(.init(proto: node, outputShapes: outputShapes, customNames: customNames)))
        }

        // final step: setup onnx graph outputs

        let finalGraph = try MPSCompiledGraph(
            compilationDescriptor: compilationDescriptor,
            device: device,
            graph: mpsGraph,
            outputTensors: model.outputs.reduce(into: [:]) {
                guard let tensor = mpsTensors[$1] else {
                    throw OnnxError.invalidModel(reason: "Tensor named \($1) not found")
                }

                $0[$1] = mpsGraph.output(
                    tensor: tensor,
                    valuesRange: config.outputs[$1]?.valuesRange
                )
            }
        )

        pipeline.append(.graph(finalGraph))

        firstGraph = pipeline.first!.graph!
        lastGraph = pipeline.last!.graph!

        self.pipeline = pipeline
        self.customNodes = customNodes
    }

    // MARK: Public

    public var inputs: [String: MPSGraphTensor] {
        firstGraph.inputs
    }

    public var outputs: [String: MPSGraphTensor] {
        lastGraph.outputs
    }

    public var input: MPSGraphTensor {
        firstGraph.input
    }

    public var output: MPSGraphTensor {
        lastGraph.output
    }

    /// single input -> single output
    public func callAsFunction(
        _ input: MPSGraphTensorData,
        in commandBuffer: MPSCommandBuffer
    ) throws -> MPSGraphTensorData {
        try encode(inputs: [firstGraph.inputs.first!.key: input], in: commandBuffer).first!.value
    }

    /// multiple inputs -> single output
    public func callAsFunction(
        _ inputs: [String: MPSGraphTensorData],
        in commandBuffer: MPSCommandBuffer
    ) throws -> MPSGraphTensorData {
        try encode(inputs: inputs, in: commandBuffer).first!.value
    }

    /// single input  -> multiple outputs
    public func callAsFunction(
        _ input: MPSGraphTensorData,
        in commandBuffer: MPSCommandBuffer
    ) throws -> [String: MPSGraphTensorData] {
        try encode(inputs: [firstGraph.inputs.first!.key: input], in: commandBuffer)
    }

    /// multiple inputs -> multiple outputs
    public func callAsFunction(
        _ inputs: [String: MPSGraphTensorData],
        in commandBuffer: MPSCommandBuffer
    ) throws -> [String: MPSGraphTensorData] {
        try encode(inputs: inputs, in: commandBuffer)
    }

    // MARK: Private

    private enum PipelineStep {
        case graph(MPSCompiledGraph)
        case custom(CustomNode)

        // MARK: Internal

        struct CustomNode {
            let proto: Onnx_NodeProto
            let outputShapes: [[Int]]
            let customNames: [String: String]
        }

        var graph: MPSCompiledGraph? {
            switch self {
            case let .graph(value):
                return value
            case .custom:
                return nil
            }
        }
    }

    private let firstGraph: MPSCompiledGraph
    private let lastGraph: MPSCompiledGraph
    private let pipeline: [PipelineStep]

    private let customNodes: [String: OnnxCustomNode]

    private func encode(
        inputs: [String: MPSGraphTensorData],
        in commandBuffer: MPSCommandBuffer
    ) throws -> [String: MPSGraphTensorData] {
        var outputs = inputs

        for step in pipeline {
            try autoreleasepool {
                switch step {
                case let .graph(graph):
                    outputs = graph(outputs, in: commandBuffer)
                case let .custom(node):
                    let nodeOutputs = try self.customNodes[node.proto.opType]!.eval(
                        inputs: node.proto.input.map {
                            guard let output = outputs[$0] else {
                                throw OnnxError.incorrectCustomNodeImplementation(
                                    opType: node.proto.opType,
                                    reason: "Input named \($0) not found"
                                )
                            }
                            return output
                        },
                        outputShapes: node.outputShapes,
                        attributesProvider: node.proto,
                        in: commandBuffer
                    )

                    guard nodeOutputs.count == node.proto.output.count else {
                        throw OnnxError.incorrectCustomNodeImplementation(
                            opType: node.proto.opType,
                            reason: "Unexpected number of outputs"
                        )
                    }

                    // use customNames as onnx -> mps name table

                    outputs = outputs.reduce(into: [:]) {
                        $0[node.customNames[$1.key, default: $1.key]] = $1.value
                    }

                    zip(node.proto.output, nodeOutputs).forEach { outputs[node.customNames[$0, default: $0]] = $1 }
                }
            }
        }

        return outputs
    }
}

public extension OnnxPipeline {
    func warmUp(in commandBuffer: MPSCommandBuffer) {
        for step in pipeline {
            guard let graph = step.graph else {
                continue
            }

            let randomInputs: [String: MPSGraphTensorData] = MPSCompiledGraph(device: commandBuffer.device) { g in
                graph.inputs.mapValues { t in
                    g.randomUniformTensor(withShape: t.shape ?? [], name: nil).cast(to: t.dataType)
                }
            }([:], in: commandBuffer)

            let _: [String: MPSGraphTensorData] = graph(randomInputs, in: commandBuffer)
        }
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
