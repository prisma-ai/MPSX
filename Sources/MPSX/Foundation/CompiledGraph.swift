import Foundation
import MetalPerformanceShadersGraph

public final class MPSCompiledGraph {
    // MARK: Lifecycle

    public convenience init(
        device: MTLDevice,
        options: Options = .init(),
        body: (MPSGraph) throws -> [String: MPSGraphTensor]
    ) rethrows {
        let graph = MPSGraph(options: options.verboseLog ? .verbose : .none)

        let compilationDescriptor = MPSGraphCompilationDescriptor()

        if #available(iOS 15.4, macOS 12.3, *) {
            compilationDescriptor.optimizationLevel = options.hardwareOptimization ? .level1 : .level0
        }

        if options.runtimeTypeInference {
            compilationDescriptor.disableTypeInference()
        }

        let outputTensors = try autoreleasepool {
            try body(graph)
        }

        self.init(
            compilationDescriptor: compilationDescriptor,
            device: device,
            graph: graph,
            outputTensors: outputTensors
        )
    }

    internal init(
        compilationDescriptor: MPSGraphCompilationDescriptor? = nil,
        device: MTLDevice,
        graph: MPSGraph,
        outputTensors: [String: MPSGraphTensor]
    ) {
        let executable = autoreleasepool {
            graph.compile(
                with: .init(mtlDevice: device),
                feeds: graph.placeholderTensors.reduce(into: [:]) {
                    $0[$1] = .init(shape: $1.shape, dataType: $1.dataType)
                },
                targetTensors: outputTensors.map(\.value),
                targetOperations: nil,
                compilationDescriptor: compilationDescriptor
            )
        }
        executable.options = graph.options

        let outputKeys = outputTensors.reduce(into: [:]) { $0[$1.value.operation.name] = $1.key }

        inputs = (executable.feedTensors ?? []).reduce(into: [:]) { $0[$1.operation.name] = $1 }
        outputs = (executable.targetTensors ?? []).reduce(into: [:]) { $0[outputKeys[$1.operation.name] ?? $1.operation.name] = $1 }

        self.graph = graph
        self.executable = executable
        self.outputKeys = outputKeys
    }

    // MARK: Public

    public struct Options {
        // MARK: Lifecycle

        public init(
            hardwareOptimization: Bool = false,
            runtimeTypeInference: Bool = false,
            verboseLog: Bool = false
        ) {
            self.hardwareOptimization = hardwareOptimization
            self.runtimeTypeInference = runtimeTypeInference
            self.verboseLog = verboseLog
        }

        // MARK: Public

        public var hardwareOptimization: Bool = false
        public var runtimeTypeInference: Bool = false
        public var verboseLog: Bool = false
    }

    public enum Input {
        case texture(MTLTexture)
        case floats([Float], shape: [Int]? = nil)
        case data(MPSGraphTensorData)
    }

    public let inputs: [String: MPSGraphTensor]
    public let outputs: [String: MPSGraphTensor]

    public func callAsFunction(
        _ input: Input,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        encode(to: commandBuffer, inputs: [input])[0]
    }

    public func callAsFunction(
        _ inputs: [String: Input],
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        encode(to: commandBuffer, inputs: array(inputs))[0]
    }

    public func callAsFunction(
        _ input: Input,
        in commandBuffer: MPSCommandBuffer
    ) -> [String: MPSGraphTensorData] {
        dictionary(encode(to: commandBuffer, inputs: [input]))
    }

    public func callAsFunction(
        _ inputs: [String: Input],
        in commandBuffer: MPSCommandBuffer
    ) -> [String: MPSGraphTensorData] {
        dictionary(encode(to: commandBuffer, inputs: array(inputs)))
    }

    // MARK: Private

    private let graph: MPSGraph
    private let executable: MPSGraphExecutable
    private let outputKeys: [String: String]

    private func array(_ inputs: [String: Input]) -> [Input] {
        (executable.feedTensors ?? []).map {
            inputs[$0.operation.name]!
        }
    }

    private func dictionary(_ outputs: [MPSGraphTensorData]) -> [String: MPSGraphTensorData] {
        zip(executable.targetTensors ?? [], outputs).reduce(into: [:]) {
            assert($1.0.shape == $1.1.shape)

            $0[self.outputKeys[$1.0.operation.name] ?? $1.0.operation.name] = $1.1
        }
    }

    private func encode(to commandBuffer: MPSCommandBuffer, inputs: [Input]) -> [MPSGraphTensorData] {
        let _inputs: [(
            data: MPSGraphTensorData,
            ndArray: MPSTemporaryNDArray?
        )] = zip(executable.feedTensors ?? [], inputs).map { tensor, input in
            switch input {
            case let .data(data):
                return (data, nil)
            case let .floats(floats, shape: shape):
                return (.floats(floats, shape: shape, device: commandBuffer.device), nil)
            case let .texture(texture):
                let shape = tensor.ishape

                assert(shape.count == 4)

                let ndArray = texture.toNDArray(
                    dataType: tensor.dataType,
                    featureChannels: shape[3],
                    targetWidth: shape[2],
                    targetHeight: shape[1],
                    in: commandBuffer
                )

                return (.init(ndArray), ndArray)
            }
        }

        let outputs = autoreleasepool {
            executable.encode(
                to: commandBuffer,
                inputs: _inputs.map(\.data),
                results: nil,
                executionDescriptor: nil
            )
        }

        commandBuffer.rootCommandBuffer.addCompletedHandler { _ in
            _ = self // keep self alive until completion
        }

        _inputs.forEach { $0.ndArray?.readCount = 0 }

        return outputs
    }
}

public extension MPSCompiledGraph {
    var input: MPSGraphTensor {
        inputs.first!.value
    }

    var output: MPSGraphTensor {
        outputs.first!.value
    }
}

public extension MPSCompiledGraph {
    func warmUp(in commandBuffer: MPSCommandBuffer) {
        let randomInputs: [String: Input] = MPSCompiledGraph(device: commandBuffer.device) { g in
            inputs.mapValues { t in
                g.randomUniformTensor(withShape: t.shape ?? [], name: nil).cast(to: t.dataType)
            }
        }([:], in: commandBuffer).mapValues {
            .data($0)
        }

        let _: [String: MPSGraphTensorData] = self(randomInputs, in: commandBuffer)
    }
}
