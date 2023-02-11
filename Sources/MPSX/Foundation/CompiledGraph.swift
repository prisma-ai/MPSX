import Foundation
import MetalPerformanceShadersGraph

public final class MPSCompiledGraph {
    // MARK: Lifecycle

    public init(
        options: MPSGraphOptions = .none,
        compilationDescriptor: MPSGraphCompilationDescriptor? = nil,
        device: MTLDevice,
        body: (MPSGraph) throws -> [String: MPSGraphTensor]
    ) rethrows {
        let graph = MPSGraph()
        graph.options = options

        let outputTensors = try autoreleasepool {
            try body(graph)
        }

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
        executable.options = options

        let outputKeys = outputTensors.reduce(into: [:]) { $0[$1.value.operation.name] = $1.key }

        inputs = (executable.feedTensors ?? []).reduce(into: [:]) { $0[$1.operation.name] = $1 }
        outputs = (executable.targetTensors ?? []).reduce(into: [:]) { $0[outputKeys[$1.operation.name] ?? $1.operation.name] = $1 }

        self.graph = graph
        self.executable = executable
        self.outputKeys = outputKeys
    }

    // MARK: Public

    public let inputs: [String: MPSGraphTensor]
    public let outputs: [String: MPSGraphTensor]

    /// single input -> single output
    public func callAsFunction(
        _ input: MPSGraphTensorData,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        encode(to: commandBuffer, inputs: [input])[0]
    }

    /// multiple inputs -> single output
    public func callAsFunction(
        _ inputs: [String: MPSGraphTensorData],
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        encode(to: commandBuffer, inputs: array(inputs))[0]
    }

    /// single input  -> multiple outputs
    public func callAsFunction(
        _ input: MPSGraphTensorData,
        in commandBuffer: MPSCommandBuffer
    ) -> [String: MPSGraphTensorData] {
        dictionary(encode(to: commandBuffer, inputs: [input]))
    }

    /// multiple inputs -> multiple outputs
    public func callAsFunction(
        _ inputs: [String: MPSGraphTensorData],
        in commandBuffer: MPSCommandBuffer
    ) -> [String: MPSGraphTensorData] {
        dictionary(encode(to: commandBuffer, inputs: array(inputs)))
    }

    // MARK: Private

    private let graph: MPSGraph
    private let executable: MPSGraphExecutable
    private let outputKeys: [String: String]

    private func array(_ inputs: [String: MPSGraphTensorData]) -> [MPSGraphTensorData] {
        (executable.feedTensors ?? []).compactMap {
            inputs[$0.operation.name]
        }
    }

    private func dictionary(_ outputs: [MPSGraphTensorData]) -> [String: MPSGraphTensorData] {
        zip(executable.targetTensors ?? [], outputs).reduce(into: [:]) {
            assert($1.0.shape == $1.1.shape)

            $0[self.outputKeys[$1.0.operation.name] ?? $1.0.operation.name] = $1.1
        }
    }

    private func encode(to commandBuffer: MPSCommandBuffer, inputs: [MPSGraphTensorData]) -> [MPSGraphTensorData] {
        autoreleasepool {
            executable.encode(
                to: commandBuffer,
                inputs: inputs,
                results: nil,
                executionDescriptor: nil
            )
        }
    }
}
