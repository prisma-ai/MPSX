import Foundation
import MetalPerformanceShadersGraph

public final class MPSCompiledGraph {
    // MARK: Lifecycle

    public init(options: MPSGraphOptions = .none, device: MTLDevice, body: (MPSGraph) throws -> [MPSGraphTensor]) rethrows {
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
                targetTensors: outputTensors,
                targetOperations: nil,
                compilationDescriptor: nil
            )
        }
        executable.options = options

        self.graph = graph
        self.executable = executable
    }

    // MARK: Public

    public let graph: MPSGraph
    public let executable: MPSGraphExecutable
}
