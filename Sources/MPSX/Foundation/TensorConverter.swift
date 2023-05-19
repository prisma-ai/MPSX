import Foundation
import MetalPerformanceShadersGraph

final class TensorConverter {
    // MARK: Lifecycle

    init(cacheCapacity: Int) {
        cache = .init(cacheCapacity)
    }

    // MARK: Internal

    struct Step {
        let name: String
        let action: (MPSGraphTensor) -> MPSGraphTensor
    }

    static let `default` = TensorConverter(cacheCapacity: 42)

    func transform(
        _ input: MPSGraphTensorData,
        _ steps: [Step],
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        guard !steps.isEmpty else {
            return input
        }

        let key = CacheKey(
            deviceID: commandBuffer.device.registryID,
            inputShape: input.shape.map(\.intValue),
            inputDataType: input.dataType,
            stepNames: steps.map(\.name)
        )

        let graph = lock.execute {
            if let graph = cache.get(key) {
                return graph
            }

            let graph = MPSCompiledGraph(device: commandBuffer.device) {
                ["Y": steps.reduce($0.placeholder(shape: input.shape, dataType: input.dataType, name: "X")) { $1.action($0) }]
            }

            cache.put(key, graph)

            return graph
        }

        let output: MPSGraphTensorData = graph(input, in: commandBuffer)

        return output
    }

    // MARK: Private

    private struct CacheKey: Hashable, CustomStringConvertible {
        let deviceID: UInt64
        let inputShape: [Int]
        let inputDataType: MPSDataType
        let stepNames: [String]

        var description: String {
            "GPU ID: \(deviceID)\nInput shape: \(inputShape)\nInput data type: \(inputDataType)\nTransformations: \(stepNames.joined(separator: " -> "))"
        }
    }

    private let cache: LRUCache<CacheKey, MPSCompiledGraph>
    private let lock: Lock = newLock()
}
