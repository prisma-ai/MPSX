import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import MPSX
import XCTest

final class FoundationTests: XCTestCase {
    func testCompiledGraphWithDSL() async throws {
        let gpu = GPU.default

        let inputTexture = try await inputTexture(arg: 1)

        let compiledGraph = MPSCompiledGraph(device: gpu.device) { graph in
            let image = graph.imagePlaceholder(dataType: .float16, height: 1024, width: 1024, channels: 3, name: "X")

            let mid = image.sum(axes: [3]) / 3

            let min = mid.min(axes: [1, 2]).squeeze()
            let max = mid.max(axes: [1, 2]).squeeze()

            let normalized = (mid - min) / (max - min)

            let resized = normalized.resize(
                mode: .nearest,
                layout: .NHWC,
                height: inputTexture.height,
                width: inputTexture.width
            )

            let result = (resized * 3).round() / 3

            return [result]
        }

        let texture = gpu.commandQueue.sync {
            compiledGraph.executable.encode(to: $0, inputs: [
                "X": .NHWC(
                    texture: inputTexture,
                    tensor: compiledGraph.executable.feeds["X"]!,
                    in: $0
                ),
            ])[0].texture2D(pixelFormat: .rgba8Unorm, converter: gpu.imageConverter, in: $0)
        }

        try save(texture: texture, arg: 2)
    }
}
