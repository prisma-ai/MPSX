import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSX
import XCTest

final class FoundationTests: XCTestCase {
    func testSqueeze() {
        assert([1, 3, 4, 5, 1].squeeze(axes: []) == [3, 4, 5])
        assert([1, 3, 4, 5, 1].squeeze(axes: [4]) == [1, 3, 4, 5])
        assert([1, 3, 4, 5, 1].squeeze(axes: [-1]) == [1, 3, 4, 5])
        assert([1, 1, 1].squeeze(axes: [1]) == [1, 1])
    }

    func testUnsqueeze() {
        do {
            var shape: [NSNumber] = [3, 4, 5]
            assert(shape.unsqueeze(axes: [0]) == [1, 3, 4, 5])
        }
        do {
            var shape: [NSNumber] = [3, 4, 5]
            assert(shape.unsqueeze(axes: [-1]) == [3, 4, 5, 1])
        }
        do {
            var shape: [NSNumber] = [3, 4, 5]
            assert(shape.unsqueeze(axes: [2, -1]) == [3, 4, 1, 5, 1])
        }
        do {
            var shape: [NSNumber] = [3, 4, 5]
            assert(shape.unsqueeze(axes: [2, 4, 5]) == [3, 4, 1, 5, 1, 1])
        }
    }

    func testCompiledGraphWithDSL() async throws {
        let gpu = GPU.default

        let compiledGraph = MPSCompiledGraph(device: gpu.device) { graph in
            let image = graph.imagePlaceholder(dataType: .float16, height: 1024, width: 1024, channels: 3, name: "X")

            return [1 - image] // invert
        }

        let inputTexture = try await gpu.textureLoader.newTexture(
            URL: .init(
                fileURLWithPath: CommandLine.arguments[1] // ⚠️⚠️⚠️ pass the input image path as a command line argument
            ),
            options: [.SRGB: false]
        )

        let texture = gpu.commandQueue.sync {
            compiledGraph.executable.encode(to: $0, inputs: [
                "X": .NHWC(
                    texture: inputTexture,
                    tensor: compiledGraph.executable.feeds["X"]!,
                    in: $0
                ),
            ])[0].texture2D(pixelFormat: .rgba8Unorm, converter: gpu.imageConverter, in: $0)
        }

        let outputImage = texture.cgImage(
            colorSpace: CGColorSpace(name: CGColorSpace.displayP3)!
        )

        try outputImage?.jpeg()?.write(
            to: .init(
                fileURLWithPath: CommandLine.arguments[2] // ⚠️⚠️⚠️ pass the output image path as a command line argument
            ),
            options: .atomic
        )
    }
}
