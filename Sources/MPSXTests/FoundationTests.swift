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

            return ["Y": result]
        }

        let texture = gpu.commandQueue.sync {
            compiledGraph(.NHWC(
                texture: inputTexture,
                tensor: compiledGraph.inputs["X"]!,
                in: $0
            ), in: $0).texture2D(pixelFormat: .rgba8Unorm, converter: gpu.imageConverter, in: $0)
        }

        try save(texture: texture, arg: 2)
    }

    func testCompiledGraphMultipleInputsAndMultipleOutputs() async throws {
        let gpu = GPU.default

        let compiledGraph = MPSCompiledGraph(device: gpu.device) { graph in
            let x = graph.placeholder(shape: [100], dataType: .float32, name: "X")
            let y = graph.placeholder(shape: [200], dataType: .float32, name: "Y")
            let z = graph.placeholder(shape: [300], dataType: .float32, name: "Z")

            let w = graph.concatTensors([x, y, z], dimension: 0, name: nil) * 0.5

            let min = w.min(axes: [0])
            let max = w.max(axes: [0])

            return [
                "min": min,
                "max": max,
                "w": w,
            ]
        }

        let x = (0 ..< 100).map { _ in Float.random(in: -200 ... -100) }
        let y = (0 ..< 200).map { _ in Float.random(in: 100 ... 200) }
        let z = (0 ..< 300).map { _ in Float.random(in: 500 ... 1000) }

        let w = (x + y + z).map { $0 / 2 }

        let results = gpu.commandQueue.sync { commandBuffer in
            compiledGraph(
                [
                    "X": .init(floats: x, device: gpu.device),
                    "Y": .init(floats: y, device: gpu.device),
                    "Z": .init(floats: z, device: gpu.device),
                ],
                in: commandBuffer
            ).mapValues {
                $0.synchronizedNDArray(in: commandBuffer)
            }
        }.mapValues {
            $0.floats
        }

        let eps: Float = 1e-8

        XCTAssert(results["min"]!.count == 1 && abs(results["min"]![0] - w.min()!) < eps)
        XCTAssert(results["max"]!.count == 1 && abs(results["max"]![0] - w.max()!) < eps)
        XCTAssert(results["w"]!.count == w.count && zip(results["w"]!, w).allSatisfy {
            abs($0.0 - $0.1) < eps
        })
    }
}
