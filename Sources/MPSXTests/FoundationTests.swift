import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSX
import XCTest

final class FoundationTests: XCTestCase {
    /// test [Float] -> Data -> [Float] conversion
    func testDataArrayConversion() {
        let array1 = (0 ..< 10).map { _ in
            Float.random(in: -5 ... 5)
        }

        let data = array1.rawData

        let array2 = data.array(of: Float.self)

        XCTAssert(array1 == array2)
    }

    /// Test fast floats conversion
    func testFPC() {
        func _test<U: Numeric, V: Numeric>(range: Range<Int> = 0 ..< 10, body: ([U]) -> [V], compare _: (V, V) -> Bool) -> Bool {
            let shuffledRange = range.shuffled()
            let input = shuffledRange.map { U(exactly: $0)! }
            let expected = shuffledRange.map { V(exactly: $0)! }
            let output = body(input)
            return output.count == expected.count && zip(output, expected).allSatisfy {
                $0.0 == $0.1
            }
        }

        #if !arch(x86_64)
        XCTAssertTrue(_test(body: { (x: [Float]) in FPC._Float32_Float16(x) }, compare: { abs($0 - $1) < .ulpOfOne }))
        XCTAssertTrue(_test(body: { (x: [Float16]) in FPC._Float16_Float32(x) }, compare: { abs($0 - $1) < Float.ulpOfOne }))
        #endif
        XCTAssertTrue(_test(body: { (x: [Int8]) in FPC._Int8_Float32(x) }, compare: ==))
        XCTAssertTrue(_test(body: { (x: [Int16]) in FPC._Int16_Float32(x) }, compare: ==))
        XCTAssertTrue(_test(body: { (x: [Int32]) in FPC._Int32_Float32(x) }, compare: ==))
        XCTAssertTrue(_test(body: { (x: [UInt8]) in FPC._UInt8_Float32(x) }, compare: ==))
        XCTAssertTrue(_test(body: { (x: [UInt16]) in FPC._UInt16_Float32(x) }, compare: ==))
        XCTAssertTrue(_test(body: { (x: [UInt32]) in FPC._UInt32_Float32(x) }, compare: ==))
    }

    /// Test image processing using MPSCompiledGraph and MPSGraph DSL
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
                matching: compiledGraph.inputs["X"]!,
                in: $0
            ), in: $0).texture2D(pixelFormat: .rgba8Unorm, converter: gpu.imageConverter, in: $0)
        }

        // ⚠️ requires manual assertion

        try save(texture: texture, arg: 2)
    }

    /// Test MPSCompiledGraph multi input/multi output
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
                    "X": .floats(x, device: gpu.device),
                    "Y": .floats(y, device: gpu.device),
                    "Z": .floats(z, device: gpu.device),
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
