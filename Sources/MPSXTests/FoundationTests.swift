import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSX
import XCTest

final class FoundationTests: XCTestCase {
    func testDataArrayConversion() {
        let array1 = (0 ..< 10).map { _ in
            Float.random(in: -5 ... 5)
        }

        let data = array1.rawData

        let array2 = data.array(of: Float.self)

        XCTAssert(array1 == array2)
    }

    func testFastFloatConversion() {
        func _test<U: Numeric, V: Numeric>(range: Range<Int> = 0 ..< 10, body: ([U]) -> [V], compare _: (V, V) -> Bool) -> Bool {
            let shuffledRange = range.shuffled()
            let input = shuffledRange.map { U(exactly: $0)! }
            let expected = shuffledRange.map { V(exactly: $0)! }
            let output = body(input)
            return output.count == expected.count && zip(output, expected).allSatisfy {
                $0.0 == $0.1
            }
        }

        #if arch(arm64)
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

    func testCompiledGraphWithDSL() async throws {
        let gpu = GPU.default

        let inputTexture = try await texture(bundlePath: "\(testResourcesPath)/tiger.jpg")

        let compiledGraph = MPSCompiledGraph(
            device: gpu.device,
            options: .init(runtimeTypeInference: true)
        ) { graph in
            let image = graph.imagePlaceholder(
                dataType: .float32,
                height: -1,
                width: -1,
                channels: 3,
                name: "X"
            )

            let mid = image.sum(axes: [3]) / 3

            let min = mid.min(axes: [1, 2]).squeeze()
            let max = mid.max(axes: [1, 2]).squeeze()

            let normalized = (mid - min) / (max - min)

            let result = (normalized * 3).round() / 3

            return ["Y": result]
        }

        let outputTexture = gpu.commandQueue.sync {
            compiledGraph(.texture(inputTexture), in: $0).texture(pixelFormat: .r8Unorm, in: $0)
        }

        let reference = try await texture(bundlePath: "\(testResourcesPath)/dsl_reference.jpg")

        XCTAssert(compare(texture: outputTexture, with: reference))
    }

    func testStencilOperator() async throws {
        let gpu = GPU.default

        let inputTexture = try await texture(bundlePath: "\(testResourcesPath)/tiger.jpg")

        let compiledGraph = MPSCompiledGraph(
            device: gpu.device,
            options: .init(runtimeTypeInference: true)
        ) { graph in
            let image = graph.imagePlaceholder(
                dataType: .float16,
                height: -1,
                width: -1,
                channels: 3,
                name: "input_image"
            )

            // https://developer.apple.com/videos/play/wwdc2021/10152/?time=1489

            let edges = graph.stencil(
                withSourceTensor: image,
                weightsTensor: graph.const([
                    0, -1, 0,
                    -1, 4, -1,
                    0, -1, 0,
                ] as [Float], shape: [1, 3, 3, 1]),
                descriptor: .init(paddingStyle: .explicit)!,
                name: nil
            )

            let final = edges.resize(
                mode: .nearest,
                layout: .NHWC,
                height: inputTexture.height,
                width: inputTexture.width
            )

            return ["output_image": final]
        }

        let outputTexture = gpu.commandQueue.sync {
            compiledGraph(.texture(inputTexture), in: $0).texture(pixelFormat: .rgba8Unorm, in: $0)
        }

        let reference = try await texture(bundlePath: "\(testResourcesPath)/stencil_ref.jpg")

        XCTAssert(compare(texture: outputTexture, with: reference))
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
                    "X": .floats(x),
                    "Y": .floats(y),
                    "Z": .floats(z),
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

    func testPixelFormatFeatureChannels() {
        let possiblePixelFormats: [MTLPixelFormat] = [
            .r8Unorm,
            .rg8Unorm,
            .rgba8Unorm,
            .bgra8Unorm,
            .r8Unorm_srgb,
            .rg8Unorm_srgb,
            .rgba8Unorm_srgb,
            .bgra8Unorm_srgb,
            .r16Unorm,
            .rg16Unorm,
            .rgba16Unorm,
            .r16Float,
            .rg16Float,
            .rgba16Float,
            .r32Float,
            .rg32Float,
            .rgba32Float,
        ]

        GPU.default.commandQueue.sync { commandBuffer in
            for pixelFormat in possiblePixelFormats {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat, width: 1, height: 1, mipmapped: false)

                let tmp = MPSTemporaryImage(commandBuffer: commandBuffer, textureDescriptor: d)

                XCTAssert(tmp.featureChannels == pixelFormat.featureChannels)

                tmp.readCount = 0
            }
        }
    }
}
