import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import MPSX
import XCTest

@available(macOS 12.0, *)
final class OnnxTests: XCTestCase {
    // https://github.com/onnx/models/tree/main/vision/classification/shufflenet
    func testShuffleNet() async throws {
        let model = try OnnxModel(data: data(bundlePath: "\(testResourcesPath)/shufflenet-v2-12.onnx"))
        let labels = try String(data: data(bundlePath: "\(testResourcesPath)/imagenet_classes.txt"), encoding: .utf8)!.split(separator: "\n")
        let inputTexture = try await texture(bundlePath: "\(testResourcesPath)/tiger.jpg")

        let gpu = GPU.default

        let graph = try MPSCompiledGraph(
            onnxModel: model,
            device: gpu.device,
            config: .init(
                inputs: [model.inputs[0].name: .init(isImage: true)],
                outputs: [model.outputs[0]: .init(valuesRange: [-10, 10])],
                tensorsDataType: .fp16
            )
        )

        gpu.commandQueue.sync {
            graph.warmUp(in: $0)
        }

        func predict() -> MPSGraphTensorData {
            gpu.commandQueue.sync {
                graph(.texture(inputTexture), in: $0)
            }
        }

        measure {
            _ = predict()
        }

        let rawData = predict()

        let ndarray = gpu.commandQueue.sync {
            rawData.synchronizedNDArray(in: $0)
        }

        let top = labels[ndarray.floats.enumerated().sorted(by: { $0.element > $1.element })[0].offset]

        XCTAssert(top == "tiger")
    }

    // https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style
    func testStyleTransfer() async throws {
        let model = try OnnxModel(data: data(bundlePath: "\(testResourcesPath)/candy-8.onnx"))
        let inputImage = try await texture(bundlePath: "\(testResourcesPath)/tiger.jpg")

        let gpu = GPU.default

        let graph = try MPSCompiledGraph(
            onnxModel: model,
            device: gpu.device,
            config: .init(
                inputs: [model.inputs[0].name: .init(isImage: true)],
                outputs: [model.outputs[0]: .init(valuesRange: .init(0, 255), isImage: true)],
                tensorsDataType: .fp16
            )
        )

        gpu.commandQueue.sync {
            graph.warmUp(in: $0)
        }

        func styleTransfer() -> MPSGraphTensorData {
            gpu.commandQueue.sync {
                graph(.texture(inputImage), in: $0)
            }
        }

        measure {
            _ = styleTransfer()
        }

        let rawData = styleTransfer()

        let outputTexture = gpu.commandQueue.sync {
            rawData.texture(pixelFormat: .rgba8Unorm, in: $0)
        }

        let reference = try await texture(bundlePath: "\(testResourcesPath)/candy-8-tiger.jpg")

        XCTAssert(compare(texture: outputTexture, with: reference))
    }
}
