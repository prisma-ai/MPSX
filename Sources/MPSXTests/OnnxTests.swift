import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import MPSX
import XCTest

@available(macOS 12.0, *)
final class OnnxTests: XCTestCase {
    // https://github.com/onnx/models/tree/main/vision/classification/shufflenet
    func testShuffleNet() async throws {
        // STEP 0️⃣: setup model and imagenet labels

        // ⚠️⚠️⚠️ You can find required files in 1.1.1 release attachments

        let model = try OnnxModel(data: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2]))) // shufflenet-v2-12.onnx
        let labels = try String(data: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[3])), encoding: .utf8)!.split(separator: "\n") // imagenet_classes.txt

        // STEP 1️⃣: setup metal stuff

        let gpu = GPU.default

        // STEP 2️⃣: create onnx graph using model instance, metal device and graph configuration

        let graph = try OnnxGraph(
            model: model,
            device: gpu.device,
            config: .init(
                outputs: [model.outputs[0]: .init(valuesRange: [-10, 10])],
                tensorsDataType: .fp16 // .fp32
            )
        )

        // STEP 3️⃣: prepare inputs and warm up graph

        let inputTexture = try await gpu.textureLoader.newTexture(
            URL: .init(
                fileURLWithPath: CommandLine.arguments[1] // ⚠️⚠️⚠️ pass the input image path as a command line argument
            ),
            options: [.SRGB: false]
        )

        let input: MPSGraphTensorData = gpu.commandQueue.sync {
            // ❕ This call is optional: first run of the graph is slower than the others, so for clear measurements we perform warm-up.

            graph.warmUp(in: $0)

            // ❕ MTLTexture -> MPSGraphTensorData transformation is tricky, so MPSX has a handy API for that.

            // ⚠️⚠️⚠️ This method automatically resizes input image to match input shape. Please keep in mind to feed unstretched square image to the graph for correct predictions. This behavior is model specific.
            return .NCHW(
                texture: inputTexture,
                tensorShape: graph.inputShapes.first!.value,
                tensorDataType: graph.inputDataTypes.first!.value,
                in: $0
            )
        }

        // STEP 4️⃣: measure and run

        func predict() -> MPSNDArray {
            gpu.commandQueue.sync {
                graph.encode(to: $0, inputs: [input])[0].synchronizedNDArray(in: $0)
            }
        }

        measure {
            _ = predict()
        }

        predict().floats.enumerated().sorted(by: { $0.element > $1.element }).prefix(3).forEach {
            print(labels[$0.offset])
        }
    }

    // https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style
    func testStyleTransfer() async throws {
        // STEP 0️⃣: setup model

        // ⚠️⚠️⚠️ You can find required files in 1.1.1 release attachments

        let model = try OnnxModel(data: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[3]))) // candy-8.onnx

        // STEP 1️⃣: setup metal stuff

        let gpu = GPU.default

        // STEP 2️⃣: create onnx graph using model instance, metal device and graph configuration

        let graph = try OnnxGraph(
            model: model,
            device: gpu.device,
            config: .init(
                outputs: [model.outputs[0]: .init(valuesRange: .init(0, 255))],
                tensorsDataType: .fp16 // .fp32
            )
        )

        // STEP 3️⃣: prepare inputs and warm up graph

        let inputTexture = try await gpu.textureLoader.newTexture(
            URL: .init(
                fileURLWithPath: CommandLine.arguments[1] // ⚠️⚠️⚠️ pass the input image path as a command line argument
            ),
            options: [.SRGB: false]
        )

        let input: MPSGraphTensorData = gpu.commandQueue.sync {
            // ❕ This call is optional: first run of the graph is slower than the others, so for clear measurements we perform warm-up.

            graph.warmUp(in: $0)

            return .NCHW(
                texture: inputTexture,
                tensorShape: graph.inputShapes.first!.value,
                tensorDataType: graph.inputDataTypes.first!.value,
                in: $0
            )
        }

        // STEP 4️⃣: measure and run

        func styleTransfer() -> MTLTexture {
            gpu.commandQueue.sync {
                graph.encode(to: $0, inputs: [input])[0].transposeNHWC(in: $0).texture2D(
                    pixelFormat: .rgba8Unorm,
                    converter: gpu.imageConverter,
                    in: $0
                )
            }
        }

        measure {
            _ = styleTransfer()
        }

        let outputImage = styleTransfer().cgImage(
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
