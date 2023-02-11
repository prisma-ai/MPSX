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

        let model = try OnnxModel(data: data(arg: 2)) // shufflenet-v2-12.onnx
        let labels = try String(data: data(arg: 3), encoding: .utf8)!.split(separator: "\n") // imagenet_classes.txt

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

        let inputTexture = try await inputTexture(arg: 1)

        let input: MPSGraphTensorData = gpu.commandQueue.sync {
            // ❕ This call is optional: first run of the graph is slower than the others, so for clear measurements we perform warm-up.

            graph.warmUp(in: $0)

            // ❕ MTLTexture -> MPSGraphTensorData transformation is tricky, so MPSX has a handy API for that.

            // ⚠️⚠️⚠️ This method automatically resizes input image to match input shape. Please keep in mind to feed unstretched square image to the graph for correct predictions. This behavior is model specific.
            return .NCHW(
                texture: inputTexture,
                tensor: graph.executable.inputs.first!.value,
                in: $0
            )
        }

        // STEP 4️⃣: measure and run

        func predict() -> MPSNDArray {
            gpu.commandQueue.sync {
                graph.executable(input, in: $0).synchronizedNDArray(in: $0)
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

        let model = try OnnxModel(data: data(arg: 1)) // candy-8.onnx

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

        let inputImage = try await inputTexture(arg: 2)

        let input: MPSGraphTensorData = gpu.commandQueue.sync {
            // ❕ This call is optional: first run of the graph is slower than the others, so for clear measurements we perform warm-up.

            graph.warmUp(in: $0)

            return .NCHW(
                texture: inputImage,
                tensor: graph.executable.inputs.first!.value,
                in: $0
            )
        }

        // STEP 4️⃣: measure and run

        func styleTransfer() -> MTLTexture {
            gpu.commandQueue.sync {
                graph.executable(input, in: $0).transposeNHWC(in: $0).texture2D(
                    pixelFormat: .rgba8Unorm,
                    converter: gpu.imageConverter,
                    in: $0
                )
            }
        }

        measure {
            _ = styleTransfer()
        }

        try save(texture: styleTransfer(), arg: 3)
    }
}
