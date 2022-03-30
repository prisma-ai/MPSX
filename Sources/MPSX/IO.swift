import MetalPerformanceShadersGraph

public enum MPSGraphIO {
    // MARK: Public

    public enum Error: Swift.Error {
        case wrongShape
    }

    public static func constant(
        value: Double,
        for inputs: [MPSGraphTensor],
        in commandBuffer: MPSCommandBuffer
    ) -> [String: MPSGraphTensorData] {
        assert(inputs.allSatisfy { $0.shape != nil })

        let graph = MPSGraph.new()
        graph.options = .none

        let outputs = inputs.reduce(into: [:]) {
            $0[$1] = graph.constant(value, shape: $1.shape!, dataType: $1.dataType)
        }

        let values = graph.encode(
            to: commandBuffer,
            feeds: [:],
            targetTensors: inputs.map { outputs[$0]! },
            targetOperations: nil,
            executionDescriptor: nil
        )

        let results = outputs.reduce(into: [:]) {
            $0[$1.key.operation.name] = values[$1.value]
        }

        return results
    }

    public static func input(
        shape: [NSNumber]?,
        dataType: MPSDataType,
        texture: MTLTexture,
        scaler: MPSImageScale,
        commandBuffer: MPSCommandBuffer
    ) throws -> MPSGraphTensorData {
        guard let nchw = shape?.map(\.intValue).quad,
              nchw.0 == 1,
              nchw.1 > 0, nchw.1 <= 4,
              nchw.2 > 0, nchw.3 > 0
        else {
            throw Error.wrongShape
        }

        let nhwc = [nchw.0, nchw.2, nchw.3, nchw.1].map {
            NSNumber(value: $0)
        }

        return autoreleasepool {
            let image = MPSTemporaryImage(
                commandBuffer: commandBuffer,
                imageDescriptor: .init(
                    channelFormat: dataType.matchingImageChannelFormat,
                    width: nhwc[2].intValue,
                    height: nhwc[1].intValue,
                    featureChannels: nhwc[3].intValue,
                    numberOfImages: nhwc[0].intValue,
                    usage: [.shaderRead, .shaderWrite]
                )
            )

            scaler.encode(
                commandBuffer: commandBuffer,
                sourceTexture: texture,
                destinationTexture: image.texture
            )

            let array = MPSTemporaryNDArray(
                commandBuffer: commandBuffer,
                descriptor: .init(
                    dataType: dataType,
                    shape: nhwc
                )
            )

            array.importData(
                with: commandBuffer,
                from: image.batchRepresentation(),
                offset: .init()
            )

            image.readCount = 0

            let result = transpose(
                array: array,
                shape: nhwc,
                perm: (2, 3, 1, 2),
                commandBuffer: commandBuffer
            )

            array.readCount = 0

            return result
        }
    }

    public static func output(
        tensor data: MPSGraphTensorData,
        commandBuffer: MPSCommandBuffer
    ) throws -> MPSTemporaryImage {
        guard data.shape.count == 4 else {
            throw Error.wrongShape
        }

        let outputValue = autoreleasepool {
            transpose(
                array: data.mpsndarray(),
                shape: data.shape,
                perm: (1, 2, 2, 3),
                commandBuffer: commandBuffer
            )
        }

        let image = MPSTemporaryImage(
            commandBuffer: commandBuffer,
            imageDescriptor: .init(
                channelFormat: data.dataType.matchingImageChannelFormat,
                width: outputValue.shape[2].intValue,
                height: outputValue.shape[1].intValue,
                featureChannels: outputValue.shape[3].intValue,
                numberOfImages: outputValue.shape[0].intValue,
                usage: [.shaderRead, .shaderWrite]
            )
        )

        image.readCount = .max

        outputValue.mpsndarray().exportData(
            with: commandBuffer,
            to: image.batchRepresentation(),
            offset: .init()
        )

        return image
    }

    // MARK: Private

    /// The weirdest part of the graph execution: if we insert a transpose operation after the original placeholder, the graph execution will be much slower than it should be. Therefore, we perform transposition as a separate graph.
    private static func transpose(
        array: MPSNDArray,
        shape: [NSNumber],
        perm: Quad<Int>,
        commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        let graph = MPSGraph.new()
        graph.options = .none

        let input = graph.placeholder(
            shape: shape,
            dataType: array.dataType,
            name: nil
        )

        let output = graph.transposeTensor(
            graph.transposeTensor(
                input,
                dimension: perm.0,
                withDimension: perm.1,
                name: nil
            ),
            dimension: perm.2,
            withDimension: perm.3,
            name: nil
        )

        let result = graph.encode(
            to: commandBuffer,
            feeds: [input: .init(array)],
            targetTensors: [output],
            targetOperations: nil,
            executionDescriptor: nil
        )[output]!

        return result
    }
}
