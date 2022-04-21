import MetalPerformanceShadersGraph

public enum MPSGraphIO {
    // MARK: Public

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
        texture: MTLTexture,
        shape: [NSNumber],
        dataType: MPSDataType,
        scaler: MPSImageScale,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        let image = MPSTemporaryImage(
            commandBuffer: commandBuffer,
            imageDescriptor: .init(
                channelFormat: dataType.matchingImageChannelFormat,
                width: shape[3].intValue,
                height: shape[2].intValue,
                featureChannels: shape[1].intValue,
                numberOfImages: shape[0].intValue,
                usage: [.shaderRead, .shaderWrite]
            )
        )

        defer {
            image.readCount = 0
        }

        scaler.encode(
            commandBuffer: commandBuffer,
            sourceTexture: texture,
            destinationTexture: image.texture
        )

        let result = input(
            image: image,
            dataType: dataType,
            in: commandBuffer
        )

        return result
    }

    public static func input(
        image: MPSImage,
        dataType: MPSDataType?,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        let shape = [image.numberOfImages, image.height, image.width, image.featureChannels].map {
            NSNumber(value: $0)
        }

        let array = MPSTemporaryNDArray(
            commandBuffer: commandBuffer,
            descriptor: .init(
                dataType: dataType ?? image.featureChannelFormat.matchingDataType,
                shape: shape
            )
        )

        defer {
            array.readCount = 0
        }

        array.importData(
            with: commandBuffer,
            from: image.batchRepresentation(),
            offset: .init()
        )

        let result = transpose(
            inputData: .init(array),
            perm: (2, 3, 1, 2),
            commandBuffer: commandBuffer
        )

        return result
    }

    public static func output(
        data: MPSGraphTensorData,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSTemporaryImage {
        let outputData = transpose(
            inputData: data,
            perm: (1, 2, 2, 3),
            commandBuffer: commandBuffer
        )

        let image = MPSTemporaryImage(
            commandBuffer: commandBuffer,
            imageDescriptor: .init(
                channelFormat: outputData.dataType.matchingImageChannelFormat,
                width: outputData.shape[2].intValue,
                height: outputData.shape[1].intValue,
                featureChannels: outputData.shape[3].intValue,
                numberOfImages: outputData.shape[0].intValue,
                usage: [.shaderRead, .shaderWrite]
            )
        )

        image.readCount = .max

        outputData.mpsndarray().exportData(
            with: commandBuffer,
            to: image.batchRepresentation(),
            offset: .init()
        )

        return image
    }

    // MARK: Private

    /// The weirdest part of the graph execution: if we insert a transpose operation after the original placeholder, the graph execution will be much slower than it should be. Therefore, we perform transposition as a separate graph.
    private static func transpose(
        inputData: MPSGraphTensorData,
        perm: Quad<Int>,
        commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        let graph = MPSGraph.new()
        graph.options = .none

        let input = graph.placeholder(
            shape: inputData.shape,
            dataType: inputData.dataType,
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

        let outputData = graph.encode(
            to: commandBuffer,
            feeds: [input: inputData],
            targetTensors: [output],
            targetOperations: nil,
            executionDescriptor: nil
        )[output]!

        return outputData
    }
}
