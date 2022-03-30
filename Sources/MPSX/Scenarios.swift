import MetalPerformanceShadersGraph

public extension OnnxGraph {
    func warmUp(in commandBuffer: MPSCommandBuffer, constant: Double = 0.5) {
        _ = encode(
            to: commandBuffer,
            inputsData: MPSGraphIO.constant(
                value: constant,
                for: graph.placeholderTensors,
                in: commandBuffer
            )
        )
    }

    func imageFrom(
        inputTextures: [String: MTLTexture],
        scaler: MPSImageScale,
        in commandBuffer: MPSCommandBuffer
    ) throws -> MPSTemporaryImage {
        let feedTensors = (executable.feedTensors ?? []).reduce(into: [:]) {
            $0[$1.operation.name] = $1
        }

        assert(Set(inputTextures.keys).isSubset(of: Set(feedTensors.keys)))

        let inputsData: [String: MPSGraphTensorData] = try inputTextures.reduce(into: [:]) {
            let tensor = feedTensors[$1.key]!

            $0[$1.key] = try MPSGraphIO.input(
                shape: tensor.shape,
                dataType: tensor.dataType,
                texture: $1.value,
                scaler: scaler,
                commandBuffer: commandBuffer
            )
        }

        let outputs = encode(to: commandBuffer, inputsData: inputsData)

        assert(outputs.count == 1)

        let image = try MPSGraphIO.output(
            tensor: outputs[0],
            commandBuffer: commandBuffer
        )

        return image
    }

    func texture2DFrom(
        inputTextures: [String: MTLTexture],
        pixelFormat: MTLPixelFormat = .bgra8Unorm,
        scaler: MPSImageScale,
        converter: MPSImageConversion,
        in commandBuffer: MPSCommandBuffer
    ) throws -> MTLTexture {
        let image = try imageFrom(
            inputTextures: inputTextures,
            scaler: scaler,
            in: commandBuffer
        )

        defer {
            image.readCount = 0
        }

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: image.width,
            height: image.height,
            mipmapped: false
        )

        textureDescriptor.usage = [.shaderRead, .shaderWrite]

        let texture = commandBuffer.device.makeTexture(descriptor: textureDescriptor)!

        converter.encode(
            commandBuffer: commandBuffer,
            sourceTexture: image.texture,
            destinationTexture: texture
        )

        return texture
    }
}
