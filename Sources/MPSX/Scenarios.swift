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
        scaler: MPSImageScale?,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSTemporaryImage {
        let feedTensors = (executable.feedTensors ?? []).reduce(into: [:]) {
            $0[$1.operation.name] = $1
        }

        assert(Set(inputTextures.keys).isSubset(of: Set(feedTensors.keys)))

        let inputsData: [String: MPSGraphTensorData] = inputTextures.reduce(into: [:]) {
            let tensor = feedTensors[$1.key]!
            let shape = tensor.shape!
            let texture = $1.value

            if let scaler = scaler {
                $0[$1.key] = MPSGraphIO.input(
                    texture: $1.value,
                    shape: shape,
                    dataType: tensor.dataType,
                    scaler: scaler,
                    in: commandBuffer
                )
            } else {
                assert(texture.height == shape[2].intValue && texture.width == shape[3].intValue)

                $0[$1.key] = MPSGraphIO.input(
                    image: .init(texture: $1.value, featureChannels: shape[1].intValue),
                    dataType: tensor.dataType,
                    in: commandBuffer
                )
            }
        }

        let outputs = encode(to: commandBuffer, inputsData: inputsData)

        assert(outputs.count == 1)

        let image = MPSGraphIO.output(
            data: outputs[0],
            in: commandBuffer
        )

        return image
    }

    func texture2DFrom(
        inputTextures: [String: MTLTexture],
        pixelFormat: MTLPixelFormat = .bgra8Unorm,
        scaler: MPSImageScale?,
        converter: MPSImageConversion,
        in commandBuffer: MPSCommandBuffer
    ) -> MTLTexture {
        let image = autoreleasepool {
            imageFrom(
                inputTextures: inputTextures,
                scaler: scaler,
                in: commandBuffer
            )
        }

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
