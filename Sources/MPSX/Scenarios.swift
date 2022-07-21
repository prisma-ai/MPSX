import MetalPerformanceShadersGraph

public extension OnnxGraph {
    /// First run of the graph is slow, and for optimal runtime performance, call this method first. Optional.
    /// - Parameters:
    ///   - commandBuffer: current GPU command buffer instance
    ///   - constant: constant value in expected input range
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

    /// Graph launch with input textures and raw output temporary image. Performs automatic inputs scaling. Requires single output.
    /// - Parameters:
    ///   - inputTextures: table of input textures (input name -> texture)
    ///   - scaler: MPSImageScale instance for automatic texture scale
    ///   - commandBuffer: current GPU command buffer instance
    /// - Returns: Graph output as an MPSTemporaryImage instance
    func imageFrom(
        inputTextures: [String: MTLTexture],
        scaler: MPSImageScale,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSTemporaryImage {
        assert(Set(inputTextures.keys).isSubset(of: Set(feedTensors.keys)))

        let inputsData: [String: MPSGraphTensorData] = inputTextures.reduce(into: [:]) {
            let tensor = self.feedTensors[$1.key]!
            let shape = tensor.shape!
            let texture = $1.value

            if texture.height == shape[2].intValue, texture.width == shape[3].intValue {
                $0[$1.key] = MPSGraphIO.input(
                    image: .init(texture: $1.value, featureChannels: shape[1].intValue),
                    dataType: tensor.dataType,
                    in: commandBuffer
                )
            } else {
                $0[$1.key] = MPSGraphIO.input(
                    texture: $1.value,
                    shape: shape,
                    dataType: tensor.dataType,
                    scaler: scaler,
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

    /// Texture-to-texture graph launch. Performs automatic inputs scaling and output conversion to the required pixel format. Requires single output.
    /// - Parameters:
    ///   - inputTextures: table of input textures (input name -> texture)
    ///   - pixelFormat: target pixel format (.r8Unorm for 1-channel textures, .bgra8Unorm/.rgba8Unorm for 3/4-channels, etc.)
    ///   - scaler: MPSImageScale instance for automatic texture scale
    ///   - converter: MPSImageConversion instance for output image conversion
    ///   - commandBuffer: Graph output as an MPSTemporaryImage instance
    /// - Returns: Graph output as a CPU accessible texture
    func texture2DFrom(
        inputTextures: [String: MTLTexture],
        pixelFormat: MTLPixelFormat = .bgra8Unorm,
        scaler: MPSImageScale,
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
