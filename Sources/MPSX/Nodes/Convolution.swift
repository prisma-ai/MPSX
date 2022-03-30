import MetalPerformanceShadersGraph

enum ConvAutoPad: String {
    case NOTSET, SAME_UPPER, SAME_LOWER, VALID

    // MARK: Internal

    static func notSet(_ rawValue: String) -> Bool {
        switch Self(rawValue: rawValue) {
        case .SAME_LOWER,
             .SAME_UPPER,
             .VALID:
            return false
        default:
            return true
        }
    }
}

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
    func conv(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let weights = tensors(node.input(1)),
              ConvAutoPad.notSet(node.attr(s: "auto_pad"))
        else { throw OnnxError.invalidInput(node.name) }

        var output = try conv(
            input: input,
            weights: weights,
            bias: tensors(node.input(2)),
            groups: node.attr(i: "group"),
            strides: node.attr(ints: "strides"),
            dilations: node.attr(ints: "dilations"),
            pads: node.attr(ints: "pads")
        )

        switch node.attr(s: "activation") {
        case "Relu":
            output = reLU(with: output, name: nil)
        default:
            break
        }

        return output
    }

    func conv(
        input: MPSGraphTensor,
        weights: MPSGraphTensor,
        bias: MPSGraphTensor?,
        groups: Int?,
        strides: [Int]?,
        dilations: [Int]?,
        pads: [Int]?
    ) throws -> MPSGraphTensor {
        let groups = groups ?? 1

        guard let strides = (strides ?? [1, 1]).pair,
              let dilations = (dilations ?? [1, 1]).pair
        else {
            throw OnnxError.invalidInput(#function)
        }

        let convolution: MPSGraphTensor

        if groups != 1, groups == weights.shape?[0].intValue {
            guard let descriptor = MPSGraphDepthwiseConvolution2DOpDescriptor(
                strideInX: strides.1,
                strideInY: strides.0,
                dilationRateInX: dilations.1,
                dilationRateInY: dilations.0,
                paddingLeft: 0,
                paddingRight: 0,
                paddingTop: 0,
                paddingBottom: 0,
                paddingStyle: .explicit,
                dataLayout: .NCHW,
                weightsLayout: .OIHW
            ) else { throw OnnxError.invalidInput(#function) }

            if let pads = pads {
                guard let quad = pads.quad else {
                    throw OnnxError.invalidInput(#function)
                }

                descriptor.setExplicitPaddingWithPaddingLeft(
                    quad.1,
                    paddingRight: quad.3,
                    paddingTop: quad.0,
                    paddingBottom: quad.2
                )
            }

            // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            // "When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a “depthwise convolution”."
            // weightsLayout == OIHW, but MPSGraphDepthwiseConvolution2DOpDescriptor declares 'O' index is channel multiplier index (see headers), so we need to transpose weights tensor
            // Maybe i'm wrong, but it works ¯\_(ツ)_/¯

            convolution = depthwiseConvolution2D(
                input,
                weights: transposeTensor(weights, dimension: 0, withDimension: 1, name: nil),
                descriptor: descriptor,
                name: nil
            )
        } else {
            guard let descriptor = MPSGraphConvolution2DOpDescriptor(
                strideInX: strides.1,
                strideInY: strides.0,
                dilationRateInX: dilations.1,
                dilationRateInY: dilations.0,
                groups: groups,
                paddingStyle: .explicit,
                dataLayout: .NCHW,
                weightsLayout: .OIHW
            ) else { throw OnnxError.invalidInput(#function) }

            if let pads = pads {
                guard let quad = pads.quad else {
                    throw OnnxError.invalidInput(#function)
                }

                descriptor.setExplicitPaddingWithPaddingLeft(
                    quad.1,
                    paddingRight: quad.3,
                    paddingTop: quad.0,
                    paddingBottom: quad.2
                )
            }

            convolution = convolution2D(
                input,
                weights: weights,
                descriptor: descriptor,
                name: nil
            )
        }

        let output: MPSGraphTensor

        if let bias = bias {
            output = addition(convolution, reshapeHW(bias), name: nil)
        } else {
            output = convolution
        }

        return output
    }
}

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
    func convTranspose(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let weights = tensors(node.input(1)),
              let outputPadding = node.attr(ints: "output_padding"),
              ConvAutoPad.notSet(node.attr(s: "auto_pad"))
        else { throw OnnxError.invalidInput(node.name) }

        return try convTranspose(
            input: input,
            weights: weights,
            bias: tensors(node.input(2)),
            groups: node.attr(i: "group"),
            strides: node.attr(ints: "strides"),
            dilations: node.attr(ints: "dilations"),
            pads: node.attr(ints: "pads"),
            outputPadding: outputPadding
        )
    }

    func convTranspose(
        input: MPSGraphTensor,
        weights: MPSGraphTensor,
        bias: MPSGraphTensor?,
        groups: Int?,
        strides: [Int]?,
        dilations: [Int]?,
        pads: [Int]?,
        outputPadding: [Int]
    ) throws -> MPSGraphTensor {
        let groups = groups ?? 1

        guard let strides = (strides ?? [1, 1]).pair,
              let dilations = (dilations ?? [1, 1]).pair,
              let outputPadding = outputPadding.pair,
              let inputShape = input.quadShape,
              let weightsShape = weights.quadShape
        else {
            throw OnnxError.invalidInput(#function)
        }

        guard let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: strides.1,
            strideInY: strides.0,
            dilationRateInX: dilations.1,
            dilationRateInY: dilations.0,
            groups: groups,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else {
            throw OnnxError.invalidInput(#function)
        }

        let padY: Int
        let padX: Int

        if let pads = pads {
            guard let quad = pads.quad else {
                throw OnnxError.invalidInput(#function)
            }

            descriptor.setExplicitPaddingWithPaddingLeft(
                quad.1,
                paddingRight: quad.3,
                paddingTop: quad.0,
                paddingBottom: quad.2
            )

            padY = quad.0 + quad.2
            padX = quad.1 + quad.3
        } else {
            padY = 0
            padX = 0
        }

        // output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

        let outputShape: [Int] = [
            inputShape.0, // N
            weightsShape.1 * groups, // C
            strides.0 * (inputShape.2 - 1) + outputPadding.0 + ((weightsShape.2 - 1) * dilations.0 + 1) - padY, // H
            strides.1 * (inputShape.3 - 1) + outputPadding.1 + ((weightsShape.3 - 1) * dilations.1 + 1) - padX, // W
        ]

        let convolution = convolutionTranspose2D(
            input,
            weights: weights,
            outputShape: outputShape.map { NSNumber(value: $0) },
            descriptor: descriptor,
            name: nil
        )

        let output: MPSGraphTensor

        if let bias = bias {
            output = addition(convolution, reshapeHW(bias), name: nil)
        } else {
            output = convolution
        }

        return output
    }
}
