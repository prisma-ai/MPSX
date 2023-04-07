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
        _ tensors: [String: MPSGraphTensor],
        swizzled: Bool
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let weights = tensors(node.input(1)),
              ConvAutoPad.notSet(node.attr(s: "auto_pad"))
        else { throw OnnxError.invalidInput(node.name) }

        return try conv(
            input: input,
            weights: weights,
            bias: tensors(node.input(2)),
            groups: node.attr(i: "group"),
            strides: node.attr(ints: "strides"),
            dilations: node.attr(ints: "dilations"),
            pads: node.attr(ints: "pads"),
            swizzled: swizzled
        )
    }

    /// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedConv
    func fusedConv(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        swizzled: Bool
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
            pads: node.attr(ints: "pads"),
            swizzled: swizzled
        )

        if let z = tensors(node.input(3)) {
            output = output + z
        }

        switch node.attr(s: "activation") {
        case "Relu": output = reLU(with: output, name: nil)
        case "": break
        default: throw OnnxError.invalidInput(node.name)
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
        pads: [Int]?,
        swizzled: Bool
    ) throws -> MPSGraphTensor {
        let groups = groups ?? 1

        guard let shape = input.shape,
              let strides = (strides ?? [1, 1]).pair,
              let dilations = (dilations ?? [1, 1]).pair,
              let pads = (pads ?? [0, 0, 0, 0]).quad
        else {
            throw OnnxError.invalidInput(#function)
        }

        let convolution: MPSGraphTensor

        // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        // "When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a “depthwise convolution”."

        if groups != 1, groups == weights.shape?[swizzled ? 1 : 0].intValue {
            guard let descriptor = MPSGraphDepthwiseConvolution2DOpDescriptor(
                strideInX: strides.1,
                strideInY: strides.0,
                dilationRateInX: dilations.1,
                dilationRateInY: dilations.0,
                paddingLeft: pads.1,
                paddingRight: pads.3,
                paddingTop: pads.0,
                paddingBottom: pads.2,
                paddingStyle: .explicit,
                dataLayout: .NCHW,
                weightsLayout: .OIHW
            ) else { throw OnnxError.invalidInput(#function) }

            // weightsLayout == OIHW, but MPSGraphDepthwiseConvolution2DOpDescriptor declares 'O' index is channel multiplier index (see headers), so we need to transpose weights tensor
            // Maybe i'm wrong, but it works ¯\_(ツ)_/¯

            convolution = depthwiseConvolution2D(
                input,
                weights: swizzled ? weights : weights.transpose(0, 1),
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
                paddingLeft: pads.1,
                paddingRight: pads.3,
                paddingTop: pads.0,
                paddingBottom: pads.2,
                paddingStyle: .explicit,
                dataLayout: .NCHW,
                weightsLayout: .OIHW
            ) else { throw OnnxError.invalidInput(#function) }

            convolution = convolution2D(
                input,
                weights: weights,
                descriptor: descriptor,
                name: nil
            )
        }

        let output = bias.flatMap { convolution + appendDimsIfNeeded(to: $0, count: shape.count - 2) } ?? convolution

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
            outputPadding: node.attr(ints: "output_padding")
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
        outputPadding: [Int]?
    ) throws -> MPSGraphTensor {
        let groups = groups ?? 1

        guard let strides = (strides ?? [1, 1]).pair,
              let dilations = (dilations ?? [1, 1]).pair,
              let pads = (pads ?? [0, 0, 0, 0]).quad,
              let outputPadding = (outputPadding ?? [0, 0]).pair,
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
            paddingLeft: pads.1,
            paddingRight: pads.3,
            paddingTop: pads.0,
            paddingBottom: pads.2,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else {
            throw OnnxError.invalidInput(#function)
        }

        // output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

        let outputShape: [Int] = [
            inputShape.0, // N
            weightsShape.1 * groups, // C
            strides.0 * (inputShape.2 - 1) + outputPadding.0 + ((weightsShape.2 - 1) * dilations.0 + 1) - pads.0 - pads.2, // H
            strides.1 * (inputShape.3 - 1) + outputPadding.1 + ((weightsShape.3 - 1) * dilations.1 + 1) - pads.1 - pads.3, // W
        ]

        let convolution = convolutionTranspose2D(
            input,
            weights: weights,
            outputShape: outputShape.nsnumbers,
            descriptor: descriptor,
            name: nil
        )

        let output = bias.flatMap { convolution + appendDimsIfNeeded(to: $0, count: 2) } ?? convolution

        return output
    }
}
