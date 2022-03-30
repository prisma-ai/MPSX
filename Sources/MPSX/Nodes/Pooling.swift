import MetalPerformanceShadersGraph

enum PoolMode {
    case avg
    case max
}

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool
    func globalPool(
        _ mode: PoolMode,
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        return try globalPool(mode: mode, input: input)
    }

    func globalPool(
        mode: PoolMode,
        input: MPSGraphTensor
    ) throws -> MPSGraphTensor {
        guard let shape = input.quadShape else {
            throw OnnxError.invalidInput(#function)
        }

        guard let descriptor = MPSGraphPooling2DOpDescriptor(
            kernelWidth: shape.3,
            kernelHeight: shape.2,
            strideInX: 1,
            strideInY: 1,
            paddingStyle: .explicit,
            dataLayout: .NCHW
        ) else {
            throw OnnxError.invalidInput(#function)
        }

        switch mode {
        case .avg:
            return avgPooling2D(
                withSourceTensor: input,
                descriptor: descriptor,
                name: nil
            )
        case .max:
            return maxPooling2D(
                withSourceTensor: input,
                descriptor: descriptor,
                name: nil
            )
        }
    }
}

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
    func pool(
        _ mode: PoolMode,
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let kernelShape = node.attr(ints: "kernel_shape")
        else { throw OnnxError.invalidInput(node.name) }

        return try pool(
            mode: mode,
            input: input,
            kernelShape: kernelShape,
            strides: node.attr(ints: "strides"),
            pads: node.attr(ints: "pads")
        )
    }

    func pool(
        mode: PoolMode,
        input: MPSGraphTensor,
        kernelShape: [Int],
        strides: [Int]?,
        pads: [Int]?
    ) throws -> MPSGraphTensor {
        guard let strides = (strides ?? [1, 1]).pair,
              let kernelShape = kernelShape.pair
        else {
            throw OnnxError.invalidInput(#function)
        }

        guard let descriptor = MPSGraphPooling2DOpDescriptor(
            kernelWidth: kernelShape.1,
            kernelHeight: kernelShape.0,
            strideInX: strides.1,
            strideInY: strides.0,
            paddingStyle: .explicit,
            dataLayout: .NCHW
        ) else {
            throw OnnxError.invalidInput(#function)
        }

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

        switch mode {
        case .avg:
            return avgPooling2D(
                withSourceTensor: input,
                descriptor: descriptor,
                name: nil
            )
        case .max:
            return maxPooling2D(
                withSourceTensor: input,
                descriptor: descriptor,
                name: nil
            )
        }
    }
}
