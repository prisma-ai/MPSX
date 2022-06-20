import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape
    func reshape(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard #available(iOS 15.0, macOS 12.0, *) else {
            throw OnnxError.unsupportedOperator(node.opType)
        }

        guard let input = tensors(node.input(0)) else {
            throw OnnxError.invalidInput(node.name)
        }

        if let shape = constants(node.input(1))?.ints {
            return reshape(
                input,
                shape: shape.map { NSNumber(value: $0) },
                name: nil
            )
        }

        if let shape = tensors(node.input(1)) {
            return reshape(
                input,
                shapeTensor: cast(shape, to: .int32, name: ""),
                name: nil
            )
        }

        throw OnnxError.invalidInput(node.name)
    }
}

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Squeeze
    func squeeze(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard #available(iOS 15.0, macOS 12.0, *) else {
            throw OnnxError.unsupportedOperator(node.opType)
        }

        guard let input = tensors(node.input(0)) else {
            throw OnnxError.invalidInput(node.name)
        }

        let axes = node.attr(ints: "axes") ?? constants(node.input(1))?.ints ?? []

        return try squeeze(input: input, axes: axes)
    }

    func squeeze(input: MPSGraphTensor, axes: [Int]) throws -> MPSGraphTensor {
        guard let shape = input.shape else {
            throw OnnxError.invalidInput(#function)
        }

        if #available(iOS 15.4, macOS 12.3, *) {
            if axes.isEmpty {
                return squeeze(input, name: nil)
            }
            return squeeze(input, axes: axes.map { NSNumber(value: $0) }, name: nil)
        }

        let newShape: [NSNumber]

        if axes.isEmpty {
            newShape = shape.filter { $0 != 1 }
        } else {
            let axeSet: Set<Int> = Set(axes.map {
                $0 < 0 ? shape.count - $0 - 1 : $0
            })

            newShape = shape.enumerated().filter {
                if !axeSet.contains($0.offset) { return true }
                if $0.element == 1 { return false }
                preconditionFailure()
            }.map(\.element)
        }

        return reshape(input, shape: newShape, name: nil)
    }
}

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze
    func unsqueeze(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard #available(iOS 15.0, macOS 12.0, *) else {
            throw OnnxError.unsupportedOperator(node.opType)
        }

        guard let input = tensors(node.input(0)),
              let axes = node.attr(ints: "axes") ?? constants(node.input(1))?.ints
        else { throw OnnxError.invalidInput(node.name) }

        return try unsqueeze(input: input, axes: axes)
    }

    func unsqueeze(input: MPSGraphTensor, axes: [Int]) throws -> MPSGraphTensor {
        guard var shape = input.shape else {
            throw OnnxError.invalidInput(#function)
        }

        if #available(iOS 15.4, macOS 12.3, *) {
            return expandDims(input, axes: axes.map { NSNumber(value: $0) }, name: nil)
        }

        axes.forEach {
            shape.insert(1, at: $0)
        }

        return reshape(input, shape: shape, name: nil)
    }
}
