import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape
    func reshape(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)) else {
            throw OnnxError.invalidInput(node.name)
        }

        if let shape = constants(node.input(1))?.ints {
            return input.reshape(shape.nsnumbers)
        }

        if let shape = tensors(node.input(1)) {
            return input.reshape(shape.cast(to: .int32))
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
        guard let input = tensors(node.input(0)) else {
            throw OnnxError.invalidInput(node.name)
        }

        let axes = node.attr(ints: "axes") ?? constants(node.input(1))?.ints ?? []

        return input.squeeze(axes)
    }
}

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze
    func unsqueeze(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let axes = node.attr(ints: "axes") ?? constants(node.input(1))?.ints
        else { throw OnnxError.invalidInput(node.name) }

        return input.unsqueeze(axes)
    }
}
