import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape
    func reshape(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let shape = tensors(node.input(1))
        else { throw OnnxError.invalidInput(node.name) }

        return reshape(
            input,
            shapeTensor: cast(shape, to: .int32, name: ""),
            name: nil
        )
    }
}
