import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater
    func greater(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let a = tensors(node.input(0)),
              let b = tensors(node.input(1))
        else { throw OnnxError.invalidInput(node.name) }

        return greaterThan(a, b, name: nil)
    }
}
