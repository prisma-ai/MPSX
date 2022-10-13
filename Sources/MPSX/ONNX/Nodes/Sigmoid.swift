import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid
    func sigmoid(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        return sigmoid(with: input, name: nil)
    }
}
