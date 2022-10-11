import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu
    func prelu(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let x = tensors(node.input(0)),
              let alpha = constants(node.input(1))?.floats?.first
        else { throw OnnxError.invalidInput(node.name) }

        return leakyReLU(with: x, alpha: Double(alpha), name: nil)
    }
}
