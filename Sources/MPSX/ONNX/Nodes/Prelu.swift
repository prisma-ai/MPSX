import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu
    func prelu(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let x = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        if let alpha = tensors(node.input(1)) {
            return leakyReLU(with: x, alphaTensor: alpha, name: nil)
        }

        if let alpha = constants(node.input(1))?.floats?.first {
            return leakyReLU(with: x, alpha: Double(alpha), name: nil)
        }

        if let alpha = node.attr(f: "alpha") {
            return leakyReLU(with: x, alpha: Double(alpha), name: nil)
        }

        throw OnnxError.invalidInput(node.name)
    }
}
