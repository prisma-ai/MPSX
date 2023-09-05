import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LeakyRelu
    func leakyRelu(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        let alpha = node.attr(f: "alpha") ?? 0.01

        return leakyReLU(with: input, alpha: Double(alpha), name: nil)
    }
}
