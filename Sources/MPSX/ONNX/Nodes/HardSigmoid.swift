import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid
    func hardSigmoid(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let x = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        // y = max(0, min(1, alpha * x + beta))

        let alpha = node.attr(f: "alpha") ?? 0.2
        let beta = node.attr(f: "beta") ?? 0.5

        return max(0, min(1, alpha * x + beta))
    }
}
