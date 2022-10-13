import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Elu
    func elu(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let x = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        let alpha = node.attr(f: "alpha") ?? 1

        // https://pytorch.org/docs/stable/generated/torch.nn.ELU.html

        return max(0, x) + alpha * (exponent(with: min(0, x), name: nil) - 1)
    }
}
