import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose
    func permute(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let permutation = node.attr(ints: "perm")
        else { throw OnnxError.invalidInput(node.name) }

        return input.transpose(permutation)
    }
}
