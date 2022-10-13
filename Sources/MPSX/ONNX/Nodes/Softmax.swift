import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
    func softmax(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        let axis = node.attr(i: "axis") ?? -1

        return softMax(with: input, axis: axis, name: nil)
    }
}
