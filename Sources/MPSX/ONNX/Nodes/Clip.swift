import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip
    func clip(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let min = tensors(node.input(1)),
              let max = tensors(node.input(2))
        else { throw OnnxError.invalidInput(node.name) }

        let output = clamp(input, min: min, max: max, name: nil)

        return output
    }
}
