import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape
    func shape(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              node.attribute.isEmpty
        else { throw OnnxError.invalidInput(node.name) }

        return input.shapeTensor.cast(to: input.dataType)
    }
}
