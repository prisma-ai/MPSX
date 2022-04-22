import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape
    func shape(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard #available(iOS 15.0, macOS 12.0, *) else {
            throw OnnxError.unsupportedOperator(node.opType)
        }

        guard let input = tensors(node.input(0)),
              node.attribute.isEmpty
        else { throw OnnxError.invalidInput(node.name) }

        return cast(shapeOf(input, name: nil), to: input.dataType, name: "")
    }
}
