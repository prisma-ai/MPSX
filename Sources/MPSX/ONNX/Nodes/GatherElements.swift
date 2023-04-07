import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements
    func gatherElements(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let indicies = tensors(node.input(1))
        else { throw OnnxError.invalidInput(node.name) }

        let axis = node.attr(i: "axis") ?? 0

        if #available(iOS 15.4, macOS 12.3, *) {
            let output = gatherAlongAxis(
                axis,
                updates: input,
                indices: indicies.cast(to: .int32),
                name: nil
            )
            return output
        } else {
            throw OnnxError.unsupportedOperator(node.opType)
        }
    }
}
