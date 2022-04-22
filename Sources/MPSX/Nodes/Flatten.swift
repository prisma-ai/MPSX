import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten
    func flatten(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard #available(iOS 15.0, macOS 12.0, *) else {
            throw OnnxError.unsupportedOperator(node.opType)
        }

        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        let axis = node.attr(i: "axis") ?? 1

        return flatten2D(input, axis: axis, name: nil)
    }
}
