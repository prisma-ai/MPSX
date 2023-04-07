import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where
    func whereOp(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let condition = tensors(node.input(0)),
              let x = tensors(node.input(1)),
              let y = tensors(node.input(2))
        else { throw OnnxError.invalidInput(node.name) }

        return select(predicate: condition, trueTensor: x, falseTensor: y, name: nil)
    }
}
