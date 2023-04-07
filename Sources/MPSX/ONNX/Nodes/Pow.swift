import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#pow
    func pow(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let x = tensors(node.input(0)),
              let y = tensors(node.input(1))
        else { throw OnnxError.invalidInput(node.name) }

        return power(x, y, name: nil)
    }
}
