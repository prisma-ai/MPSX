import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout
    func dropout(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        if let rate = constants(node.input(1))?.floats?.first {
            return dropout(input, rate: Double(rate), name: nil)
        }

        if let rate = tensors(node.input(1)) {
            return dropout(input, rate: rate, name: nil)
        }

        throw OnnxError.invalidInput(node.name)
    }
}
