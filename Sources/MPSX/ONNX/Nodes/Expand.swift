import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand
    func expand(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let shape = constants(node.input(1))?.ints
        else { throw OnnxError.invalidInput(node.name) }

        let output = input * constant(1.0, shape: shape.nsnumbers, dataType: input.dataType)
        return output
    }
}
