import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile
    func tile(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let repeats = constants(node.input(1))?.ints
        else { throw OnnxError.invalidInput(node.name) }

        let output = tileTensor(input, withMultiplier: repeats.nsnumbers, name: nil)
        return output
    }
}
