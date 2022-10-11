import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
    func concat(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let axis = node.attr(i: "axis") else {
            throw OnnxError.invalidInput(node.name)
        }

        let tensorsToConcat: [MPSGraphTensor] = try node.input.map {
            guard let tensor = tensors[$0] else {
                throw OnnxError.invalidInput(node.name)
            }
            return tensor
        }

        let output = concatTensors(
            tensorsToConcat,
            dimension: axis,
            name: nil
        )

        return output
    }
}
