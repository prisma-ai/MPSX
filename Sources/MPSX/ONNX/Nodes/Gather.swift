import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
    func gather(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let indicies = tensors(node.input(1))
        else { throw OnnxError.invalidInput(node.name) }

        let axis = node.attr(i: "axis") ?? 0

        let output = gather(
            withUpdatesTensor: input,
            indicesTensor: indicies.cast(to: .int32),
            axis: axis,
            batchDimensions: 0,
            name: nil
        ).squeeze()
        return output
    }
}
