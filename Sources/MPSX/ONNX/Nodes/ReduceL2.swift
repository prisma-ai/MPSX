import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2
    func reduceL2(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let axes = node.attr(ints: "axes") ?? constants(node.input(1))?.ints
        else { throw OnnxError.invalidInput(node.name) }

        let keepDims = (node.attr(i: "keepdims") ?? 1) > 0
        var output = input.sum(axes: axes).squareRoot()

        // TODO: support noop_with_empty_axes

        if !keepDims {
            output = output.squeeze(axes)
        }

        return output
    }
}
