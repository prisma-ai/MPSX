import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
    func reduceMean(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let axes = node.attr(ints: "axes") ?? constants(node.input(1))?.ints
        else { throw OnnxError.invalidInput(node.name) }

        let keepDims = (node.attr(i: "keepdims") ?? 1) > 0

        var output = mean(of: input, axes: axes.nsnumbers, name: nil)

        if !keepDims {
            output = output.squeeze()
        }

        return output
    }
}
