import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#DepthToSpace
    func depthToSpace(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              input.quadShape != nil,
              let blocksize = node.attr(i: "blocksize")
        else { throw OnnxError.invalidInput(node.name) }

        let output = depth(
            toSpace2DTensor: input,
            widthAxis: 3,
            heightAxis: 2,
            depthAxis: 1,
            blockSize: blocksize,
            usePixelShuffleOrder: node.attr(s: "mode") == "CRD",
            name: nil
        )

        return output
    }
}
