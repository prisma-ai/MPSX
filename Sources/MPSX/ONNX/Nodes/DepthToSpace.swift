import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#DepthToSpace
    func depthToSpace(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let shape = input.quadShape,
              let blocksize = node.attr(i: "blocksize")
        else { throw OnnxError.invalidInput(node.name) }

        let (b, c, h, w) = shape

        var x = input

        switch node.attr(s: "mode") {
        case "CRD":
            x = x.reshape([b, c / (blocksize * blocksize), blocksize, blocksize, h, w])
            x = x.transpose([0, 1, 4, 2, 5, 3])
        default: // "DCR"
            x = x.reshape([b, blocksize, blocksize, c / (blocksize * blocksize), h, w])
            x = x.transpose([0, 3, 4, 1, 5, 2])
        }

        x = x.reshape([b, c / (blocksize * blocksize), h * blocksize, w * blocksize])

        return x
    }
}
