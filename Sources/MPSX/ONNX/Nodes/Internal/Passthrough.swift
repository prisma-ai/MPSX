import MetalPerformanceShadersGraph

extension MPSGraph {
    func passthrough(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }
        return input
    }
}
