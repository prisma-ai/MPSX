import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Upsample
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
    func resize(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let shape = input.quadShape
        else { throw OnnxError.invalidInput(node.name) }

        let scales: Pair<Float> = (node.input.dropFirst().compactMap { constants[$0]?.floats?.quad }.first ?? node.attr(floats: "scales")?.quad).flatMap {
            ($0.2, $0.3)
        } ?? (1, 1)

        guard scales.0 != 1 || scales.1 != 1 else {
            return input
        }

        return input.resize(
            mode: node.attr(s: "mode") == "linear" ? .bilinear : .nearest,
            layout: .NCHW,
            height: Int((Float(shape.2) * scales.0).rounded()),
            width: Int((Float(shape.3) * scales.1).rounded())
        )
    }
}
