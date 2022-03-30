import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Upsample
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
    func resize(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        return try resize(
            input: input,
            scales: node.input.dropFirst().compactMap { constants[$0]?.floats?.quad }.first ?? node.attr(floats: "scales")?.quad
        )
    }

    func resize(
        input: MPSGraphTensor,
        scales: Quad<Float>?
    ) throws -> MPSGraphTensor {
        guard let scales = scales, scales.2 != 1, scales.3 != 1 else {
            return input
        }

        guard let shape = input.quadShape else {
            throw OnnxError.invalidInput(#function)
        }

        return resize(
            input,
            size: [
                NSNumber(value: (Float(shape.2) * scales.2).rounded(.down)),
                NSNumber(value: (Float(shape.3) * scales.3).rounded(.down)),
            ],
            mode: .nearest,
            centerResult: false,
            alignCorners: true,
            layout: .NCHW,
            name: nil
        )
    }
}
