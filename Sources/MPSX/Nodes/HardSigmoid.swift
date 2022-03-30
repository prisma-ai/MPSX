import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid
    func hardSigmoid(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        return hardSigmoid(
            input: input,
            alpha: node.attr(f: "alpha"),
            beta: node.attr(f: "beta")
        )
    }

    func hardSigmoid(
        input: MPSGraphTensor,
        alpha: Float?,
        beta: Float?
    ) -> MPSGraphTensor {
        maximum(
            constant(0, dataType: input.dataType),
            minimum(
                constant(1, dataType: input.dataType),
                addition(
                    multiplication(
                        constant(Double(alpha ?? 0.2), dataType: input.dataType),
                        input,
                        name: nil
                    ),
                    constant(Double(beta ?? 0.5), dataType: input.dataType),
                    name: nil
                ),
                name: nil
            ),
            name: nil
        )
    }
}
