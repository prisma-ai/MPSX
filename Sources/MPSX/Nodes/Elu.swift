import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Elu
    func elu(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        return elu(
            input: input,
            alpha: node.attr(f: "alpha")
        )
    }

    /// https://pytorch.org/docs/stable/generated/torch.nn.ELU.html
    func elu(
        input: MPSGraphTensor,
        alpha: Float?
    ) -> MPSGraphTensor {
        let zero = constant(0, dataType: input.dataType)

        var output = subtraction(
            exponent(
                with: minimum(zero, input, name: nil),
                name: nil
            ),
            constant(1, dataType: input.dataType),
            name: nil
        )

        if let alpha = alpha, alpha != 1 {
            output = multiplication(
                output,
                constant(Double(alpha), dataType: input.dataType),
                name: nil
            )
        }

        output = addition(
            maximum(zero, input, name: nil),
            output,
            name: nil
        )

        return output
    }
}
