import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu
    func prelu(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let alpha = constants(node.input(1))?.floats?.first
        else { throw OnnxError.invalidInput(node.name) }

        if #available(iOS 15.0, macOS 12.0, *) {
            return leakyReLU(with: input, alpha: Double(alpha), name: nil)
        } else {
            return prelu(input: input, alpha: alpha)
        }
    }

    /// https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
    func prelu(
        input: MPSGraphTensor,
        alpha: Float
    ) -> MPSGraphTensor {
        guard alpha != 1 else {
            return input
        }

        let zero = constant(0, dataType: input.dataType)

        return addition(
            maximum(zero, input, name: nil),
            multiplication(
                constant(Double(alpha), dataType: input.dataType),
                minimum(zero, input, name: nil),
                name: nil
            ),
            name: nil
        )
    }
}
