import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
    func gemm(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard var A = tensors(node.input(0)),
              var B = tensors(node.input(1))
        else { throw OnnxError.invalidInput(node.name) }

        // Y = alpha * A' * B' + beta * C

        if (node.attr(i: "transA") ?? 0) > 0 {
            A = A.transpose(0, 1)
        }

        if (node.attr(i: "transB") ?? 0) > 0 {
            B = B.transpose(0, 1)
        }

        let alpha = node.attr(f: "alpha") ?? 1
        let beta = node.attr(f: "beta") ?? 1

        if let C = tensors(node.input(2)) {
            return alpha * matmul(A, B) + beta * C
        } else {
            return alpha * matmul(A, B)
        }
    }
}
