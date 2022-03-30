import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
    func gemm(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let A = tensors(node.input(0)),
              let B = tensors(node.input(1))
        else { throw OnnxError.invalidInput(node.name) }

        return gemm(
            A: A,
            B: B,
            C: tensors(node.input(2)),
            alpha: node.attr(f: "alpha"),
            beta: node.attr(f: "beta"),
            transA: (node.attr(i: "transA") ?? 0) > 0,
            transB: (node.attr(i: "transB") ?? 0) > 0
        )
    }

    /// Y = alpha * A' * B' + beta * C
    func gemm(
        A: MPSGraphTensor,
        B: MPSGraphTensor,
        C: MPSGraphTensor?,
        alpha: Float?,
        beta: Float?,
        transA: Bool,
        transB: Bool
    ) -> MPSGraphTensor {
        var output = matrixMultiplication(
            primary: transA ? transposeTensor(A, dimension: 0, withDimension: 1, name: nil) : A,
            secondary: transB ? transposeTensor(B, dimension: 0, withDimension: 1, name: nil) : B,
            name: nil
        )

        if let alpha = alpha, alpha != 1 {
            output = multiplication(
                constant(Double(alpha), dataType: output.dataType),
                output,
                name: nil
            )
        }

        if var C = C {
            if let beta = beta, beta != 1 {
                C = multiplication(
                    constant(Double(beta), dataType: C.dataType),
                    C,
                    name: nil
                )
            }

            output = addition(output, C, name: nil)
        }

        return output
    }
}
