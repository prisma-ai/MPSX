import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose
    func permute(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let perm = node.attr(ints: "perm")
        else { throw OnnxError.invalidInput(node.name) }

        let transpositions: [(Int, Int)]

        // pixel shuffle permutation (TODO: general algorithm)
        if perm == [0, 1, 4, 2, 5, 3] {
            transpositions = [(2, 4), (3, 4), (4, 5)]
        } else {
            throw OnnxError.unsupportedOperator(node.name)
        }

        var output = input

        transpositions.forEach { dim1, dim2 in
            output = transposeTensor(
                output,
                dimension: dim2,
                withDimension: dim1,
                name: nil
            )
        }

        return output
    }
}
