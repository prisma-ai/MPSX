import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose
    func permute(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              var perm = node.attr(ints: "perm")
        else { throw OnnxError.invalidInput(node.name) }

        // https://github.com/pytorch/pytorch/blob/af3964a8725236c78ce969b827fdeee1c5c54110/torch/tensor.py#L200

        var output = input

        for i in perm.indices {
            let p = perm[i]

            if p != i, p != -1 {
                var j = i

                while true {
                    let k = perm[j]

                    output = output.transpose(j, k)

                    perm[j] = -1

                    j = k

                    if perm[j] == i {
                        break
                    }
                }

                perm[j] = -1
            }
        }

        return output
    }
}
