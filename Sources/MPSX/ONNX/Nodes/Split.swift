import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
    func split(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> [String: MPSGraphTensor] {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        return try split(
            input: input,
            outputs: node.output,
            chunks: node.attr(ints: "split") ?? [],
            axis: node.attr(i: "axis") ?? 0
        )
    }

    func split(
        input: MPSGraphTensor,
        outputs: [String],
        chunks: [Int],
        axis: Int
    ) throws -> [String: MPSGraphTensor] {
        guard chunks.count == outputs.count else {
            throw OnnxError.invalidInput(#function)
        }

        var outputTensors: [String: MPSGraphTensor] = [:]
        outputTensors.reserveCapacity(outputs.count)

        (0 ..< outputs.count).forEach { i in
            outputTensors[outputs[i]] = sliceTensor(
                input,
                dimension: axis,
                start: chunks.prefix(upTo: i).reduce(0, +),
                length: chunks[i],
                name: nil
            )
        }

        return outputTensors
    }
}
