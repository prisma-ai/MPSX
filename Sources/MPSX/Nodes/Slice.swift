import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice
    func slice(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard node.input.count == 3,
              let input = tensors(node.input(0)),
              let starts = constants(node.input(1))?.ints,
              let ends = constants(node.input(2))?.ints,
              starts.count == ends.count
        else { throw OnnxError.invalidInput(node.name) }

        let output = sliceTensor(
            input,
            starts: starts.map { NSNumber(value: $0) },
            ends: ends.map { NSNumber(value: $0) },
            strides: .init(repeating: NSNumber(value: 1), count: starts.count),
            name: nil
        )

        return output
    }
}
