import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice
    func slice(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let shape = input.shape,
              let starts = constants(node.input(1))?.ints,
              let ends = constants(node.input(2))?.ints
        else { throw OnnxError.invalidInput(node.name) }

        let steps = constants(node.input(4))?.ints

        var shapedStart = [Int]()
        var shapedEnds = [Int]()
        var shapedStrides = [Int]()
        if let axes = constants(node.input(3))?.ints {
            // manually filling missed dimensions
            for (dim, count) in shape.enumerated() {
                if let index = axes.firstIndex(where: { $0 == dim }) {
                    shapedStart.append(starts[index])
                    shapedEnds.append(Swift.min(ends[index], count.intValue))
                    shapedStrides.append(steps?[index] ?? 1)
                } else {
                    shapedStart.append(0)
                    shapedEnds.append(count.intValue)
                    shapedStrides.append(1)
                }
            }
        } else {
            shapedStart = starts
            shapedEnds = ends
            shapedStrides = steps ?? .init(repeating: 1, count: starts.count)
        }

        let output = sliceTensor(
            input,
            starts: shapedStart.nsnumbers,
            ends: shapedEnds.nsnumbers,
            strides: shapedStrides.nsnumbers,
            name: nil
        )

        return output
    }
}
