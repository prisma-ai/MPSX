import MetalPerformanceShadersGraph

enum PadMode: String {
    case reflect, constant

    // MARK: Internal

    var mpsPaddingMode: MPSGraphPaddingMode {
        switch self {
        case .constant: return .constant
        case .reflect: return .reflect
        }
    }
}

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad
    func pad(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0))
        else { throw OnnxError.invalidInput(node.name) }

        return try pad(
            input: input,
            mode: PadMode(rawValue: node.attr(s: "mode")),
            pads: constants(node.input(1))?.ints ?? node.attr(ints: "pads"),
            value: constants(node.input(2))?.floats?.first ?? node.attr(f: "value")
        )
    }

    func pad(
        input: MPSGraphTensor,
        mode: PadMode?,
        pads: [Int]?,
        value: Float?
    ) throws -> MPSGraphTensor {
        guard let pads, pads.count == 8 else {
            throw OnnxError.invalidInput(#function)
        }

        let output = padTensor(
            input,
            with: (mode ?? .constant).mpsPaddingMode,
            leftPadding: pads[0 ..< 4].nsnumbers,
            rightPadding: pads[4...].nsnumbers,
            constantValue: Double(value ?? 0),
            name: nil
        )

        return output
    }
}
