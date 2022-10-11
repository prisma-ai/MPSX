import MetalPerformanceShadersGraph

enum ArithmeticOp {
    case add, sub, mul, div
}

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div
    func arithmetic(
        op: ArithmeticOp,
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let a = tensors(node.input(0)),
              let b = tensors(node.input(1))
        else { throw OnnxError.invalidInput(node.name) }

        switch op {
        case .add: return a + b
        case .sub: return a - b
        case .mul: return a * b
        case .div: return a / b
        }
    }
}
