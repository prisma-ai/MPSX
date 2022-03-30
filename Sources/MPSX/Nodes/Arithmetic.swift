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
        case .add:
            return addition(a, b, name: nil)
        case .sub:
            return subtraction(a, b, name: nil)
        case .mul:
            return multiplication(a, b, name: nil)
        case .div:
            return division(a, b, name: nil)
        }
    }
}
