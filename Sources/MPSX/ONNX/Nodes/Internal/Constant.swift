import MetalPerformanceShadersGraph

extension MPSGraph {
    func constant(
        _ tensor: Onnx_TensorProto,
        targetDataType: MPSDataType
    ) throws -> MPSGraphTensor {
        let data: Data?

        switch targetDataType {
        case .float16:
            data = tensor.halfs?.rawData
        case .float32:
            data = tensor.floats?.rawData
        default:
            throw OnnxError.unsupportedTensorDataType(onnx: nil, mps: targetDataType.rawValue)
        }

        guard let data else {
            throw OnnxError.unsupportedTensorDataType(onnx: tensor.dataType, mps: nil)
        }

        return constant(
            data,
            shape: tensor.mpsShape,
            dataType: targetDataType
        )
    }
}
