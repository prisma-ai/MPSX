import MetalPerformanceShadersGraph

extension MPSGraph {
    func constant(
        _ tensor: Onnx_TensorProto,
        targetDataType: MPSDataType
    ) throws -> MPSGraphTensor {
        switch (targetDataType, Onnx_TensorProto.DataType(rawValue: Int(tensor.dataType))) {
        case (.float16, .float16),
             (.float32, .float):
            return constant(tensor.rawData, shape: tensor.mpsShape, dataType: targetDataType)
        case (.float16, .int64):
            if tensor.rawData.count != 0 {
                guard let values = tensor.halfs else {
                    throw OnnxError.unsupportedTensorDataType(onnx: tensor.dataType, mps: nil)
                }
                return constant(values.rawData, shape: tensor.mpsShape, dataType: targetDataType)
            } else {
                guard !tensor.int64Data.isEmpty else {
                    throw OnnxError.unsupportedTensorDataType(onnx: tensor.dataType, mps: nil)
                }
                return constant(tensor.int64Data.rawData, shape: tensor.mpsShape, dataType: targetDataType)
            }
        case (.float32, .int64):
            if tensor.rawData.count != 0 {
                guard let values = tensor.floats else {
                    throw OnnxError.unsupportedTensorDataType(onnx: tensor.dataType, mps: nil)
                }
                return constant(values.rawData, shape: tensor.mpsShape, dataType: targetDataType)
            } else {
                guard !tensor.int64Data.isEmpty else {
                    throw OnnxError.unsupportedTensorDataType(onnx: tensor.dataType, mps: nil)
                }
                return constant(tensor.int64Data.rawData, shape: tensor.mpsShape, dataType: targetDataType)
            }
        case (.float16, _):
            guard let values = tensor.halfs, values.rawData.count != 0 else {
                throw OnnxError.unsupportedTensorDataType(onnx: tensor.dataType, mps: nil)
            }
            return constant(values.rawData, shape: tensor.mpsShape, dataType: targetDataType)
        case (.float32, _):
            guard let values = tensor.floats, values.rawData.count != 0 else {
                throw OnnxError.unsupportedTensorDataType(onnx: tensor.dataType, mps: nil)
            }
            return constant(values.rawData, shape: tensor.mpsShape, dataType: targetDataType)
        case (_, _):
            throw OnnxError.unsupportedTensorDataType(onnx: tensor.dataType, mps: targetDataType.rawValue)
        }
    }
}
