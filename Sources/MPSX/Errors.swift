public enum OnnxError: Swift.Error {
    case invalidModel
    case unsupportedOperator(String)
    case invalidInput(String)
    case unsupportedTensorDataType(onnx: Int32?, mps: UInt32?)
}
