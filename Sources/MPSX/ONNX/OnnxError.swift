public enum OnnxError: Swift.Error {
    /// ONNX model has an inconsistent structure or some unsupported features
    case invalidModel(reason: String)
    /// MPSX only supports a subset of the available ONNX operators, so this error will be thrown if an operator is not supported
    case unsupportedOperator(String)
    /// ONNX is a very volatile format with a bunch of opsets available, so if the layer input is invalid, this error will be thrown
    case invalidInput(String)
    /// You will mainly use FP16/FP32, less often integers. But in the case of some exotic data types, this error will be thrown.
    case unsupportedTensorDataType(onnx: Int32?, mps: UInt32?)
}
