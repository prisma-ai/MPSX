public struct OnnxGraphConfig: Codable {
    // MARK: Lifecycle

    public init(
        inputs: [String: Input] = [:],
        outputs: [String: Output] = [:],
        tensorsDataType: TensorsDataType = .fp16
    ) {
        self.inputs = inputs
        self.outputs = outputs
        self.tensorsDataType = tensorsDataType
    }

    // MARK: Public

    public enum TensorsDataType: Codable {
        case fp16
        case fp32
    }

    public struct Input: Codable {
        // MARK: Lifecycle

        public init(
            dims: [Int: Int]? = nil,
            valuesRange: SIMD2<Float>? = nil
        ) {
            self.dims = dims
            self.valuesRange = valuesRange
        }

        // MARK: Public

        /// dynamic shape values: dim -> value (ex.: NCHW `[2: 512, 3: 512]`)
        public var dims: [Int: Int]? = nil
        /// values range for denormalization (ex.: [0;1] -> [-1;1])
        public var valuesRange: SIMD2<Float>?
    }

    public struct Output: Codable {
        // MARK: Lifecycle

        public init(
            valuesRange: SIMD2<Float>? = nil
        ) {
            self.valuesRange = valuesRange
        }

        // MARK: Public

        /// values range for normalization (ex.: [-1;1] -> [0;1])
        public var valuesRange: SIMD2<Float>?
    }

    public var inputs: [String: Input] = [:]
    public var outputs: [String: Output] = [:]
    public var tensorsDataType: TensorsDataType = .fp16
}
