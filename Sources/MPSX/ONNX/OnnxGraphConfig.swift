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
            valuesRange: SIMD2<Float>? = nil,
            isImage: Bool = false
        ) {
            self.dims = dims
            self.valuesRange = valuesRange
            self.isImage = isImage
        }

        // MARK: Public

        /// Dynamic shape values: dim -> value (ex.: NCHW `[2: 512, 3: 512]`)
        public var dims: [Int: Int]? = nil
        /// Values range for denormalization (ex.: [0;1] -> [-1;1])
        public var valuesRange: SIMD2<Float>?

        /// Performs nhwc -> nchw transposition
        public var isImage: Bool
    }

    public struct Output: Codable {
        // MARK: Lifecycle

        public init(
            valuesRange: SIMD2<Float>? = nil,
            isImage: Bool = false
        ) {
            self.valuesRange = valuesRange
            self.isImage = isImage
        }

        // MARK: Public

        /// Values range for normalization (ex.: [-1;1] -> [0;1])
        public var valuesRange: SIMD2<Float>?

        /// Performs nchw -> nhwc transposition
        public var isImage: Bool
    }

    public var inputs: [String: Input] = [:]
    public var outputs: [String: Output] = [:]

    /// All tensors (constants, external and internal inputs/outputs) will have this data type
    public var tensorsDataType: TensorsDataType = .fp16
}
