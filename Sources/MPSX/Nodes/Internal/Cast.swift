import MetalPerformanceShadersGraph

extension MPSGraph {
    @available(iOS 15.0, macOS 12.0, *)
    func cast(
        _ input: MPSGraphTensor,
        to dataType: MPSDataType
    ) -> MPSGraphTensor {
        // empty name causes crash on ios 16
        cast(input, to: dataType, name: "cast_\(UUID().uuidString)")
    }
}
