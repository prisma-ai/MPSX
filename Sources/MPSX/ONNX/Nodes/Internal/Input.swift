import MetalPerformanceShadersGraph

extension MPSGraph {
    func input(
        shape: [NSNumber]?,
        dataType: MPSDataType,
        valuesRange: SIMD2<Float>?,
        name: String?
    ) -> MPSGraphTensor {
        let inputTensor = placeholder(
            shape: shape,
            dataType: dataType,
            name: name
        )

        if let valuesRange {
            return (valuesRange.y - valuesRange.x) * inputTensor + valuesRange.x
        }

        return inputTensor
    }
}
