import MetalPerformanceShadersGraph

extension MPSGraph {
    func input(
        shape: [NSNumber]?,
        dataType: MPSDataType,
        valuesRange: SIMD2<Float>?,
        name: String?
    ) -> MPSGraphTensor {
        var inputTensor = placeholder(
            shape: shape,
            dataType: dataType,
            name: name
        )

        if let valuesRange = valuesRange {
            inputTensor = addition(
                multiplication(
                    constant(.init(valuesRange.y - valuesRange.x), dataType: dataType),
                    inputTensor,
                    name: nil
                ),
                constant(.init(valuesRange.x), dataType: dataType),
                name: nil
            )
        }

        return inputTensor
    }
}
