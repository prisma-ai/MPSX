import MetalPerformanceShadersGraph

extension MPSGraph {
    func input(
        shape: [NSNumber],
        dataType: MPSDataType,
        valuesRange: SIMD2<Float>?,
        isImage: Bool,
        name: String?
    ) -> MPSGraphTensor {
        var shape = shape

        if isImage {
            shape.move(from: 1, to: 3)
        }

        var input = placeholder(
            shape: shape,
            dataType: dataType,
            name: name
        )

        if isImage {
            input = input.toNCHW()
        }

        if let valuesRange {
            input = (valuesRange.y - valuesRange.x) * input + valuesRange.x
        }

        return input
    }
}
