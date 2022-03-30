import MetalPerformanceShadersGraph

extension MPSGraph {
    func output(
        tensor: MPSGraphTensor,
        valuesRange: SIMD2<Float>?
    ) -> MPSGraphTensor {
        var outputTensor = tensor

        if let valuesRange = valuesRange {
            let dataType = outputTensor.dataType

            outputTensor = division(
                subtraction(
                    outputTensor,
                    constant(.init(valuesRange.x), dataType: dataType),
                    name: nil
                ),
                constant(.init(valuesRange.y - valuesRange.x), dataType: dataType),
                name: nil
            )
        }

        return outputTensor
    }
}
