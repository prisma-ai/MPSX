import MetalPerformanceShadersGraph

extension MPSGraph {
    func output(
        tensor: MPSGraphTensor,
        valuesRange: SIMD2<Float>?,
        isImage: Bool
    ) -> MPSGraphTensor {
        var output = tensor

        if let valuesRange {
            output = (output - valuesRange.x) / (valuesRange.y - valuesRange.x)
        }

        if isImage {
            output = output.toNHWC()
        }

        return output
    }
}
