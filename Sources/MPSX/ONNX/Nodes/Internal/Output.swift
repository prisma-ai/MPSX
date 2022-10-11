import MetalPerformanceShadersGraph

extension MPSGraph {
    func output(
        tensor: MPSGraphTensor,
        valuesRange: SIMD2<Float>?
    ) -> MPSGraphTensor {
        if let valuesRange {
            return (tensor - valuesRange.x) / (valuesRange.y - valuesRange.x)
        }
        return tensor
    }
}
