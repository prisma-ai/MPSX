import MetalPerformanceShadersGraph

extension MPSGraph {
    func appendDimsIfNeeded(to tensor: MPSGraphTensor, count: Int) -> MPSGraphTensor {
        if let shape = tensor.shape, shape.count == 1 {
            return tensor.reshape(shape.map(\.intValue) + Array(repeating: 1, count: count))
        }
        return tensor
    }
}
