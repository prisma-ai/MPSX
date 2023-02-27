import MetalPerformanceShadersGraph

extension MPSGraph {
    func reshapeHW(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
        if let shape = tensor.shape, shape.count == 1 {
            return tensor.reshape([shape[0].intValue, 1, 1])
        }
        return tensor
    }
}
