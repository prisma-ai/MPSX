import MetalPerformanceShadersGraph

extension MPSGraph {
    func reshapeHW(_ tensor: MPSGraphTensor, rank: Int) -> MPSGraphTensor {
        if let shape = tensor.shape, shape.count == 1 {
            
            return tensor.reshape([shape[0].intValue] + Array(repeating: 1, count: rank - 1))
        }
        return tensor
    }
}
