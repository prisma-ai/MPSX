import MetalPerformanceShadersGraph

extension MPSGraph {
    func reshapeHW(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
        if let shape = tensor.shape, shape.count == 1 {
            return reshape(tensor, shape: [shape[0], 1, 1], name: nil)
        }
        return tensor
    }
}
