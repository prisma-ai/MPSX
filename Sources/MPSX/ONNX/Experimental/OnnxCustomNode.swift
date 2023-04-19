import MetalPerformanceShadersGraph

public protocol OnnxAttributesProvider {
    func attr(s name: String) -> String
    func attr(f name: String) -> Float?
    func attr(floats name: String) -> [Float]?
    func attr(i name: String) -> Int?
    func attr(ints name: String) -> [Int]?
}

public protocol OnnxCustomNode {
    func preprocess(
        inputTensor: MPSGraphTensor,
        inputName: String,
        graph: MPSGraph
    ) -> MPSGraphTensor

    func postprocess(
        outputName: String,
        outputShape: [Int],
        requiredDataType: MPSDataType,
        graph: MPSGraph
    ) -> (placeholder: MPSGraphTensor, tensor: MPSGraphTensor)

    func eval(
        inputs: [MPSGraphTensorData],
        outputShapes: [[Int]],
        attributesProvider: OnnxAttributesProvider,
        in commandBuffer: MPSCommandBuffer
    ) throws -> [MPSGraphTensorData]
}
