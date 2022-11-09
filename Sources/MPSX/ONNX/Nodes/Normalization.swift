import MetalPerformanceShadersGraph

extension MPSGraph {
    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization
    func batchNorm(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let gamma = tensors(node.input(1)),
              let beta = tensors(node.input(2)),
              let mean = tensors(node.input(3)),
              let variance = tensors(node.input(4))
        else { throw OnnxError.invalidInput(node.name) }

        let output = normalize(
            input,
            mean: reshapeHW(mean),
            variance: reshapeHW(variance),
            gamma: reshapeHW(gamma),
            beta: reshapeHW(beta),
            epsilon: node.attr(f: "epsilon") ?? 1e-05,
            name: nil
        )

        return output
    }

    /// https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization
    func instanceNorm(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let gamma = tensors(node.input(1)),
              let beta = tensors(node.input(2))
        else { throw OnnxError.invalidInput(node.name) }

        let axes: [NSNumber] = [2, 3]

        let mean = mean(of: input, axes: axes, name: nil)
        let variance = variance(of: input, mean: mean, axes: axes, name: nil)

        let output = normalize(
            input,
            mean: mean,
            variance: variance,
            gamma: reshapeHW(gamma),
            beta: reshapeHW(beta),
            epsilon: node.attr(f: "epsilon") ?? 1e-05,
            name: nil
        )

        return output
    }
}

extension MPSGraph {
    func groupNorm(
        _ node: Onnx_NodeProto,
        _ tensors: [String: MPSGraphTensor],
        _ constants: [String: Onnx_TensorProto]
    ) throws -> MPSGraphTensor {
        guard let input = tensors(node.input(0)),
              let gamma = tensors(node.input(2)),
              let beta = tensors(node.input(3)),
              let groups = constants(node.input(1))?.floats?.first
        else { throw OnnxError.invalidInput(node.name) }

        let epsilon = constants(node.input(4))?.floats?.first

        return try groupNorm(
            input: input,
            gamma: gamma,
            beta: beta,
            groups: Int(groups),
            epsilon: epsilon
        )
    }

    /// https://github.com/ppwwyyxx/GroupNorm-reproduce/blob/883f694b5589e520bec6221451de159f9ec67c2d/ImageNet-ResNet-TensorFlow/resnet_model.py#L17
    /// https://github.com/google-research/big_transfer/blob/140de6e704fd8d61f3e5ea20ffde130b7d5fd065/bit_tf2/normalization.py#L20
    func groupNorm(
        input: MPSGraphTensor,
        gamma: MPSGraphTensor,
        beta: MPSGraphTensor,
        groups: Int,
        epsilon: Float?
    ) throws -> MPSGraphTensor {
        guard let origShape = input.quadShape else {
            throw OnnxError.invalidInput(#function)
        }

        let (g, c, h, w) = (
            groups,
            origShape.1,
            origShape.2,
            origShape.3
        )

        guard c % g == 0 else {
            throw OnnxError.invalidInput(#function)
        }

        let s = c / g

        let x = input.reshape([-1, g, s, h, w].nsnumbers)

        let axes: [NSNumber] = [2, 3, 4]

        let mean = mean(of: x, axes: axes, name: nil)
        let variance = variance(of: x, mean: mean, axes: axes, name: nil)

        let newShape = [1, g, s, 1, 1].nsnumbers

        let output = normalize(
            x,
            mean: mean,
            variance: variance,
            gamma: reshape(gamma, shape: newShape, name: nil),
            beta: reshape(beta, shape: newShape, name: nil),
            epsilon: epsilon ?? 1e-05,
            name: nil
        )

        return output.reshape(input.shape!)
    }
}
