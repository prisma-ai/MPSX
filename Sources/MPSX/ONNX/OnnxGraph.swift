import MetalPerformanceShadersGraph

public extension MPSCompiledGraph {
    convenience init(
        onnxModel: OnnxModel,
        device: MTLDevice,
        options: Options = .init(),
        config: OnnxGraphConfig = .init(),
        transformOutputs: (([String: MPSGraphTensor]) -> [String: MPSGraphTensor])? = nil
    ) throws {
        try self.init(device: device, options: options) { mpsGraph in
            let onnxGraph = onnxModel.proto.graph

            let tensorsDataType = config.tensorsDataType.mpsDataType

            var constants: [String: Onnx_TensorProto] = onnxModel.initializer

            var tensors: [String: MPSGraphTensor] = try constants.mapValues {
                try mpsGraph.constant($0, targetDataType: tensorsDataType)
            }

            onnxModel.inputs.forEach { input in
                let options = config.inputs[input.name]

                let shape = input.shape.enumerated().map {
                    options?.dims?[$0.offset] ?? Int($0.element)
                }

                tensors[input.name] = mpsGraph.input(
                    shape: shape.nsnumbers,
                    dataType: tensorsDataType,
                    valuesRange: options?.valuesRange,
                    isImage: options?.isImage ?? false,
                    name: input.name
                )
            }

            for node in onnxGraph.node {
                let success = try mpsGraph.onnx(
                    node: node,
                    optimizedForMPS: onnxModel.optimizedForMPS,
                    tensorsDataType: tensorsDataType,
                    tensors: &tensors,
                    constants: &constants
                )

                if success {
                    continue
                }

                throw OnnxError.unsupportedOperator(node.opType)
            }

            var outputs: [String: MPSGraphTensor] = try onnxModel.outputs.reduce(into: [:]) {
                guard let tensor = tensors[$1] else {
                    throw OnnxError.invalidModel(reason: "tensor named \($1) not found")
                }

                let outputConfig = config.outputs[$1]

                $0[$1] = mpsGraph.output(
                    tensor: tensor,
                    valuesRange: outputConfig?.valuesRange,
                    isImage: outputConfig?.isImage ?? false
                )
            }

            if let transformOutputs {
                outputs = transformOutputs(outputs)
            }

            return outputs
        }
    }
}
