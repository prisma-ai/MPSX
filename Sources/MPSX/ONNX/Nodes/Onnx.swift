import MetalPerformanceShadersGraph

extension MPSGraph {
    /// entry point for onnx -> mps mapping
    func onnx(
        node: Onnx_NodeProto,
        optimizedForMPS: Bool,
        tensorsDataType: MPSDataType,
        tensors: inout [String: MPSGraphTensor],
        constants: inout [String: Onnx_TensorProto]
    ) throws -> Bool {
        let output: MPSGraphTensor

        switch node.opType {
        case "Add":
            output = try arithmetic(op: .add, node, tensors)
        case "Sub":
            output = try arithmetic(op: .sub, node, tensors)
        case "Mul":
            output = try arithmetic(op: .mul, node, tensors)
        case "Div":
            output = try arithmetic(op: .div, node, tensors)
        case "Sqrt":
            output = try sqrt(node, tensors)
        case "Exp":
            output = try exp(node, tensors)
        case "Log":
            output = try log(node, tensors)
        case "Floor":
            output = try floor(node, tensors)
        case "Less":
            output = try less(node, tensors)
        case "Greater":
            output = try greater(node, tensors)
        case "Where":
            output = try whereOp(node, tensors)
        case "BatchNormalization":
            output = try batchNorm(node, tensors)
        case "InstanceNormalization":
            output = try instanceNorm(node, tensors)
        case "custom_group_norm": // onnx does not support group norm out of the box
            output = try groupNorm(node, tensors, constants)
        case "Concat":
            output = try concat(node, tensors)
        case "Conv":
            output = try conv(node, tensors, swizzled: optimizedForMPS)
        case "FusedConv":
            output = try fusedConv(node, tensors, swizzled: optimizedForMPS)
        case "ConvTranspose":
            output = try convTranspose(node, tensors)
        case "Gemm",
             "MatMul":
            output = try gemm(node, tensors)
        case "GlobalAveragePool":
            output = try globalPool(.avg, node, tensors)
        case "AveragePool":
            output = try pool(.avg, node, tensors)
        case "MaxPool":
            output = try pool(.max, node, tensors)
        case "Pad":
            output = try pad(node, tensors, constants)
        case "Reshape":
            output = try reshape(node, tensors, constants)
        case "Squeeze":
            output = try squeeze(node, tensors, constants)
        case "Unsqueeze":
            output = try unsqueeze(node, tensors, constants)
        case "Shape":
            output = try shape(node, tensors)
        case "Relu":
            output = try relu(node, tensors)
        case "PRelu", "LeakyRelu":
            output = try prelu(node, tensors, constants)
        case "Elu":
            output = try elu(node, tensors)
        case "Sigmoid":
            output = try sigmoid(node, tensors)
        case "HardSigmoid":
            output = try hardSigmoid(node, tensors)
        case "Upsample",
             "Resize":
            output = try resize(node, tensors, constants)
        case "Tanh":
            output = try tanh(node, tensors)
        case "Softmax":
            output = try softmax(node, tensors)
        case "Flatten":
            output = try flatten(node, tensors)
        case "Transpose":
            output = try permute(node, tensors)
        case "Slice":
            output = try slice(node, tensors, constants)
        case "ReduceMean":
            output = try reduceMean(node, tensors, constants)
        case "ReduceSum":
            output = try reduceSum(node, tensors, constants)
        case "ReduceL2":
            output = try reduceL2(node, tensors, constants)
        case "Dropout":
            output = try dropout(node, tensors, constants)
        case "DepthToSpace":
            output = try depthToSpace(node, tensors)
        case "Constant":
            guard let value = node.attr("value") else {
                throw OnnxError.invalidInput(node.name)
            }
            node.output.forEach {
                constants[$0] = value.t
            }
            output = try constant(value.t, targetDataType: tensorsDataType)
        case "Cast":
            output = try passthrough(node, tensors)
        case "Clip":
            output = try clip(node, tensors)
        case "Pow":
            output = try pow(node, tensors)
        case "Tile":
            output = try tile(node, tensors, constants)
        case "Gather":
            output = try gather(node, tensors, constants)
        case "GatherElements":
            output = try gatherElements(node, tensors, constants)
        case "Expand":
            output = try expand(node, tensors, constants)
        case "Neg":
            output = try neg(node, tensors)
        case "Split":
            try split(node, tensors).forEach {
                tensors[$0.0] = $0.1
            }
            return true
        default:
            return false
        }

        node.output.forEach {
            tensors[$0] = output
        }

        return true
    }
}
