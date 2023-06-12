import Foundation

public final class OnnxModel {
    // MARK: Lifecycle

    public init(data: Data) throws {
        let proto = try Onnx_ModelProto(serializedData: data)

        var initializer: [String: Onnx_TensorProto] = [:]

        initializer.reserveCapacity(proto.graph.initializer.count)

        proto.graph.initializer.forEach {
            initializer[$0.name] = $0
        }

        let inputs: [Input] = try proto.graph.input.filter {
            // tensor names from initializer MAY also appear in the input list
            initializer[$0.name] == nil
        }.map {
            let tensorType = $0.type.tensorType

            guard Onnx_TensorProto.DataType(rawValue: Int(tensorType.elemType))?.matchingMPSDataType != nil else {
                throw OnnxError.unsupportedTensorDataType(onnx: tensorType.elemType, mps: nil)
            }

            return Input(
                name: $0.name,
                shape: tensorType.shape.dim.map {
                    switch $0.value {
                    case let .dimValue(value):
                        return Int(value)
                    default:
                        return -1
                    }
                }
            )
        }

        let outputs = proto.graph.output.map(\.name)

        self.proto = proto
        self.initializer = initializer
        self.inputs = inputs
        self.outputs = outputs

        optimizedForMPS = proto.producerName == "MPSX"
    }

    // MARK: Public

    public struct Input {
        public let name: String
        public let shape: [Int]
    }

    public let inputs: [Input]
    public let outputs: [String]

    // MARK: Internal

    let proto: Onnx_ModelProto
    let initializer: [String: Onnx_TensorProto]
    let optimizedForMPS: Bool
}
