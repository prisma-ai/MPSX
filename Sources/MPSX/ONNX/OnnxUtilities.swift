import Foundation
import MetalPerformanceShadersGraph

extension OnnxGraphConfig.TensorsDataType {
    var mpsDataType: MPSDataType {
        switch self {
        case .fp16: return .float16
        case .fp32: return .float32
        }
    }
}

extension Onnx_TensorProto.DataType {
    var matchingMPSDataType: MPSDataType? {
        switch self {
        case .float: return .float32
        case .float16: return .float16
        case .uint8: return .uInt8
        case .uint16: return .uInt16
        case .uint32: return .uInt32
        case .uint64: if #available(iOS 14.1, *) { return .uInt64 }; return nil
        case .int8: return .int8
        case .int16: return .int16
        case .int32: return .int32
        case .int64: if #available(iOS 14.1, *) { return .int64 }; return nil
        default: return nil
        }
    }
}

extension Onnx_NodeProto {
    func attr(_ name: String) -> Onnx_AttributeProto? {
        attribute.first { $0.name == name }
    }

    func attr(s name: String) -> String {
        attr(name).flatMap { String(data: $0.s, encoding: .utf8) } ?? ""
    }

    func attr(f name: String) -> Float? {
        attr(name)?.f
    }

    func attr(floats name: String) -> [Float]? {
        attr(name)?.floats
    }

    func attr(i name: String) -> Int? {
        attr(name).flatMap { Int($0.i) }
    }

    func attr(ints name: String) -> [Int]? {
        attr(name).flatMap { $0.ints.map { Int($0) } }
    }
}

extension Data {
    func signedIntegers(assumingIntegerDataType dataType: Onnx_TensorProto.DataType) -> [Int]? {
        switch dataType {
        case .int8: return mapMemory(of: Int8.self) { $0.map { Int($0) } }
        case .int16: return mapMemory(of: Int16.self) { $0.map { Int($0) } }
        case .int32: return mapMemory(of: Int32.self) { $0.map { Int($0) } }
        case .int64: return array(of: Int.self)
        default: return nil
        }
    }

    func unsignedIntegers(assumingIntegerDataType dataType: Onnx_TensorProto.DataType) -> [UInt]? {
        switch dataType {
        case .uint8: return mapMemory(of: UInt8.self) { $0.map { UInt($0) } }
        case .uint16: return mapMemory(of: UInt16.self) { $0.map { UInt($0) } }
        case .uint32: return mapMemory(of: UInt32.self) { $0.map { UInt($0) } }
        case .uint64: return array(of: UInt.self)
        default: return nil
        }
    }

    func floats(assumingIntegerDataType dataType: Onnx_TensorProto.DataType) -> [Float]? {
        switch dataType {
        case .int8: return mapMemory(of: Int8.self, FPC._Int8_Float32)
        case .int16: return mapMemory(of: Int16.self, FPC._Int16_Float32)
        case .int32: return mapMemory(of: Int32.self, FPC._Int32_Float32)
        case .int64: return mapMemory(of: Int64.self) { $0.map { Float($0) } }
        case .uint8: return mapMemory(of: UInt8.self, FPC._UInt8_Float32)
        case .uint16: return mapMemory(of: UInt16.self, FPC._UInt16_Float32)
        case .uint32: return mapMemory(of: UInt32.self, FPC._UInt32_Float32)
        case .uint64: return mapMemory(of: UInt64.self) { $0.map { Float($0) } }
        default: return nil
        }
    }

    func floats16(assumingDataType dataType: Onnx_TensorProto.DataType) -> [Float16]? {
        switch dataType {
        case .float16: return array(of: Float16.self)
        case .float: return mapMemory(of: Float.self, FPC._Float32_Float16)
        default: return floats(assumingIntegerDataType: dataType).flatMap { FPC._Float32_Float16($0) }
        }
    }

    func floats32(assumingDataType dataType: Onnx_TensorProto.DataType) -> [Float]? {
        switch dataType {
        case .float16: return mapMemory(of: Float16.self, FPC._Float16_Float32)
        case .float: return array(of: Float.self)
        default: return floats(assumingIntegerDataType: dataType)
        }
    }
}

extension Onnx_TensorProto {
    var mpsShape: [NSNumber] {
        dims.isEmpty ? [1] : dims.nsnumbers
    }

    var _dataType: Onnx_TensorProto.DataType? {
        .init(rawValue: Int(dataType))
    }

    var floats: [Float]? {
        _dataType.flatMap(rawData.floats32(assumingDataType:))
    }

    var halfs: [Float16]? {
        _dataType.flatMap(rawData.floats16(assumingDataType:))
    }

    var ints: [Int]? {
        _dataType.flatMap(rawData.signedIntegers(assumingIntegerDataType:))
    }

    var uints: [UInt]? {
        _dataType.flatMap(rawData.unsignedIntegers(assumingIntegerDataType:))
    }
}
