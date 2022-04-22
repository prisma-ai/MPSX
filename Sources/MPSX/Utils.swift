import Accelerate
import Foundation
import MetalPerformanceShadersGraph

typealias Pair<T> = (T, T)
typealias Quad<T> = (T, T, T, T)

extension Array {
    var pair: Pair<Element>? {
        count == 2 ? (self[0], self[1]) : nil
    }

    var quad: Quad<Element>? {
        count == 4 ? (self[0], self[1], self[2], self[3]) : nil
    }
}

extension MPSGraphTensor {
    var quadShape: Quad<Int>? {
        shape?.map(\.intValue).quad
    }
}

extension Dictionary {
    func callAsFunction(_ key: Key?) -> Value? {
        key.flatMap { self[$0] }
    }
}

extension Array {
    func callAsFunction(_ i: Index) -> Element? {
        indices.contains(i) ? self[i] : nil
    }
}

extension Array {
    var rawData: Data {
        withUnsafeBufferPointer {
            Data(buffer: $0)
        }
    }
}

extension Data {
    func arrayOf<T>(_: T.Type) -> [T] {
        let n = count / MemoryLayout<T>.stride

        return withUnsafeBytes {
            $0.baseAddress
                .flatMap { $0.bindMemory(to: T.self, capacity: n) }
                .flatMap { Array(UnsafeBufferPointer<T>(start: $0, count: n)) }
        } ?? []
    }
}

extension MPSDataType {
    var matchingImageChannelFormat: MPSImageFeatureChannelFormat {
        switch self {
        case .float16: return .float16
        case .float32: return .float32
        case .uInt8: return .unorm8
        case .uInt16: return .unorm16
        default: assertionFailure(); return .float32
        }
    }
}

extension MPSImageFeatureChannelFormat {
    var matchingDataType: MPSDataType {
        switch self {
        case .float16: return .float16
        case .float32: return .float32
        case .unorm8: return .uInt8
        case .unorm16: return .uInt16
        default: assertionFailure(); return .float32
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
        case .int8: return arrayOf(Int8.self).map { Int($0) }
        case .int16: return arrayOf(Int16.self).map { Int($0) }
        case .int32: return arrayOf(Int32.self).map { Int($0) }
        case .int64: return arrayOf(Int.self)
        default: return nil
        }
    }

    func unsignedIntegers(assumingIntegerDataType dataType: Onnx_TensorProto.DataType) -> [UInt]? {
        switch dataType {
        case .uint8: return arrayOf(UInt8.self).map { UInt($0) }
        case .uint16: return arrayOf(UInt16.self).map { UInt($0) }
        case .uint32: return arrayOf(UInt32.self).map { UInt($0) }
        case .uint64: return arrayOf(UInt.self)
        default: return nil
        }
    }

    func floats(assumingNonFloatDataType dataType: Onnx_TensorProto.DataType) -> [Float]? {
        switch dataType {
        case .int8: return FPAC._Int8_Float32(arrayOf(Int8.self))
        case .int16: return FPAC._Int16_Float32(arrayOf(Int16.self))
        case .int32: return FPAC._Int32_Float32(arrayOf(Int32.self))
        case .int64: return arrayOf(Int64.self).map { Float($0) }
        case .uint8: return FPAC._UInt8_Float32(arrayOf(UInt8.self))
        case .uint16: return FPAC._UInt16_Float32(arrayOf(UInt16.self))
        case .uint32: return FPAC._UInt32_Float32(arrayOf(UInt32.self))
        case .uint64: return arrayOf(UInt64.self).map { Float($0) }
        default: return nil
        }
    }

    func floats16(assumingDataType dataType: Onnx_TensorProto.DataType) -> [Float16]? {
        switch dataType {
        case .float16: return arrayOf(Float16.self)
        case .float: return FPAC._Float32_Float16(arrayOf(Float.self))
        default: return floats(assumingNonFloatDataType: dataType).flatMap(FPAC._Float32_Float16(_:))
        }
    }

    func floats32(assumingDataType dataType: Onnx_TensorProto.DataType) -> [Float]? {
        switch dataType {
        case .float16: return FPAC._Float16_Float32(arrayOf(Float16.self))
        case .float: return arrayOf(Float.self)
        default: return floats(assumingNonFloatDataType: dataType)
        }
    }
}

extension Onnx_TensorProto {
    var mpsShape: [NSNumber] {
        dims.isEmpty ? [1] : dims.map { NSNumber(value: $0) }
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
