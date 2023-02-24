import Accelerate
import Foundation

#if !arch(arm64)
typealias Float16 = UInt16
#endif

/// fast floating point conversion
enum FPC {
    // MARK: Internal

    static func _Float32_Float16<T: AccelerateBuffer>(_ input: T) -> [Float16] where T.Element == Float {
        convert(input, vImageConvert_PlanarFtoPlanar16F)
    }

    static func _Float16_Float32<T: AccelerateBuffer>(_ input: T) -> [Float] where T.Element == Float16 {
        convert(input, vImageConvert_Planar16FtoPlanarF)
    }

    static func _Int8_Float32<T: AccelerateBuffer>(_ input: T) -> [Float] where T.Element == Int8 {
        convert(input, vDSP.convertElements)
    }

    static func _Int16_Float32<T: AccelerateBuffer>(_ input: T) -> [Float] where T.Element == Int16 {
        convert(input, vDSP.convertElements)
    }

    static func _Int32_Float32<T: AccelerateBuffer>(_ input: T) -> [Float] where T.Element == Int32 {
        convert(input, vDSP.convertElements)
    }

    static func _UInt8_Float32<T: AccelerateBuffer>(_ input: T) -> [Float] where T.Element == UInt8 {
        convert(input, vDSP.convertElements)
    }

    static func _UInt16_Float32<T: AccelerateBuffer>(_ input: T) -> [Float] where T.Element == UInt16 {
        convert(input, vDSP.convertElements)
    }

    static func _UInt32_Float32<T: AccelerateBuffer>(_ input: T) -> [Float] where T.Element == UInt32 {
        convert(input, vDSP.convertElements)
    }

    // MARK: Private

    @_transparent
    private static func convert<U: AccelerateBuffer, V: Numeric>(_ input: U, _ body: (U, inout UnsafeMutableBufferPointer<V>) -> Void) -> [V] {
        .init(unsafeUninitializedCapacity: input.count) { buffer, initializedCount in
            body(input, &buffer)
            initializedCount = input.count
        }
    }

    private static func convert<U: AccelerateBuffer, V: Numeric>(
        _ input: U,
        _ body: (UnsafePointer<vImage_Buffer>, UnsafePointer<vImage_Buffer>, vImage_Flags) -> vImage_Error
    ) -> [V] where U.Element: Numeric {
        var output = [V](repeating: 0, count: input.count)

        @_transparent
        func buffer<T>(of _: T.Type, pointer: UnsafeMutableRawPointer, count: Int) -> vImage_Buffer {
            .init(data: pointer, height: 1, width: UInt(count), rowBytes: count * MemoryLayout<T>.stride)
        }

        input.withUnsafeBufferPointer { inputPointer in
            output.withUnsafeMutableBufferPointer { outputPointer in
                var inputBuffer = buffer(of: U.self, pointer: .init(mutating: inputPointer.baseAddress!), count: inputPointer.count)
                var outputBuffer = buffer(of: V.self, pointer: .init(mutating: outputPointer.baseAddress!), count: outputPointer.count)

                _ = body(&inputBuffer, &outputBuffer, 0)
            }
        }

        return output
    }
}
