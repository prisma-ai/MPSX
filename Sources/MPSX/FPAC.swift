import Accelerate
import Foundation

/// Floating Point Accelerate Conversion
enum FPAC {
    // MARK: Internal

    static func _Float32_Float16(_ input: [Float]) -> [Float16] {
        let count = input.count
        var input = input
        var output = [Float16](repeating: 0, count: count)
        _PlanarFtoPlanar16F(&input, &output, count)
        return output
    }

    static func _Float16_Float32(_ input: [Float16]) -> [Float] {
        let count = input.count
        var input = input
        var output = [Float](repeating: 0, count: count)
        _Planar16FtoPlanarF(&input, &output, count)
        return output
    }

    static func _Int8_Float32(_ input: [Int8]) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        vDSP.convertElements(of: input, to: &output)
        return output
    }

    static func _Int16_Float32(_ input: [Int16]) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        vDSP.convertElements(of: input, to: &output)
        return output
    }

    static func _Int32_Float32(_ input: [Int32]) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        vDSP.convertElements(of: input, to: &output)
        return output
    }

    static func _UInt8_Float32(_ input: [UInt8]) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        vDSP.convertElements(of: input, to: &output)
        return output
    }

    static func _UInt16_Float32(_ input: [UInt16]) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        vDSP.convertElements(of: input, to: &output)
        return output
    }

    static func _UInt32_Float32(_ input: [UInt32]) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        vDSP.convertElements(of: input, to: &output)
        return output
    }

    // MARK: Private

    @discardableResult
    private static func _PlanarFtoPlanar16F(
        _ input: UnsafeMutableRawPointer,
        _ output: UnsafeMutableRawPointer,
        _ count: Int
    ) -> Bool {
        var inputBuffer = vImage_Buffer(
            data: input,
            height: 1,
            width: UInt(count),
            rowBytes: count * 4
        )
        var outputBuffer = vImage_Buffer(
            data: output,
            height: 1,
            width: UInt(count),
            rowBytes: count * 2
        )
        return vImageConvert_PlanarFtoPlanar16F(&inputBuffer, &outputBuffer, 0) == kvImageNoError
    }

    @discardableResult
    private static func _Planar16FtoPlanarF(
        _ input: UnsafeMutableRawPointer,
        _ output: UnsafeMutableRawPointer,
        _ count: Int
    ) -> Bool {
        var inputBuffer = vImage_Buffer(
            data: input,
            height: 1,
            width: UInt(count),
            rowBytes: count * 2
        )
        var outputBuffer = vImage_Buffer(
            data: output,
            height: 1,
            width: UInt(count),
            rowBytes: count * 4
        )
        return vImageConvert_Planar16FtoPlanarF(&inputBuffer, &outputBuffer, 0) == kvImageNoError
    }
}
