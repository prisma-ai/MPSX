import Foundation
import MetalPerformanceShaders
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
    @usableFromInline
    var rawData: Data {
        withUnsafeBufferPointer {
            Data(buffer: $0) // copy
        }
    }
}

extension ArraySlice {
    @usableFromInline
    var rawData: Data {
        withUnsafeBufferPointer {
            Data(buffer: $0) // copy
        }
    }
}

extension Data {
    func mapMemory<T, R>(of _: T.Type, _ body: (UnsafeBufferPointer<T>) throws -> R) rethrows -> R {
        try withUnsafeBytes {
            try $0.withMemoryRebound(to: T.self, body)
        }
    }

    func array<T>(of _: T.Type) -> [T] {
        mapMemory(of: T.self) { Array($0) } // copy
    }
}

extension [Int] {
    func squeeze(axes: [Int]) -> [Int] {
        if axes.isEmpty { return filter { $0 != 1 } }
        let axes = Set(axes.map { $0 < 0 ? count + $0 : $0 })
        let shape = (0 ..< count).filter { self[$0] != 1 || !axes.contains($0) }.map { self[$0] }
        return shape
    }

    func unsqueeze(axes: [Int]) -> [Int] {
        let shapeSize = count + axes.count
        let axes = Set(axes.map { $0 < 0 ? shapeSize + $0 : $0 })
        var iterator = makeIterator()
        let shape = (0 ..< shapeSize).map { ax in
            axes.contains(ax) ? 1 : iterator.next()!
        }
        return shape
    }
}

extension Sequence where Element: BinaryInteger {
    @usableFromInline
    var nsnumbers: [NSNumber] {
        map { NSNumber(value: Int($0)) }
    }
}

extension MPSGraphResizeMode: CustomStringConvertible {
    public var description: String {
        switch self {
        case .nearest: return "nearest"
        case .bilinear: return "bilinear"
        @unknown default: return "?"
        }
    }
}

extension MPSDataType: CustomStringConvertible {
    public var description: String {
        switch self {
        case .invalid: return "invalid"
        case .floatBit: return "floatBit"
        case .float32: return "float32"
        case .float16: return "float16"
        case .complexBit: return "complexBit"
        case .complexFloat32: return "complexFloat32"
        case .complexFloat16: return "complexFloat16"
        case .signedBit: return "signedBit"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .uInt8: return "uInt8"
        case .uInt16: return "uInt16"
        case .uInt32: return "uInt32"
        case .uInt64: return "uInt64"
        case .alternateEncodingBit: return "alternateEncodingBit"
        case .bool: return "bool"
        case .normalizedBit: return "normalizedBit"
        case .unorm1: return "unorm1"
        case .unorm8: return "unorm8"
        @unknown default: return "?"
        }
    }
}

private extension MPSNDArray {
    var shape: [Int] {
        (0 ..< numberOfDimensions).map(length(ofDimension:))
    }

    func array<T: Numeric>(of _: T.Type) -> [T] {
        assert(MemoryLayout<T>.stride == dataTypeSize)

        let count = shape.reduce(1, *)

        return .init(unsafeUninitializedCapacity: count) { buffer, initializedCount in
            readBytes(buffer.baseAddress!, strideBytes: nil)
            initializedCount = count
        }
    }

    func buffer<T: Numeric>(
        of _: T.Type,
        options: MTLResourceOptions,
        commandBuffer: MTLCommandBuffer
    ) -> MTLBuffer? {
        let stride = MemoryLayout<T>.stride

        assert(stride == dataTypeSize)

        let count = shape.reduce(1, *)

        return device.makeBuffer(length: count * stride, options: options).flatMap {
            exportData(
                with: commandBuffer,
                to: $0,
                destinationDataType: dataType,
                offset: 0,
                rowStrides: nil
            )
            return $0
        }
    }
}

public extension MPSNDArray {
    var floats: [Float] {
        switch dataType {
        case .float32: return array(of: Float.self)
        case .float16: return FPC._Float16_Float32(array(of: Float16.self))

        case .int8: return FPC._Int8_Float32(array(of: Int8.self))
        case .int16: return FPC._Int16_Float32(array(of: Int16.self))
        case .int32: return FPC._Int32_Float32(array(of: Int32.self))

        case .uInt8: return FPC._UInt8_Float32(array(of: UInt8.self))
        case .uInt16: return FPC._UInt16_Float32(array(of: UInt16.self))
        case .uInt32: return FPC._UInt32_Float32(array(of: UInt32.self))

        default: assertionFailure(); return []
        }
    }

    func buffer(
        options: MTLResourceOptions,
        commandBuffer: MTLCommandBuffer
    ) -> MTLBuffer? {
        switch dataType {
        case .float32: return buffer(of: Float.self, options: options, commandBuffer: commandBuffer)
        case .float16: return buffer(of: Float16.self, options: options, commandBuffer: commandBuffer)

        case .int8: return buffer(of: Int8.self, options: options, commandBuffer: commandBuffer)
        case .int16: return buffer(of: Int16.self, options: options, commandBuffer: commandBuffer)
        case .int32: return buffer(of: Int32.self, options: options, commandBuffer: commandBuffer)

        case .uInt8: return buffer(of: UInt8.self, options: options, commandBuffer: commandBuffer)
        case .uInt16: return buffer(of: UInt16.self, options: options, commandBuffer: commandBuffer)
        case .uInt32: return buffer(of: UInt32.self, options: options, commandBuffer: commandBuffer)

        default: assertionFailure(); return nil
        }
    }
}

private extension MPSDataType {
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

public extension MPSGraphTensorData {
    static func floats(_ array: [Float], shape: [Int]? = nil, device: MTLDevice) -> MPSGraphTensorData {
        let shape = shape ?? [array.count]

        assert(shape.reduce(1, *) == array.count)

        return .init(
            device: .init(mtlDevice: device),
            data: array.rawData,
            shape: shape.nsnumbers,
            dataType: .float32
        )
    }

    func nhwc(in commandBuffer: MPSCommandBuffer) -> MPSGraphTensorData {
        TensorConverter.default.transform(self, [
            .init(name: "transpose_nhwc", action: { $0.toNHWC() }),
        ], in: commandBuffer)
    }

    func temporaryImage(in commandBuffer: MPSCommandBuffer) -> MPSTemporaryImage {
        assert(shape.count == 4)

        let shape = shape.map(\.intValue)

        let image = MPSTemporaryImage(
            commandBuffer: commandBuffer,
            imageDescriptor: .init(
                channelFormat: dataType.matchingImageChannelFormat,
                width: shape[2],
                height: shape[1],
                featureChannels: shape[3],
                numberOfImages: shape[0],
                usage: [.shaderRead, .shaderWrite]
            )
        )

        image.readCount = .max

        mpsndarray().exportData(
            with: commandBuffer,
            to: image.batchRepresentation(),
            offset: .init()
        )

        return image
    }

    func temporaryImage2D(
        pixelFormat: MTLPixelFormat? = nil,
        converter: MPSImageConversion? = nil,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSTemporaryImage {
        let source = temporaryImage(in: commandBuffer)

        guard let pixelFormat,
              pixelFormat != source.pixelFormat,
              let converter
        else {
            return source
        }

        defer { source.readCount = 0 }

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: source.width,
            height: source.height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        textureDescriptor.storageMode = .private

        let destination = MPSTemporaryImage(
            commandBuffer: commandBuffer,
            textureDescriptor: textureDescriptor
        )
        destination.readCount = .max

        converter.encode(
            commandBuffer: commandBuffer,
            sourceTexture: source.texture,
            destinationTexture: destination.texture
        )

        return destination
    }

    func texture2D(
        pixelFormat: MTLPixelFormat,
        converter: MPSImageConversion,
        in commandBuffer: MPSCommandBuffer
    ) -> MTLTexture {
        let source = temporaryImage(in: commandBuffer)
        defer { source.readCount = 0 }

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: source.width,
            height: source.height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]

        let destination = commandBuffer.device.makeTexture(descriptor: textureDescriptor)!

        converter.encode(
            commandBuffer: commandBuffer,
            sourceTexture: source.texture,
            destinationTexture: destination
        )

        return destination
    }

    func synchronizedNDArray(in commandBuffer: MTLCommandBuffer) -> MPSNDArray {
        let ndarray = mpsndarray()
        ndarray.synchronize(on: commandBuffer)
        return ndarray
    }
}

private extension MPSImage {
    func toNDArray(dataType: MPSDataType, in commandBuffer: MPSCommandBuffer) -> MPSTemporaryNDArray {
        let array = MPSTemporaryNDArray(
            commandBuffer: commandBuffer,
            descriptor: .init(
                dataType: dataType,
                shape: [numberOfImages, height, width, featureChannels].nsnumbers
            )
        )

        array.readCount = .max

        array.importData(
            with: commandBuffer,
            from: batchRepresentation(),
            offset: .init()
        )

        return array
    }
}

public extension MPSGraphTensorData {
    static func from(
        image: MPSImage,
        tensorShape: [Int],
        tensorDataType: MPSDataType,
        resizeMode: MPSGraphResizeMode = .bilinear,
        channelsFirst: Bool,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        let (c, h, w) = channelsFirst ? (tensorShape[1], tensorShape[2], tensorShape[3]) : (tensorShape[3], tensorShape[1], tensorShape[2])

        precondition(tensorShape.count == 4 && tensorShape[0] == 1 && image.featureChannels == c)

        let ndArray = image.toNDArray(dataType: tensorDataType, in: commandBuffer)

        let data = MPSGraphTensorData(ndArray)

        let dataShape = data.shape.map(\.intValue)

        var steps: [TensorConverter.Step] = []

        if dataShape[1] != h || dataShape[2] != w {
            steps.append(.init(name: "resize_\(resizeMode)_\(h)_\(w)", action: {
                $0.resize(mode: resizeMode, layout: .NHWC, height: h, width: w)
            }))
        }

        if data.dataType != tensorDataType {
            steps.append(.init(name: "cast_\(tensorDataType)", action: {
                $0.cast(to: tensorDataType)
            }))
        }

        if channelsFirst {
            steps.append(.init(name: "transpose_nchw", action: {
                $0.toNCHW()
            }))
        }

        return TensorConverter.default.transform(data, steps, in: commandBuffer)
    }
}

public extension MPSGraphTensorData {
    @inlinable
    static func NHWC(
        texture: MTLTexture,
        tensorShape: [Int],
        tensorDataType: MPSDataType,
        resizeMode: MPSGraphResizeMode = .bilinear,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        from(
            image: .init(texture: texture, featureChannels: tensorShape[3]),
            tensorShape: tensorShape,
            tensorDataType: tensorDataType,
            resizeMode: resizeMode,
            channelsFirst: false,
            in: commandBuffer
        )
    }

    @inlinable
    static func NCHW(
        texture: MTLTexture,
        tensorShape: [Int],
        tensorDataType: MPSDataType,
        resizeMode: MPSGraphResizeMode = .bilinear,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        from(
            image: .init(texture: texture, featureChannels: tensorShape[1]),
            tensorShape: tensorShape,
            tensorDataType: tensorDataType,
            resizeMode: resizeMode,
            channelsFirst: true,
            in: commandBuffer
        )
    }

    @inlinable
    static func NHWC(
        texture: MTLTexture,
        matching tensor: MPSGraphTensor,
        resizeMode: MPSGraphResizeMode = .bilinear,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        NHWC(
            texture: texture,
            tensorShape: tensor.ishape,
            tensorDataType: tensor.dataType,
            resizeMode: resizeMode,
            in: commandBuffer
        )
    }

    @inlinable
    static func NCHW(
        texture: MTLTexture,
        matching tensor: MPSGraphTensor,
        resizeMode: MPSGraphResizeMode = .bilinear,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        NCHW(
            texture: texture,
            tensorShape: tensor.ishape,
            tensorDataType: tensor.dataType,
            resizeMode: resizeMode,
            in: commandBuffer
        )
    }
}

public extension MPSGraph {
    convenience init(options: MPSGraphOptions) {
        self.init()
        self.options = options
    }
}

public extension MPSGraphTensor {
    var ishape: [Int] {
        shape?.map(\.intValue) ?? []
    }
}
