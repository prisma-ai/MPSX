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

extension Array {
    mutating func move(from i: Index, to j: Index) {
        insert(remove(at: i), at: j)
    }
}

final class MPSImageKernels {
    // MARK: Lifecycle

    private init(devices: [MTLDevice]) {
        bilinearScale = devices.reduce(into: [:]) {
            $0[$1.registryID] = MPSImageBilinearScale(device: $1)
        }

        conversion = devices.reduce(into: [:]) {
            $0[$1.registryID] = MPSImageConversion(device: $1)
        }

        bilinearScale.forEach {
            $0.value.edgeMode = .clamp
        }
    }

    // MARK: Internal

    #if os(macOS) || targetEnvironment(macCatalyst)
    static let `default` = MPSImageKernels(devices: MTLCopyAllDevices().filter {
        MPSSupportsMTLDevice($0)
    })
    #else
    static let `default` = MPSImageKernels(devices: [MTLCreateSystemDefaultDevice()!])
    #endif

    let bilinearScale: [UInt64: MPSImageBilinearScale]
    let conversion: [UInt64: MPSImageConversion]
}

extension MPSNDArray {
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

extension MTLPixelFormat {
    var featureChannels: Int {
        switch self {
        case .r8Unorm: return 1
        case .rg8Unorm: return 2
        case .rgba8Unorm: return 4
        case .bgra8Unorm: return 4
        case .r8Unorm_srgb: return 1
        case .rg8Unorm_srgb: return 2
        case .rgba8Unorm_srgb: return 4
        case .bgra8Unorm_srgb: return 4
        case .r16Unorm: return 1
        case .rg16Unorm: return 2
        case .rgba16Unorm: return 4
        case .r16Float: return 1
        case .rg16Float: return 2
        case .rgba16Float: return 4
        case .r32Float: return 1
        case .rg32Float: return 2
        case .rgba32Float: return 4
        default: assertionFailure(); return .max
        }
    }
}

private extension MPSGraphTensorData {
    func toImage(in commandBuffer: MPSCommandBuffer) -> MPSTemporaryImage {
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

    func temporaryImage(
        pixelFormat: MTLPixelFormat? = nil,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSTemporaryImage {
        let source = toImage(in: commandBuffer)

        guard let pixelFormat, pixelFormat != source.pixelFormat else {
            return source
        }

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

        MPSImageKernels.default.conversion[commandBuffer.device.registryID]!.encode(
            commandBuffer: commandBuffer,
            sourceTexture: source.texture,
            destinationTexture: destination.texture
        )

        source.readCount = 0

        return destination
    }

    func texture(
        pixelFormat: MTLPixelFormat,
        in commandBuffer: MPSCommandBuffer
    ) -> MTLTexture {
        let source = toImage(in: commandBuffer)

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: source.width,
            height: source.height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]

        let destination = commandBuffer.device.makeTexture(descriptor: textureDescriptor)!

        MPSImageKernels.default.conversion[commandBuffer.device.registryID]!.encode(
            commandBuffer: commandBuffer,
            sourceTexture: source.texture,
            destinationTexture: destination
        )

        source.readCount = 0

        return destination
    }

    func synchronizedNDArray(in commandBuffer: MTLCommandBuffer) -> MPSNDArray {
        let ndarray = mpsndarray()
        ndarray.synchronize(on: commandBuffer)
        return ndarray
    }
}

public extension MPSImage {
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

public extension MTLTexture {
    func toNDArray(
        dataType: MPSDataType,
        featureChannels: Int,
        targetWidth: Int,
        targetHeight: Int,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSTemporaryNDArray {
        assert(textureType == .type2D)

        let mpsImage = MPSImage(
            texture: self,
            featureChannels: featureChannels > 0 ? featureChannels : min(3, pixelFormat.featureChannels)
        )

        if targetHeight < 0 || targetWidth < 0 || (mpsImage.height == targetHeight && mpsImage.width == targetWidth) {
            let ndArray = mpsImage.toNDArray(dataType: dataType, in: commandBuffer)

            return ndArray
        }

        let tmpImage = MPSTemporaryImage(
            commandBuffer: commandBuffer,
            imageDescriptor: .init(
                channelFormat: mpsImage.featureChannelFormat,
                width: targetWidth,
                height: targetHeight,
                featureChannels: mpsImage.featureChannels
            )
        )

        tmpImage.readCount = .max

        MPSImageKernels.default.bilinearScale[commandBuffer.device.registryID]!.encode(
            commandBuffer: commandBuffer,
            sourceImage: mpsImage,
            destinationImage: tmpImage
        )

        let ndArray = tmpImage.toNDArray(dataType: dataType, in: commandBuffer)

        tmpImage.readCount = 0

        return ndArray
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
