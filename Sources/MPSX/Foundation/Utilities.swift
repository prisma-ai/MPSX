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

public extension Sequence where Element: BinaryInteger {
    var nsnumbers: [NSNumber] {
        map { NSNumber(value: Int($0)) }
    }
}

private extension MPSNDArray {
    func arrayOf<T: Numeric>(_: T.Type) -> [T] {
        let stride = MemoryLayout<T>.stride

        assert(stride == dataTypeSize)

        let count = (0 ..< numberOfDimensions).reduce(1) {
            $0 * length(ofDimension: $1)
        }

        var array = [T](repeating: 0, count: count)
        readBytes(&array, strideBytes: nil)
        return array
    }

    func bufferOf<T: Numeric>(
        _: T.Type,
        options: MTLResourceOptions,
        commandBuffer: MTLCommandBuffer
    ) -> MTLBuffer? {
        let stride = MemoryLayout<T>.stride

        assert(stride == dataTypeSize)

        let count = (0 ..< numberOfDimensions).reduce(1) {
            $0 * length(ofDimension: $1)
        }

        guard let buffer = device.makeBuffer(length: count * stride, options: options) else {
            return nil
        }

        exportData(
            with: commandBuffer,
            to: buffer,
            destinationDataType: dataType,
            offset: 0,
            rowStrides: nil
        )

        return buffer
    }
}

public extension MPSNDArray {
    var floats: [Float] {
        switch dataType {
        case .float32: return arrayOf(Float.self)
        case .float16: return FPAC._Float16_Float32(arrayOf(Float16.self))

        case .int8: return FPAC._Int8_Float32(arrayOf(Int8.self))
        case .int16: return FPAC._Int16_Float32(arrayOf(Int16.self))
        case .int32: return FPAC._Int32_Float32(arrayOf(Int32.self))

        case .uInt8: return FPAC._UInt8_Float32(arrayOf(UInt8.self))
        case .uInt16: return FPAC._UInt16_Float32(arrayOf(UInt16.self))
        case .uInt32: return FPAC._UInt32_Float32(arrayOf(UInt32.self))

        default: assertionFailure(); return []
        }
    }

    func buffer(
        options: MTLResourceOptions,
        commandBuffer: MTLCommandBuffer
    ) -> MTLBuffer? {
        switch dataType {
        case .float32: return bufferOf(Float.self, options: options, commandBuffer: commandBuffer)
        case .float16: return bufferOf(Float16.self, options: options, commandBuffer: commandBuffer)

        case .int8: return bufferOf(Int8.self, options: options, commandBuffer: commandBuffer)
        case .int16: return bufferOf(Int16.self, options: options, commandBuffer: commandBuffer)
        case .int32: return bufferOf(Int32.self, options: options, commandBuffer: commandBuffer)

        case .uInt8: return bufferOf(UInt8.self, options: options, commandBuffer: commandBuffer)
        case .uInt16: return bufferOf(UInt16.self, options: options, commandBuffer: commandBuffer)
        case .uInt32: return bufferOf(UInt32.self, options: options, commandBuffer: commandBuffer)

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

public extension MPSGraphExecutable {
    var feeds: [String: MPSGraphTensor] {
        (feedTensors ?? []).reduce(into: [:]) {
            $0[$1.operation.name] = $1
        }
    }

    @inlinable
    @inline(__always)
    func encode(to commandBuffer: MPSCommandBuffer, inputs: [MPSGraphTensorData]) -> [MPSGraphTensorData] {
        autoreleasepool {
            encode(to: commandBuffer, inputs: inputs, results: nil, executionDescriptor: nil)
        }
    }

    @inlinable
    @inline(__always)
    func encode(to commandBuffer: MPSCommandBuffer, inputs: [String: MPSGraphTensorData]) -> [MPSGraphTensorData] {
        autoreleasepool {
            encode(to: commandBuffer, inputs: (feedTensors ?? []).compactMap {
                inputs[$0.operation.name]
            }, results: nil, executionDescriptor: nil)
        }
    }
}

public extension MPSGraphTensorData {
    convenience init(floats: [Float], shape: [Int], device: MTLDevice) {
        self.init(
            device: .init(mtlDevice: device),
            data: floats.rawData,
            shape: shape.nsnumbers,
            dataType: .float32
        )
    }

    func transform(in commandBuffer: MPSCommandBuffer, _ actions: [(MPSGraphTensor) -> MPSGraphTensor]) -> MPSGraphTensorData {
        actions.isEmpty ? self : MPSCompiledGraph(device: commandBuffer.device) { graph in
            [
                actions.reduce(graph.placeholder(shape: shape, dataType: dataType, name: nil)) { $1($0) },
            ]
        }.executable.encode(to: commandBuffer, inputs: [self])[0]
    }

    static func NHWC(
        texture: MTLTexture,
        tensorShape: [Int],
        tensorDataType: MPSDataType,
        resizeMode: MPSGraphResizeMode = .bilinear,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        precondition(tensorShape.count == 4 && tensorShape[0] == 1)

        let data = autoreleasepool {
            MPSGraphTensorData(MPSImage(texture: texture, featureChannels: tensorShape[3]).batchRepresentation())
        }

        let dataShape = data.shape.map(\.intValue)

        var transformers: [(MPSGraphTensor) -> MPSGraphTensor] = []

        if data.dataType != tensorDataType {
            transformers.append {
                $0.cast(to: tensorDataType)
            }
        }

        if dataShape[1] != tensorShape[1] || dataShape[2] != tensorShape[2] {
            transformers.append {
                $0.resize(mode: resizeMode, layout: .NHWC, height: tensorShape[1], width: tensorShape[2])
            }
        }

        return data.transform(in: commandBuffer, transformers)
    }

    static func NCHW(
        texture: MTLTexture,
        tensorShape: [Int],
        tensorDataType: MPSDataType,
        resizeMode: MPSGraphResizeMode = .bilinear,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        precondition(tensorShape.count == 4 && tensorShape[0] == 1)

        let data = autoreleasepool {
            MPSGraphTensorData(MPSImage(texture: texture, featureChannels: tensorShape[1]).batchRepresentation())
        }

        let dataShape = data.shape.map(\.intValue)

        var transformers: [(MPSGraphTensor) -> MPSGraphTensor] = []

        if data.dataType != tensorDataType {
            transformers.append {
                $0.cast(to: tensorDataType)
            }
        }

        if dataShape[1] != tensorShape[2] || dataShape[2] != tensorShape[3] {
            transformers.append {
                $0.resize(mode: resizeMode, layout: .NHWC, height: tensorShape[2], width: tensorShape[3])
            }
        }

        transformers.append {
            $0.transpose(2, 3).transpose(1, 2)
        }

        return data.transform(in: commandBuffer, transformers)
    }

    @inlinable
    @inline(__always)
    static func NHWC(
        texture: MTLTexture,
        tensor: MPSGraphTensor,
        resizeMode: MPSGraphResizeMode = .bilinear,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        NHWC(
            texture: texture,
            tensorShape: (tensor.shape ?? []).map(\.intValue),
            tensorDataType: tensor.dataType,
            resizeMode: resizeMode,
            in: commandBuffer
        )
    }

    @inlinable
    @inline(__always)
    static func NCHW(
        texture: MTLTexture,
        tensor: MPSGraphTensor,
        resizeMode: MPSGraphResizeMode = .bilinear,
        in commandBuffer: MPSCommandBuffer
    ) -> MPSGraphTensorData {
        NCHW(
            texture: texture,
            tensorShape: (tensor.shape ?? []).map(\.intValue),
            tensorDataType: tensor.dataType,
            resizeMode: resizeMode,
            in: commandBuffer
        )
    }

    func transposeNHWC(in commandBuffer: MPSCommandBuffer) -> MPSGraphTensorData {
        transform(in: commandBuffer, [
            { $0.transpose(1, 2).transpose(2, 3) },
        ])
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

    func texture2D(
        pixelFormat: MTLPixelFormat,
        converter: MPSImageConversion,
        in commandBuffer: MPSCommandBuffer
    ) -> MTLTexture {
        let image = temporaryImage(in: commandBuffer)
        defer { image.readCount = 0 }

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: image.width,
            height: image.height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]

        let texture = commandBuffer.device.makeTexture(descriptor: textureDescriptor)!

        converter.encode(
            commandBuffer: commandBuffer,
            sourceTexture: image.texture,
            destinationTexture: texture
        )

        return texture
    }

    func synchronizedNDArray(in commandBuffer: MTLCommandBuffer) -> MPSNDArray {
        let ndarray = mpsndarray()
        ndarray.synchronize(on: commandBuffer)
        return ndarray
    }
}
