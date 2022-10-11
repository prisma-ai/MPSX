import Accelerate
import CoreGraphics
import Metal
import MetalKit
import MetalPerformanceShaders

extension MTLCommandQueue {
    @discardableResult
    func sync<T>(_ body: (MPSCommandBuffer) throws -> T) rethrows -> T {
        let commandBuffer = MPSCommandBuffer(from: self)

        let result = try autoreleasepool {
            try body(commandBuffer)
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return result
    }
}

extension CGImage {
    func jpeg(compressionQuality: Float = 1.0) -> Data? {
        guard
            let mutableData = CFDataCreateMutable(kCFAllocatorDefault, 0),
            let destination = CGImageDestinationCreateWithData(mutableData, "public.jpeg" as CFString, 1, nil)
        else { return nil }

        let properties: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: compressionQuality as CFNumber,
        ]

        CGImageDestinationAddImage(destination, self, properties as CFDictionary)

        guard CGImageDestinationFinalize(destination) else {
            return nil
        }

        return mutableData as Data
    }
}

extension MTLTexture {
    func cgImage(colorSpace: CGColorSpace) -> CGImage? {
        assert(pixelFormat == .rgba8Unorm)

        let rowBytes = width * 4
        let length = rowBytes * height

        let bytes = UnsafeMutableRawPointer.allocate(
            byteCount: length,
            alignment: MemoryLayout<UInt8>.alignment
        )
        defer { bytes.deallocate() }

        getBytes(
            bytes,
            bytesPerRow: rowBytes,
            from: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0
        )

        guard
            let data = CFDataCreate(
                nil,
                bytes.assumingMemoryBound(to: UInt8.self),
                length
            ),
            let dataProvider = CGDataProvider(data: data),
            let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: rowBytes,
                space: colorSpace,
                bitmapInfo: .init(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: dataProvider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
            )
        else { return nil }

        return cgImage
    }
}

final class GPU {
    // MARK: Lifecycle

    init(commandQueue: MTLCommandQueue) {
        self.commandQueue = commandQueue
        textureLoader = .init(device: commandQueue.device)
        imageScaler = MPSImageBilinearScale(device: commandQueue.device)
        imageScaler.edgeMode = .clamp
        imageConverter = .init(device: commandQueue.device)
    }

    // MARK: Internal

    static let `default` = GPU(commandQueue: MTLCreateSystemDefaultDevice()!.makeCommandQueue()!)

    let commandQueue: MTLCommandQueue
    let textureLoader: MTKTextureLoader
    let imageScaler: MPSImageScale
    let imageConverter: MPSImageConversion

    var device: MTLDevice {
        commandQueue.device
    }
}
