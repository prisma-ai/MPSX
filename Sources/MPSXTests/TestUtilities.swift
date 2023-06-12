import Accelerate
import CoreGraphics
import Metal
import MetalKit
import MetalPerformanceShaders
import MPSX

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
    func cgImage(colorSpace: CGColorSpace? = nil) -> CGImage? {
        switch pixelFormat {
        case .r8Unorm:
            let rowBytes = width
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
                    bitsPerPixel: 8,
                    bytesPerRow: rowBytes,
                    space: colorSpace ?? CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: .init(rawValue: CGImageAlphaInfo.none.rawValue),
                    provider: dataProvider,
                    decode: nil,
                    shouldInterpolate: true,
                    intent: .defaultIntent
                )
            else { return nil }

            return cgImage
        case .rgba8Unorm,
             .rgba8Unorm_srgb:
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
                    space: colorSpace ?? CGColorSpace(name: CGColorSpace.displayP3)!,
                    bitmapInfo: .init(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                    provider: dataProvider,
                    decode: nil,
                    shouldInterpolate: false,
                    intent: .defaultIntent
                )
            else { return nil }

            return cgImage
        default: return nil
        }
    }
}

final class GPU {
    // MARK: Lifecycle

    init(commandQueue: MTLCommandQueue) {
        self.commandQueue = commandQueue
        textureLoader = .init(device: commandQueue.device)
    }

    // MARK: Internal

    static let `default` = GPU(commandQueue: MTLCreateSystemDefaultDevice()!.makeCommandQueue()!)

    let commandQueue: MTLCommandQueue
    let textureLoader: MTKTextureLoader

    var device: MTLDevice {
        commandQueue.device
    }
}

func compare(texture: MTLTexture, with reference: MTLTexture, treshold: Float = 1e-3) -> Bool {
    let graph = MPSCompiledGraph(
        device: GPU.default.device,
        options: .init(runtimeTypeInference: true)
    ) { graph in
        let x = graph.imagePlaceholder(
            dataType: .float32,
            height: reference.height,
            width: reference.width,
            channels: -1,
            name: "X"
        )

        let y = graph.imagePlaceholder(
            dataType: .float32,
            height: reference.height,
            width: reference.width,
            channels: -1,
            name: "Y"
        )

        let z = (x.mean(axes: [3]) - y.mean(axes: [3])).pow(2).mean(axes: [1, 2])

        return ["Z": z]
    }

    let result = GPU.default.commandQueue.sync {
        graph([
            "X": .texture(texture),
            "Y": .texture(reference),
        ], in: $0).synchronizedNDArray(in: $0)
    }.floats

    return result[0] < treshold
}

let testResourcesPath = "TestResources"

func data(bundlePath: String) throws -> Data {
    try Data(contentsOf: Bundle.module.url(forResource: bundlePath, withExtension: nil)!)
}

func texture(bundlePath: String) async throws -> MTLTexture {
    try await GPU.default.textureLoader.newTexture(
        URL: .init(
            fileURLWithPath: Bundle.module.path(forResource: bundlePath, ofType: nil)!
        ),
        options: [.SRGB: false]
    )
}

func data(arg: Int) throws -> Data {
    try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[arg]))
}

func inputTexture(arg: Int) async throws -> MTLTexture {
    try await GPU.default.textureLoader.newTexture(
        URL: .init(
            fileURLWithPath: CommandLine.arguments[arg]
        ),
        options: [.SRGB: false]
    )
}

func save(texture: MTLTexture, arg: Int) throws {
    let image = texture.cgImage()

    try image?.jpeg(compressionQuality: 0.85)?.write(
        to: .init(
            fileURLWithPath: CommandLine.arguments[arg]
        ),
        options: .atomic
    )
}
