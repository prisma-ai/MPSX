// swift-tools-version:5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MPSX",
    platforms: [
        .iOS(.v14),
        .macOS(.v11),
    ],
    products: [
        .library(name: "MPSX", targets: ["MPSX"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/swift-protobuf.git",
            from: "1.20.1"
        ),
    ],
    targets: [
        .target(
            name: "MPSX",
            dependencies: [
                .product(name: "SwiftProtobuf", package: "swift-protobuf"),
            ]
        ),
        .testTarget(
            name: "MPSXTests",
            dependencies: ["MPSX"]
        ),
    ]
)
