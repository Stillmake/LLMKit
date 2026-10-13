// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "LLMKit",
    platforms: [.macOS(.v14), .iOS(.v17), .visionOS(.v1)],
    products: [
        .library(name: "LLMKit", targets: ["LLMKit"]),
    ],
    targets: [
        .target(
            name: "LLMKit",
            resources: [.copy("Resources/models.json")]
        ),
        .testTarget(
            name: "LLMKitTests",
            dependencies: ["LLMKit"]
        ),
    ]
)
