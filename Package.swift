// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "UForm",
    platforms: [
        // Linux doesn't have to be explicitly listed
        .iOS(.v16),  // For iOS, version 13 and later
        .tvOS(.v16),  // For tvOS, version 13 and later
        .macOS(.v13),  // For macOS, version 10.15 (Catalina) and later
        .watchOS(.v6),  // For watchOS, version 6 and later
    ],
    products: [
        .library(
            name: "UForm",
            targets: ["UForm"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/ashvardanian/swift-transformers",
            revision: "9ef46a51eca46978b62773f8887926dfe72b0ab4"
        )
    ],
    targets: [
        .target(
            name: "UForm",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "swift",
            exclude: ["EncodersTests.swift"]
        ),
        .testTarget(
            name: "UFormTests",
            dependencies: ["UForm"],
            path: "swift",
            sources: ["EncodersTests.swift"]
        ),
    ]
)
