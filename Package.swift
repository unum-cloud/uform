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
            revision: "89fb5d97e1df347f9f588f62fc538dcad6fdb16c"
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
