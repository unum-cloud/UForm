// swift-tools-version:6.4

import PackageDescription

// The C surface is consumed as a library, not as sources: `src/` pulls the NumKong, ForkUnion and
// StringZilla stack, has CUDA and HIP siblings SwiftPM cannot drive, and is built by CMake into
// `libuform_static` for every other binding already. SwiftPM links what CMake produced: an
// XCFramework on Apple platforms and an artifact bundle elsewhere. A checkout points
// `UFORM_SWIFT_ARTIFACT` at the one `cmake --build --preset swift` wrote, relative to this
// directory; everyone else downloads the release's, whose checksums the release workflow writes.
let release = "https://github.com/unum-cloud/UForm/releases/download/v4.0.0-rc.1"
let appleChecksum = "0000000000000000000000000000000000000000000000000000000000000000"
let portableChecksum = "0000000000000000000000000000000000000000000000000000000000000000"

let library: [Target]
let libraryDependencies: [Target.Dependency]
if let artifact = Context.environment["UFORM_SWIFT_ARTIFACT"] {
    library = [.binaryTarget(name: "UFormC", path: artifact)]
    libraryDependencies = ["UFormC"]
}
else {
    library = [
        .binaryTarget(name: "UFormCApple", url: "\(release)/UFormC.xcframework.zip", checksum: appleChecksum),
        .binaryTarget(name: "UFormCPortable", url: "\(release)/UFormC.artifactbundle.zip", checksum: portableChecksum),
    ]
    libraryDependencies = [
        .target(
            name: "UFormCApple",
            condition: .when(platforms: [.macOS, .macCatalyst, .iOS, .tvOS, .watchOS, .visionOS])
        ),
        .target(name: "UFormCPortable", condition: .when(platforms: [.linux, .android, .windows, .wasi])),
    ]
}

let package = Package(
    name: "UForm",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .tvOS(.v18),
        .watchOS(.v11),
        .visionOS(.v2),
    ],
    products: [
        .library(
            name: "UForm",
            targets: ["UForm"]
        )
    ],
    targets: library + [
        .target(
            name: "UForm",
            dependencies: libraryDependencies,
            path: "swift",
            exclude: ["README.md", "Test.swift"],
            sources: ["Types.swift", "Model.swift", "Encode.swift", "Train.swift"],
            linkerSettings: [
                // The engine is C++, and Apple SDKs ship only libc++ - there is no
                // libstdc++ to name.
                .linkedLibrary("c++", .when(platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS])),
                .linkedLibrary("stdc++", .when(platforms: [.linux])),
            ]
        ),
        .testTarget(
            name: "UFormTests",
            dependencies: ["UForm"],
            path: "swift",
            exclude: ["README.md", "Types.swift", "Model.swift", "Encode.swift", "Train.swift"],
            sources: ["Test.swift"]
        ),
    ]
)
