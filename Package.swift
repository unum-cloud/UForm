// swift-tools-version:6.2

import Foundation
import PackageDescription

// The C surface is consumed as headers, not as sources: `src/` pulls the NumKong, ForkUnion and StringZilla stack,
// has CUDA and HIP siblings SwiftPM cannot drive, and is built by CMake into `libuform_static` for every other binding
// already. So `UFormC` is the headers under `c/`, and the Swift target links the library CMake produced. Point the
// linker at it with `swift build -Xlinker -L<build directory>`, or install it system-wide.
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
    targets: [
        .target(
            name: "UFormC",
            path: ".",
            // Everything else under this path - the sibling checkouts, other bindings, and CMake build trees
            // holding NumKong's Metal kernels - would read as resources, and build trees take any name.
            exclude: try FileManager.default.contentsOfDirectory(atPath: Context.packageDirectory)
                .filter { !$0.hasPrefix(".") && $0 != "c" && $0 != "swift" },
            sources: ["swift/shim.c"],
            publicHeadersPath: "c",
            cSettings: [.headerSearchPath("c")]
        ),
        .target(
            name: "UForm",
            dependencies: ["UFormC"],
            path: "swift",
            exclude: ["README.md", "Test.swift", "shim.c"],
            sources: ["Types.swift", "Model.swift", "Encode.swift", "Train.swift"],
            linkerSettings: [
                .linkedLibrary("uform_static"),
                // The engine is C++, and Apple SDKs ship only libc++ - there is no libstdc++ to name.
                .linkedLibrary("c++", .when(platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS])),
                .linkedLibrary("stdc++", .when(platforms: [.linux])),
            ]
        ),
        .testTarget(
            name: "UFormTests",
            dependencies: ["UForm"],
            path: "swift",
            exclude: ["README.md", "shim.c", "Types.swift", "Model.swift", "Encode.swift", "Train.swift"],
            sources: ["Test.swift"]
        ),
    ],
    cLanguageStandard: .c11
)
