//! Builds the library through CMake and links what it produced.
//!
//! No C or C++ is compiled here directly: the `CMakeLists.txt` files own the targets, the
//! per-target compiler flags and the GPU backends, and duplicating any of that in Rust would be a
//! second source of truth that drifts. So this drives CMake into Cargo's own output directory and
//! consumes what it leaves there, never a shared build tree.
//!
//! One implementation of the C interface is linked, never two. Each names a CMake target and the
//! directory it lands in; all of them answer the same verbs, and the engine and the stand-in
//! even carry the same archive name, so choosing between them is a search path rather than a
//! different link line.
//!
//! File: build.rs
//! Author: Ash Vardanian
//! Date: October 18, 2026

use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};

/// Every CMake option an implementation can switch on, with the values switching it on and off,
/// so a reconfigured tree never keeps an earlier choice.
const KNOWN_OPTIONS: &[(&str, &str, &str)] = &[
    ("UFORM_BUILD_STUB", "ON", "OFF"),
    ("UFORM_CUDA_ARCH", "all", ""),
    ("UFORM_ROCM_ARCH", "all", ""),
];

/// One implementation of the C interface, and what CMake must be told to produce it.
#[derive(Clone, Copy)]
struct Implementation {
    /// The features that select it, named when a refused pair is reported.
    feature: &'static str,
    /// The options switched on for it; every other one this crate knows is switched off.
    options: &'static [&'static str],
    /// The target CMake is asked to build.
    target: &'static str,
    /// Where the generator leaves the archive, relative to the build tree.
    directory: &'static str,
    /// The archive's name, without the platform's prefix or extension.
    library: &'static str,
    /// The system libraries the archive leaves unresolved, outside WebAssembly.
    dependencies: &'static [&'static str],
    /// The directories whose contents rebuild it, beside the interface every one of them reads.
    sources: &'static [&'static str],
}

/// The engine for the host alone, an archive carrying its NumKong, StringZilla and yyjson objects.
const ENGINE: Implementation = Implementation {
    feature: "static",
    options: &[],
    target: "uform_static",
    directory: "c",
    library: "uform_static",
    dependencies: &[
        "turbojpeg",
        "png",
        "webp",
        "avformat",
        "avcodec",
        "avutil",
        "swresample",
        "swscale",
    ],
    sources: &["src", "cmake"],
};

/// The engine with the NVIDIA residency, the trainer and the mesh.
const ENGINE_CUDA: Implementation = Implementation {
    feature: "cuda",
    options: &["UFORM_CUDA_ARCH"],
    ..ENGINE
};

/// The engine with the AMD residency.
const ENGINE_ROCM: Implementation = Implementation {
    feature: "rocm",
    options: &["UFORM_ROCM_ARCH"],
    ..ENGINE
};

/// The engine with both GPU residencies, driven by one binary.
const ENGINE_CUDA_ROCM: Implementation = Implementation {
    feature: "cuda and rocm",
    options: &["UFORM_CUDA_ARCH", "UFORM_ROCM_ARCH"],
    ..ENGINE
};

/// The stand-in, which answers the same verbs over the C++ standard library alone.
const STUB: Implementation = Implementation {
    feature: "stub",
    options: &["UFORM_BUILD_STUB"],
    target: "uform_static_stub",
    directory: "stub",
    library: "uform_static",
    dependencies: &[],
    sources: &["src_stub"],
};

/// The tree a sibling crate reports through its `links` metadata.
fn sibling_source(links: &str) -> Result<String, String> {
    let variable = format!("DEP_{}_SOURCE", links.to_uppercase());
    env::var(&variable).map_err(|_| format!("{variable} is unset: this `{links}` crate predates reporting its tree"))
}

/// Answers the one implementation the features select, refusing the stub beside a GPU backend
/// rather than choosing.
///
/// Naming none is the host engine, which is why `static` is not a feature: Cargo unions features
/// across the whole dependency graph and never subtracts, so a named default could only be turned
/// off by every consumer at once.
fn selected() -> Result<Implementation, String> {
    match (cfg!(feature = "cuda"), cfg!(feature = "rocm"), cfg!(feature = "stub")) {
        (false, false, false) => Ok(ENGINE),
        (true, false, false) => Ok(ENGINE_CUDA),
        (false, true, false) => Ok(ENGINE_ROCM),
        (true, true, false) => Ok(ENGINE_CUDA_ROCM),
        (false, false, true) => Ok(STUB),
        (cuda, _, true) => Err(format!(
            "the stub and {} features are mutually exclusive, and both were enabled - the stub is a whole \
             implementation of the C interface carrying no GPU backend. Cargo unions features across the \
             dependency graph, so this usually means a second crate asked for the other one.",
            if cuda { ENGINE_CUDA.feature } else { ENGINE_ROCM.feature }
        )),
    }
}

/// Rebuilds when a directory's contents change, recursing because Cargo watches only what it is
/// told and a nested source would otherwise go unnoticed.
fn watch(directory: &Path) {
    println!("cargo:rerun-if-changed={}", directory.display());
    for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
        let path = entry.path();
        match path.is_dir() {
            true => watch(&path),
            false => println!("cargo:rerun-if-changed={}", path.display()),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    if env::var_os("DOCS_RS").is_some() {
        return Ok(());
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let chosen = selected()?;
    let target_os = env::var("CARGO_CFG_TARGET_OS")?;

    // A directory holding the chosen archive that CMake already built, like a parent project's
    // build tree or a release, skips the build here. Pass the archive's exact directory.
    println!("cargo:rerun-if-env-changed=UFORM_LIBRARY_DIR");
    match env::var_os("UFORM_LIBRARY_DIR") {
        Some(directory) => {
            let directory = PathBuf::from(directory);
            let archive = match env::var("CARGO_CFG_TARGET_ENV")?.as_str() {
                "msvc" => format!("{}.lib", chosen.library),
                _ => format!("lib{}.a", chosen.library),
            };
            println!("cargo:rerun-if-changed={}", directory.join(archive).display());
            println!("cargo:rustc-link-search=native={}", directory.display());
        }
        None => build_library(&manifest, chosen, &target_os)?,
    }

    println!("cargo:rustc-link-lib=static={}", chosen.library);
    if target_os != "wasi" {
        for dependency in chosen.dependencies {
            println!("cargo:rustc-link-lib=dylib={dependency}");
        }
    }

    // The archive is C++, so the standard library it was compiled against has to come along.
    match target_os.as_str() {
        "windows" => {}
        "macos" | "ios" => println!("cargo:rustc-link-lib=dylib=c++"),
        // wasi-sdk ships libc++ and libc++abi as archives under its sysroot, in the `noeh` multilib
        // the toolchain file compiles against.
        "wasi" => {
            let sysroot = PathBuf::from(env::var("WASI_SDK_PATH")?).join("share/wasi-sysroot/lib/wasm32-wasip1/noeh");
            println!("cargo:rustc-link-search=native={}", sysroot.display());
            println!("cargo:rustc-link-lib=static=c++");
            println!("cargo:rustc-link-lib=static=c++abi");
        }
        _ => println!("cargo:rustc-link-lib=dylib=stdc++"),
    }

    // The whole tree, so a dependent's CMake adds the checkout Cargo resolved as a subdirectory.
    println!("cargo:source={}", manifest.display());
    Ok(())
}

/// Builds the chosen archive through CMake into Cargo's output directory and adds where it landed
/// to the search path.
fn build_library(manifest: &Path, chosen: Implementation, target_os: &str) -> Result<(), Box<dyn Error>> {
    let mut library = cmake::Config::new(manifest);
    // The crate reconfigures on every invocation by default, which on this tree is a fresh probe of
    // every dependency before a build that has nothing to do.
    // CMake's Debug configuration instruments for sanitizers whose runtimes rustc never links.
    library
        .profile("RelWithDebInfo")
        .build_target(chosen.target)
        .always_configure(false);
    for (option, on, off) in KNOWN_OPTIONS {
        library.define(option, if chosen.options.contains(option) { on } else { off });
    }

    // `FetchContent` builds each sibling crate's tree rather than downloading the
    // pinned revision again.
    for (variable, links) in [
        ("FETCHCONTENT_SOURCE_DIR_FORKUNION", "forkunion"),
        ("FETCHCONTENT_SOURCE_DIR_NUMKONG", "numkong"),
        ("FETCHCONTENT_SOURCE_DIR_SMASHTABLE", "smashtable"),
        ("FETCHCONTENT_SOURCE_DIR_STRINGZILLA", "stringzilla"),
    ] {
        library.define(variable, sibling_source(links)?);
    }

    // Every preset compiles with Clang, so an unset `CXX` takes `clang++` rather than
    // the platform default.
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CXX");
    let compiler = env::var("CXX").unwrap_or_else(|_| String::from("clang++"));
    if env::var_os("CXX").is_none() {
        library
            .define("CMAKE_C_COMPILER", "clang")
            .define("CMAKE_CXX_COMPILER", &compiler);
    }
    if chosen.options.contains(&"UFORM_CUDA_ARCH") {
        library.define("CMAKE_CUDA_HOST_COMPILER", &compiler);
    }

    // `cmake-rs` has no WASI system name to hand CMake, so the checked-in toolchain names the SDK
    // and the sysroot; the one thing a user sets is `WASI_SDK_PATH`.
    if target_os == "wasi" {
        println!("cargo:rerun-if-env-changed=WASI_SDK_PATH");
        println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
        library.define(
            "CMAKE_TOOLCHAIN_FILE",
            manifest.join("cmake/toolchain-wasm32-wasi.cmake"),
        );
        // An engine validates a module whole, so the SIMD tier is the one the Rust target declares;
        // the flags `cmake-rs` passes replace the toolchain's initial ones, so the tier and
        // `-fno-exceptions` ride with them.
        let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
        if !target_features.split(',').any(|declared| declared == "simd128") {
            return Err("UForm on WebAssembly needs `-C target-feature=+simd128`".into());
        }
        library
            .cflag("-msimd128")
            .cxxflag("-msimd128")
            .cxxflag("-fno-exceptions");
    }

    // `build_target` stops short of installing, so the archive sits where the generator left it:
    // directly under the target's own directory, or one config deeper on the multi-config ones.
    let build = library.build().join("build").join(chosen.directory);
    println!("cargo:rustc-link-search=native={}", build.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build.join("RelWithDebInfo").display()
    );

    // Only what the chosen implementation actually compiles, so a stand-in build ignores the
    // engine's sources and an engine build ignores the stand-in's.
    println!("cargo:rerun-if-changed={}", manifest.join("CMakeLists.txt").display());
    watch(&manifest.join("c"));
    for source in chosen.sources {
        watch(&manifest.join(source));
    }
    Ok(())
}
