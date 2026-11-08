#!/usr/bin/env bash
# Writes the symbol bundle unum.cloud renders: each binding's own tool output, or its sources where no tool emits one.
#
#     .github/scripts/build-docs.sh        # → build/uform_docs_<version>.tar.gz
set -euo pipefail
version=$(cat VERSION)
name=uform_docs_$version out=build/uform_docs_$version
# Rustdoc's JSON is unstable, and `rust-version` needs 1.100, which reaches stable on 2026-11-12.
toolchain=${UFORM_RUST_TOOLCHAIN:-beta-2026-10-18}
rm -rf "$out" && mkdir -p "$out"/{doxygen,headers,python,rust,swift,javascript} build/docs-scratch/library

rm -rf build/xml && (cd docs && doxygen conf.dox) && cp -r build/xml/. "$out/doxygen"
git ls-files ':(glob)c/uform/*.h' | xargs cp --parents -t "$out/headers"
griffe dump uform --search python --no-inspection --docstyle google --full --output "$out/python/uform.json"
cp python/uform/lib.pyi "$out/python/"
UFORM_LIBRARY_DIR=$PWD/build/docs-scratch/library RUSTC_BOOTSTRAP=1 \
    cargo +"$toolchain" rustdoc --lib -- -Z unstable-options --output-format json
cp "${CARGO_TARGET_DIR:-target}/doc/uform.json" "$out/rust/"
# CMake writes this module map only into a Swift artifact; here it points at the checkout's headers instead.
printf 'module UFormC {\n    header "../../c/uform/uform.h"\n    export *\n}\n' > build/docs-scratch/module.modulemap
swiftc -emit-module -emit-module-path build/docs-scratch/UForm.swiftmodule -module-name UForm \
    -package-name UForm -parse-as-library -I build/docs-scratch -I c \
    -emit-symbol-graph -emit-symbol-graph-dir "$out/swift" swift/{Types,Model,Encode,Train}.swift
cp javascript/*.ts "$out/javascript/"
# griffe, rustdoc and Doxygen's body locations hold absolute paths, which would leak the build machine's directories.
sed -i "s|$PWD/||g; s|$PWD|.|g; s|$HOME|~|g" "$out/python/uform.json" "$out/rust/uform.json" "$out"/doxygen/*.xml
printf '{"project":"uform","version":"%s","commit":"%s"}\n' "$version" "$(git rev-parse HEAD)" > "$out/manifest.json"
tar -czf "build/$name.tar.gz" -C build "$name"
