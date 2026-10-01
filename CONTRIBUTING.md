# Contributing to UForm

UForm 4 is a CUDA training and inference stack with Python data tooling around it.

## CUDA

The prerequisites are listed at the top of `CMakeLists.txt`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Python

The `python_data/` tooling is tested the way CI runs it:

```sh
pip install --group test
python -X faulthandler -m pytest -x
```

`UFORM_FILTER` keeps the tests whose node id matches it, as a regex or else as a substring.
`UFORM_SEED` pins the seed, `42` by default or `random`, and pytest prints both in its header.

## Swift

To build and test the Swift package, use the following command:

```bash
swift build
swift test
```

Swift formatting is enforced with `swift-format` default utility from Apple.
To install and run it on all the files in the project, use the following command:

```bash
brew install swift-format
swift-format . -i -r
```

The style is controlled by the `.swift-format` JSON file in the root of the repository.
As there is no standard for Swift formatting, even Apple's own `swift-format` tool and Xcode differ in their formatting rules, and available settings.

## JavaScript

For rapid development you can avoid the TypeScript precompilation step:

```sh
npm install -g ts-node
ts-node javascript/embeddings.mts
```

Before submitting any changes, please make sure that the tests pass.

```sh
npm install
npm test
```
