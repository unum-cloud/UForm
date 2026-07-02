# UForm Swift SDK

UForm offers first-party support for Swift.
To get started, add UForm to your project using Swift Package Manager.

```bash
swift package init --type executable
swift package add uform
```

Then, import UForm in your Swift code:

```swift
import UForm
```

## Embeddings

### Text Embeddings

```swift
let textModel = try await TextEncoder(
    modelName: "unum-cloud/uform3-image-text-english-small",
    computeUnits: .cpuAndNeuralEngine
)
let text = "A group of friends enjoy a barbecue on a sandy beach, with one person grilling over a large black grill, while the other sits nearby, laughing and enjoying the camaraderie."
let textEmbedding: Embedding = try textModel.encode(text)
let textVector: [Float32] = textEmbedding.asFloats()
```

### Image Embeddings

```swift
let imageModel = try await ImageEncoder(
    modelName: "unum-cloud/uform3-image-text-english-small",
    computeUnits: .cpuAndNeuralEngine
)
let imageURL = "https://github.com/ashvardanian/ashvardanian/blob/master/demos/bbq-on-beach.jpg?raw=true"
guard let url = URL(string: imageURL),
    let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
    let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) {
    throw Exception("Could not load image from URL: \(imageURL)")
}

var imageEmbedding: Embedding = try imageModel.encode(cgImage)
var imageVector: [Float32] = embedding.asFloats()
```

### Choosing Target Device

Apple chips provide several functional units capable of high-throughput matrix multiplication and AI inference.
Those `computeUnits` include the CPU, GPU, and Neural Engine.
For maximum compatibility, the `.all` option is used by default.
Sadly, Apple's scheduler is not always optimal, and it might be beneficial to specify the target device explicitly, especially if the models are pre-compiled for the Apple Neural Engine, as it may yield significant performance gains.

| Model               | GPU Text E. | ANE Text E. | GPU Image E. | ANE Image E. |
| :------------------ | ----------: | ----------: | -----------: | -----------: |
| `english-small`     |     2.53 ms |     0.53 ms |      6.57 ms |      1.23 ms |
| `english-base`      |     2.54 ms |     0.61 ms |     18.90 ms |      3.79 ms |
| `english-large`     |     2.30 ms |     0.61 ms |     79.68 ms |     20.94 ms |
| `multilingual-base` |     2.34 ms |     0.50 ms |     18.98 ms |      3.77 ms |

> On Apple M4 iPad, running iOS 18.2.
> Batch size is 1, and the model is pre-loaded into memory.
> The original encoders use `f32` single-precision numbers for maximum compatibility, and mostly rely on __GPU__ for computation.
> The quantized encoders use a mixture of `i8`, `f16`, and `f32` numbers for maximum performance, and mostly rely on the Apple Neural Engine (__ANE__) for computation.
> The median latency is reported.

### Computing Distances

There are several ways to compute distances between embeddings, once you have them.
Naive Swift code might look like this:

```swift
func cosineSimilarity(_ a: [Float32], _ b: [Float32]) -> Float32 {
    let dotProduct = zip(a, b).map(*).reduce(0, +)
    let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
    let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))
    return dotProduct / (normA * normB)
}
```

A faster way to compute distances is to use the Accelerate framework:

```swift
import Accelerate

func cosineSimilarity(_ a: [Float32], _ b: [Float32]) -> Float32 {
    var result: Float32 = 0
    var aNorm: Float32 = 0
    var bNorm: Float32 = 0
    vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
    vDSP_svesq(a, 1, &aNorm, vDSP_Length(a.count))
    vDSP_svesq(b, 1, &bNorm, vDSP_Length(b.count))
    return result / sqrt(aNorm * bNorm)
}
```

An even faster approach would be to use USearch or SimSIMD, that work not only for `Float32` and `Float64`, but also for `Float16`, `Int8`, and binary embeddings.
