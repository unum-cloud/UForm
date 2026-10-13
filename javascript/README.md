# UForm for JavaScript

UForm multimodal AI SDK offers a simple way to integrate multimodal AI capabilities into your JavaScript applications.
Built around ONNX, the SDK is supposed to work with most runtimes and almost any hardware.

## Installation

There are several ways to install the UForm JavaScript SDK from NPM.

```bash
pnpm add uform 
npm add uform  
yarn add uform  
```

## Quick Start

### Embeddings

```js
import { getModel, Modality, TextProcessor, TextEncoder, ImageEncoder, ImageProcessor } from '@unum-cloud/uform';

const { configPath, modalityPaths, tokenizerPath } = await getModel(
    modelId: 'unum-cloud/uform3-image-text-english-small',
    modalities: [Modality.TextEncoder, Modality.ImageEncoder] 
);

const textProcessor = new TextProcessor(configPath, tokenizerPath);
await textProcessor.init();
const processedTexts = await textProcessor.process(["a small red panda in a zoo"]);

const textEncoder = new TextEncoder(modalityPaths.text_encoder, textProcessor);
await textEncoder.init();
const textOutput = await textEncoder.encode(processedTexts);
assert(textOutput.embeddings.dims.length === 2, "Output should be 2D");
await textEncoder.dispose();

const imageProcessor = new ImageProcessor(configPath);
await imageProcessor.init();
const processedImages = await imageProcessor.process("path/to/image.png");

const imageEncoder = new ImageEncoder(modalityPaths.image_encoder, imageProcessor);
await imageEncoder.init();
const imageOutput = await imageEncoder.encode(processedImages);
assert(imageOutput.embeddings.dims.length === 2, "Output should be 2D");
```

The `textOutput` and `imageOutput` would contain `features` and `embeddings` properties, which are the same as the `features` and `embeddings` properties in the Python SDK.
The embeddings can later be compared using the cosine similarity or other distance metrics.

### Generative Models

Coming soon ...

## Technical Details

### Faster Search

Depending on the application, the embeddings can be down-casted to smaller numeric representations without losing much recall.
Independent of the quantization level, native JavaScript functionality may be too slow for large-scale search.
In such cases, consider using [USearch][github-usearch] or [SimSimD][github-simsimd].

[github-usearch]: https://github.com/unum-cloud/usearch
[github-simsimd]: https://github.com/ashvardanian/simsimd
