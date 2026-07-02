import { existsSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';
import assert from 'assert';
import fetch from 'node-fetch';

import { getModel, Modality, TextProcessor, TextEncoder, ImageEncoder, ImageProcessor } from './index.mjs';

// Check if the HuggingFace Hub API token is set in the environment variable.
let hf_token = process.env.HUGGINGFACE_HUB_TOKEN;
if (!hf_token) {
    const dirname = path.dirname(fileURLToPath(import.meta.url));
    const tokenPath = path.join(dirname, '../', '.hf_token');
    if (existsSync(tokenPath)) {
        hf_token = readFileSync(tokenPath, 'utf8').trim();
    }
}

async function tryGettingCheckpoint(modelId, modalities) {
    const { configPath, modalityPaths, tokenizerPath } = await getModel(
        modelId,
        modalities,
        hf_token,
        '.onnx'
    );

    assert(configPath !== null, "Config path should not be null");
    assert(modalityPaths !== null, "Modality paths should not be null");
    assert(tokenizerPath !== null, "Tokenizer path should not be null");

    // Check if the file actually exists
    assert(existsSync(configPath), `Config file should exist at ${configPath}`);
    assert(existsSync(tokenizerPath), `Tokenizer file should exist at ${tokenizerPath}`);
    for (const modalityPath of Object.values(modalityPaths)) {
        assert(existsSync(modalityPath), `Modality file should exist at ${modalityPath}`);
    }
}

async function testGetCheckpoint() {
    console.log("- `testGetCheckpoint`: Start");

    try {
        const modalities = [Modality.TextEncoder, Modality.ImageEncoder];

        for (const modelId of [
            'unum-cloud/uform3-image-text-english-small',
            'unum-cloud/uform3-image-text-english-base',
            'unum-cloud/uform3-image-text-english-large',
            'unum-cloud/uform3-image-text-multilingual-base',
        ]) {
            await tryGettingCheckpoint(modelId, modalities, hf_token);
        }

        console.log("- `testGetCheckpoint`: Success");
    } catch (error) {
        console.error("- `testGetCheckpoint`: Failed", error);
    }
}

async function tryTextEncoderForwardPass(modelId) {
    const modalities = [Modality.TextEncoder];
    const { configPath, modalityPaths, tokenizerPath } = await getModel(
        modelId,
        modalities,
        hf_token,
        '.onnx'
    );

    const textProcessor = new TextProcessor(configPath, tokenizerPath);
    await textProcessor.init();
    const processedTexts = await textProcessor.process("a small red panda in a zoo");

    const textEncoder = new TextEncoder(modalityPaths.text_encoder, textProcessor);
    await textEncoder.init();
    const textOutput = await textEncoder.encode(processedTexts);
    assert(textOutput.embeddings.dims.length === 2, "Output should be 2D");

    await textEncoder.dispose();
}

async function tryImageEncoderForwardPass(modelId) {
    const modalities = [Modality.ImageEncoder];
    const { configPath, modalityPaths } = await getModel(
        modelId,
        modalities,
        hf_token,
        '.onnx'
    );

    const imageProcessor = new ImageProcessor(configPath);
    await imageProcessor.init();
    const processedImages = await imageProcessor.process("assets/unum.png");

    const imageEncoder = new ImageEncoder(modalityPaths.image_encoder, imageProcessor);
    await imageEncoder.init();
    const imageOutput = await imageEncoder.encode(processedImages);
    assert(imageOutput.embeddings.dims.length === 2, "Output should be 2D");

    await imageEncoder.dispose();
}

function cosineSimilarity(vecA, vecB) {
    // We may be receiving a complex tensor type, so let's check if it
    // has an array member named `data`.
    if (vecA.data) {
        vecA = vecA.data;
    }
    if (vecB.data) {
        vecB = vecB.data;
    }

    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * 1.0 * vecB[i];
        normA += vecA[i] * 1.0 * vecA[i];
        normB += vecB[i] * 1.0 * vecB[i];
    }
    if (normA === 0 || normB === 0) {
        return 0;
    } else {
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}

async function fetchImage(url) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    return buffer;
}

async function tryCrossReferencingImageAndText(modelId) {

    const modalities = [Modality.ImageEncoder, Modality.TextEncoder];
    const { configPath, modalityPaths, tokenizerPath } = await getModel(
        modelId,
        modalities,
        hf_token,
        '.onnx'
    );

    const imageProcessor = new ImageProcessor(configPath);
    await imageProcessor.init();
    const imageEncoder = new ImageEncoder(modalityPaths.image_encoder, imageProcessor);
    await imageEncoder.init();
    const textProcessor = new TextProcessor(configPath, tokenizerPath);
    await textProcessor.init();
    const textEncoder = new TextEncoder(modalityPaths.text_encoder, textProcessor);
    await textEncoder.init();

    const texts = [
        "A group of friends enjoy a barbecue on a sandy beach, with one person grilling over a large black grill, while the other sits nearby, laughing and enjoying the camaraderie.",
        "A white and orange cat stands on its hind legs, reaching towards a wicker basket filled with red raspberries on a wooden table in a garden, surrounded by orange flowers and a white teapot, creating a serene and whimsical scene.",
        "A little girl in a yellow dress stands in a grassy field, holding an umbrella and looking at the camera, amidst rain.",
        "This serene bedroom features a white bed with a black canopy, a gray armchair, a black dresser with a mirror, a vase with a plant, a window with white curtains, a rug, and a wooden floor, creating a tranquil and elegant atmosphere.",
        "The image captures the iconic Louvre Museum in Paris, illuminated by warm lights against a dark sky, with the iconic glass pyramid in the center, surrounded by ornate buildings and a large courtyard, showcasing the museum's grandeur and historical significance.",
    ];
    const imageUrls = [
        "https://github.com/ashvardanian/ashvardanian/blob/master/demos/bbq-on-beach.jpg?raw=true",
        "https://github.com/ashvardanian/ashvardanian/blob/master/demos/cat-in-garden.jpg?raw=true",
        "https://github.com/ashvardanian/ashvardanian/blob/master/demos/girl-and-rain.jpg?raw=true",
        "https://github.com/ashvardanian/ashvardanian/blob/master/demos/light-bedroom-furniture.jpg?raw=true",
        "https://github.com/ashvardanian/ashvardanian/blob/master/demos/louvre-at-night.jpg?raw=true",
    ];

    const textEmbeddings = [];
    const imageEmbeddings = [];

    for (let i = 0; i < texts.length; i++) {
        const text = texts[i];
        const imageUrl = imageUrls[i];
        const imageBuffer = await fetchImage(imageUrl);

        const processedText = await textProcessor.process(text);
        const processedImage = await imageProcessor.process(imageBuffer);

        const textEmbedding = await textEncoder.encode(processedText);
        const imageEmbedding = await imageEncoder.encode(processedImage);

        textEmbeddings.push(new Float32Array(textEmbedding.embeddings.data));
        imageEmbeddings.push(new Float32Array(imageEmbedding.embeddings.data));

        // Print-based debugging at its best :)
        // console.log(`Text: ${text}, Image: ${imageUrl}`);
        // console.log(`Text embedding first components: ${textEmbeddings[i].slice(0, 5)}`);
        // console.log(`Image embedding first components: ${imageEmbeddings[i].slice(0, 5)}`);
        console.log(`Similarity: ${cosineSimilarity(textEmbeddings[i], imageEmbeddings[i])}`)
    }

    for (let i = 0; i < texts.length; i++) {
        const pairSimilarity = cosineSimilarity(textEmbeddings[i], imageEmbeddings[i]);
        const otherTextSimilarities = textEmbeddings.map((te, idx) => idx === i ? -Infinity : cosineSimilarity(te, imageEmbeddings[i]));
        const otherImageSimilarities = imageEmbeddings.map((ie, idx) => idx === i ? -Infinity : cosineSimilarity(textEmbeddings[i], ie));

        const maxOtherTextSimilarity = Math.max(...otherTextSimilarities);
        const maxOtherImageSimilarity = Math.max(...otherImageSimilarities);

        assert(pairSimilarity > maxOtherTextSimilarity, "Text should be more similar to its corresponding image than to other images.");
        assert(pairSimilarity > maxOtherImageSimilarity, "Image should be more similar to its corresponding text than to other texts.");
    }

    await textEncoder.dispose();
    await imageEncoder.dispose();
}

async function testEncoders() {
    console.log("- `testEncoders`: Start");

    try {

        // Go through the bi-modal models
        for (const modelId of [
            'unum-cloud/uform3-image-text-english-small',
            // 'unum-cloud/uform3-image-text-english-base',
            // 'unum-cloud/uform3-image-text-english-large',
            // 'unum-cloud/uform3-image-text-multilingual-base',
        ]) {
            await tryTextEncoderForwardPass(modelId, hf_token);
            await tryImageEncoderForwardPass(modelId, hf_token);
            await tryCrossReferencingImageAndText(modelId, hf_token);
        }

        console.log("- `testEncoders`: Success");
    } catch (error) {
        console.error("- `testEncoders`: Failed", error);
    }
}

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
});

testGetCheckpoint();
testEncoders();
