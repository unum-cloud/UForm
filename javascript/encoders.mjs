import { readFileSync } from 'fs';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { PreTrainedTokenizer } from '@xenova/transformers';
import sharp from 'sharp';

/**
 * A processor for text data that prepares input for the text encoder model.
 */
class TextProcessor {

    /**
     * Constructs a new TextProcessor instance.
     *
     * @param {string} configPath - The path to the configuration file for the text encoder.
     * @param {string} tokenizerPath - The path to the tokenizer configuration file.
     */
    constructor(configPath, tokenizerPath) {
        this.configPath = configPath;
        this.tokenizerPath = tokenizerPath;

        this.maxSeqLen = 0;
        this.padTokenIdx = 0;
        this.tokenizer = null;
    }

    /**
     * Initializes the TextProcessor by loading configurations and setting up the tokenizer.
     */
    async init() {
        var config = JSON.parse(readFileSync(this.configPath, { encoding: 'utf8' }));
        if (config.text_encoder !== undefined) {
            config = config.text_encoder;
        }

        this.maxSeqLen = config.max_position_embeddings;
        this.padTokenIdx = config.padding_idx;

        const tokenizerConfig = JSON.parse(readFileSync(this.tokenizerPath, { encoding: 'utf8' }));
        this.tokenizer = new PreTrainedTokenizer(tokenizerConfig, config);
        this.tokenizer.model_max_length = this.maxSeqLen;
        this.tokenizer.pad_token_id = this.padTokenIdx;
    }

    /**
     * Processes a list of text strings into model-ready format, including padding and attention masks.
     *
     * @param {Array<string>} texts - An array of text strings to process.
     * @return {Object} The processed texts as model input features.
     */
    async process(texts) {

        const encoded = await this.tokenizer(texts, {
            add_special_tokens: true,
            padding: 'max_length',
            max_length: this.maxSeqLen,
            truncation: true,
        });

        return {
            'input_ids': encoded.input_ids,
            'attention_mask': encoded.attention_mask,
        };
    }
}

/**
 * An encoder for text data that uses a pre-trained model to encode text.
 */
class TextEncoder {

    /**
     * Constructs a new TextEncoder instance.
     *
     * @param {string} modelPath - The path to the pre-trained ONNX model.
     */
    constructor(modelPath) {
        this.modelPath = modelPath;
        this.session = null;
    }

    /**
     * Initializes the ONNX session with the pre-trained model.
     */
    async init() {
        this.session = await InferenceSession.create(this.modelPath);
    }

    /**
     * Releases the ONNX session resources.
     */
    async dispose() {
        if (this.session) {
            await this.session.release().catch(error => console.error("Failed to release session", error));
            this.session = null;
        }
    }

    /**
     * Encodes the input data using the pre-trained model.
     *
     * @param {Object} inputs - The input data containing input_ids and attention_mask.
     * @return {Object} The encoded outputs from the model.
     */
    async encode(inputs) {
        if (!this.session) {
            throw new Error("Session is not initialized.");
        }

        // Helper function to convert BigInt64Array to Int32Array or validate Int32Array
        function ensureInt32Array(data) {
            if (data instanceof Int32Array) {
                return data; // Use as is if already Int32Array
            }
            if (data instanceof BigInt64Array) {
                // Convert BigInt64Array to Int32Array, ensuring all values are in range
                return new Int32Array(Array.from(data).map(bigInt => {
                    if (bigInt > 2147483647n || bigInt < -2147483648n) {
                        throw new Error("Value out of range for Int32.");
                    }
                    return Number(bigInt); // Convert BigInt to Number
                }));
            }
            // Additional case: handle conversion from generic Arrays or other typed arrays to Int32Array
            if (Array.isArray(data) || data instanceof Uint32Array || data instanceof Uint8Array) {
                return new Int32Array(data); // Convert directly
            }
            throw new Error("Unsupported data type for tensor conversion.");
        }

        // Prepare tensor data
        const inputIDsData = ensureInt32Array(inputs.input_ids.data);
        const attentionMaskData = ensureInt32Array(inputs.attention_mask.data);

        // Create ONNX Tensors as 'int32'
        const inputIDs = new Tensor('int32', inputIDsData, inputs.input_ids.dims);
        const attentionMask = new Tensor('int32', attentionMaskData, inputs.attention_mask.dims);

        // Run model inference
        return this.session.run({
            input_ids: inputIDs,
            attention_mask: attentionMask,
        });
    }

}

/**
 * A processor for image data that prepares images for the image encoder model.
 */
class ImageProcessor {
    constructor(configPath) {
        this.configPath = configPath;
    }

    /**
     * Initializes the ImageProcessor by loading configuration settings for image preprocessing.
     */
    async init() {
        var config = JSON.parse(readFileSync(this.configPath, 'utf8'));
        if (config.image_encoder !== undefined) {
            config = config.image_encoder;
        }

        this.imageSize = config.image_size;
        this.normalizationMeans = config.normalization_means;
        this.normalizationDeviations = config.normalization_deviations;

        this.imageMean = new Float32Array(this.normalizationMeans);
        this.imageStd = new Float32Array(this.normalizationDeviations);
    }
    /**
     * Processes raw image data into a model-ready format, including resizing, cropping, and normalizing.
     *
     * @param {Buffer|Array<Buffer>} images - A single image or an array of images to process.
     * @return {Array<Float32Array>} The processed image data as an array of Float32Arrays.
     */
    async process(images) {
        const processSingle = async (image) => {
            let img = sharp(image).toColorspace('srgb');
            const metadata = await img.metadata();
            const scale = this.imageSize / Math.min(metadata.width, metadata.height);
            const scaledWidth = Math.ceil(metadata.width * scale);
            const scaledHeight = Math.ceil(metadata.height * scale);
            img = img.resize({
                width: scaledWidth,
                height: scaledHeight,
                fit: sharp.fit.cover,
                position: sharp.strategy.entropy,
                options: sharp.interpolators.bicubic
            }).extract({
                left: Math.max(0, Math.floor((scaledWidth - this.imageSize) / 2)),
                top: Math.max(0, Math.floor((scaledHeight - this.imageSize) / 2)),
                width: this.imageSize,
                height: this.imageSize
            }).removeAlpha();

            let buffer = await img.raw().toBuffer();
            let array = new Float32Array(buffer.length);

            // When we export into the `array`, we reorder the dimensions of the tensor 
            // from HWC to CHW, and normalize the pixel values.
            let channelSize = this.imageSize * this.imageSize;
            for (let i = 0; i < this.imageSize * this.imageSize; i++) {
                let r = buffer[i * 3];
                let g = buffer[i * 3 + 1];
                let b = buffer[i * 3 + 2];
                array[i] = (r / 255.0 - this.imageMean[0]) / this.imageStd[0];
                array[channelSize + i] = (g / 255.0 - this.imageMean[1]) / this.imageStd[1];
                array[channelSize * 2 + i] = (b / 255.0 - this.imageMean[2]) / this.imageStd[2];
            }

            return array;
        };

        if (Array.isArray(images)) {
            return Promise.all(images.map(img => processSingle(img)));
        } else {
            return [await processSingle(images)];
        }
    }
}

/**
 * An encoder for image data that uses a pre-trained model to encode images.
 */
class ImageEncoder {
    constructor(modelPath, processor) {
        this.modelPath = modelPath;
        this.imageSize = processor.imageSize;
    }

    /**
     * Initializes the ONNX session with the pre-trained model.
     */
    async init() {
        this.session = await InferenceSession.create(this.modelPath);
    }

    /**
     * Releases the ONNX session resources.
     */
    async dispose() {
        if (this.session) {
            await this.session.release().catch(error => console.error("Failed to release session", error));
            this.session = null;
        }
    }

    /**
     * Encodes the processed image data using the pre-trained model.
     *
     * @param {Float32Array|Array<Float32Array>} images - The processed image data.
     * @return {Object} The encoded outputs from the model.
     */
    async encode(images) {
        if (!this.session) {
            throw new Error("Session is not initialized.");
        }

        // Helper function to ensure data is a Float32Array.
        const ensureFloat32Array = (data) => {
            if (!(data instanceof Float32Array)) {
                throw new Error("Unsupported data type for tensor conversion.");
            }
            return data;
        };

        // Helper function to concatenate multiple Float32Arrays into a single Float32Array.
        const concatFloat32Arrays = (arrays) => {
            const totalLength = arrays.reduce((acc, val) => acc + val.length, 0);
            const result = new Float32Array(totalLength);
            let offset = 0;
            for (let arr of arrays) {
                result.set(arr, offset);
                offset += arr.length;
            }
            return result;
        };

        let imagesData;
        let dims;

        if (Array.isArray(images)) {
            // Assuming each image in the array is a Float32Array representing an image already processed to a fixed size.
            const arrays = images.map(ensureFloat32Array);
            imagesData = concatFloat32Arrays(arrays);
            const numImages = arrays.length;
            const numChannels = 3;
            const height = this.imageSize;
            const width = this.imageSize;
            dims = [numImages, numChannels, height, width];
        } else {
            // Single image images, which is already a Float32Array.
            imagesData = ensureFloat32Array(images);
            const numChannels = 3;
            const height = this.imageSize;
            const width = this.imageSize;
            dims = [1, numChannels, height, width];
        }

        // Create ONNX Tensor
        const imagesTensor = new Tensor('float32', imagesData, dims);

        // Run model inference
        return this.session.run({
            images: imagesTensor,
        });
    }
}

export { TextProcessor, TextEncoder, ImageProcessor, ImageEncoder };
