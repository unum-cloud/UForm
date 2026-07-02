//
//  Embeddings.swift
//
//
//  Created by Ash Vardanian on 3/27/24.
//
import Accelerate
import CoreGraphics
import CoreML
import Foundation
import Hub  // `Config`
import Tokenizers  // `AutoTokenizer`

/// Defines custom errors related to the encoder's functionality.
enum EncoderError: Error {
    case downloadError(String)
    case loadingError(String)
    case invalidInput(String)
    case modelPredictionFailed(String)
    case unknownError(String)
}

/// Represents different types of embeddings as arrays of different numeric types.
public enum Embedding {
    case i32s([Int32])
    case f16s([Float16])
    case f32s([Float32])
    case f64s([Float64])

    /// Initializes an embedding from a `MLMultiArray`.
    /// - Parameter multiArray: The MLMultiArray to convert into an Embedding.
    /// - Returns: nil if the data type is unsupported.
    init?(from multiArray: MLMultiArray) {
        switch multiArray.dataType {
        case .float64, .double:
            self = .f64s(
                Array(
                    UnsafeBufferPointer(
                        start: multiArray.dataPointer.assumingMemoryBound(to: Float64.self),
                        count: Int(truncating: multiArray.shape[1])
                    )
                )
            )
        case .float32, .float:
            self = .f32s(
                Array(
                    UnsafeBufferPointer(
                        start: multiArray.dataPointer.assumingMemoryBound(to: Float32.self),
                        count: Int(truncating: multiArray.shape[1])
                    )
                )
            )
        case .float16:
            self = .f16s(
                Array(
                    UnsafeBufferPointer(
                        start: multiArray.dataPointer.assumingMemoryBound(to: Float16.self),
                        count: Int(truncating: multiArray.shape[1])
                    )
                )
            )
        case .int32:
            self = .i32s(
                Array(
                    UnsafeBufferPointer(
                        start: multiArray.dataPointer.assumingMemoryBound(to: Int32.self),
                        count: Int(truncating: multiArray.shape[1])
                    )
                )
            )
        @unknown default:
            return nil
        }
    }

    /// Converts the embedding to an array of `Float`.
    public func asFloats() -> [Float] {
        switch self {
        case .f32s(let array): return array
        case .i32s(let array): return array.map(Float.init)
        case .f16s(let array): return array.map(Float.init)
        case .f64s(let array): return array.map(Float.init)
        }
    }
}

/// Provides methods for reading and handling configurations and models.
/// - Parameter path: The file path where the configuration file is located.
/// - Returns: A dictionary containing the configuration data.
func readConfig(fromPath path: String) throws -> [String: Any] {
    let absPath = path.hasPrefix("/") ? path : FileManager.default.currentDirectoryPath + "/" + path
    let data = try Data(contentsOf: URL(fileURLWithPath: absPath))
    return try JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]
}

/// Compiles and loads a machine learning model from a URL.
/// - Parameters:
///   - modelURL: The URL where the model package is located.
///   - computeUnits: The hardware devices to use for model computation.
/// - Returns: An instance of `MLModel`.
func readModel(fromURL modelURL: URL, computeUnits: MLComputeUnits = .all) throws -> MLModel {
    let compiledModelURL =
        modelURL.absoluteString.hasSuffix(".mlmodelc/") ? modelURL : try MLModel.compileModel(at: modelURL)
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits
    return try MLModel(contentsOf: compiledModelURL, configuration: config)
}

/// Loads a machine learning model from a local file path.
/// - Parameters:
///   - path: The file path where the model file is located.
///   - computeUnits: The hardware devices to use for model computation.
/// - Returns: An instance of `MLModel`.
func readModel(fromPath path: String, computeUnits: MLComputeUnits = .all) throws -> MLModel {
    let absPath = path.hasPrefix("/") ? path : FileManager.default.currentDirectoryPath + "/" + path
    let modelURL = URL(fileURLWithPath: absPath, isDirectory: true)
    return try readModel(fromURL: modelURL, computeUnits: computeUnits)
}

/// Encodes text input into embeddings using a machine learning model.
public class TextEncoder {
    let model: MLModel
    let processor: TextProcessor

    /// Initializes a `TextEncoder` using paths for the model and configuration.
    /// - Parameters:
    ///   - modelPath: The path to the directory containing the machine learning model.
    ///   - configPath: Optional. The path to the configuration file. Defaults to config.json in the model directory.
    ///   - tokenizerPath: Optional. The path to the tokenizer file. Defaults to tokenizer.json in the model directory.
    ///   - computeUnits: The hardware devices to use for model computation. Use `.cpuAndNeuralEngine` for best performance.
    public init(
        modelPath: String,
        configPath: String? = nil,
        tokenizerPath: String? = nil,
        computeUnits: MLComputeUnits = .all
    ) throws {
        let finalConfigPath = configPath ?? modelPath + "/config.json"
        let finalTokenizerPath = tokenizerPath ?? modelPath + "/tokenizer.json"
        self.model = try readModel(fromPath: modelPath, computeUnits: computeUnits)
        self.processor = try TextProcessor(
            configPath: finalConfigPath,
            tokenizerPath: finalTokenizerPath,
            model: self.model
        )
    }

    /// Initializes a `TextEncoder` using a model name and an API for fetching models.
    /// - Parameters:
    ///   - modelName: The identifier for the model repository.
    ///   - hubApi: The API object to interact with the model hub. Defaults to a shared instance.
    ///   - computeUnits: The hardware devices to use for model computation. Use `.cpuAndNeuralEngine` for best performance.
    public init(modelName: String, hubApi: HubApi = .shared, computeUnits: MLComputeUnits = .all) async throws {
        let repo = Hub.Repo(id: modelName)
        let encoderMask =
            computeUnits == .cpuAndNeuralEngine ? "text_encoder_neural.mlpackage" : "text_encoder.mlpackage"
        let modelURL = try await hubApi.snapshot(
            from: repo,
            matching: ["\(encoderMask)/*", "config.json", "tokenizer.json"]
        )
        let configPath = modelURL.appendingPathComponent("config.json").path
        let tokenizerPath = modelURL.appendingPathComponent("tokenizer.json").path
        self.model = try readModel(
            fromURL: modelURL.appendingPathComponent(encoderMask, isDirectory: true),
            computeUnits: computeUnits
        )
        self.processor = try TextProcessor(configPath: configPath, tokenizerPath: tokenizerPath, model: self.model)
    }

    /// Processes text and returns embeddings. Throws an error if processing fails.
    /// - Parameter text: The text input to encode.
    /// - Returns: An `Embedding` object containing the model output.
    public func encode(_ text: String) throws -> Embedding {
        let inputFeatureProvider = try self.processor.preprocess(text)
        guard let prediction = try? self.model.prediction(from: inputFeatureProvider),
            let predictionFeature = prediction.featureValue(for: "embeddings"),
            let output = predictionFeature.multiArrayValue,
            let embedding = Embedding(from: output)
        else {
            throw EncoderError.modelPredictionFailed("Failed to extract embeddings or unsupported data type.")
        }
        return embedding
    }
}

/// Encodes image input into embeddings using a machine learning model.
public class ImageEncoder {
    let model: MLModel
    let processor: ImageProcessor

    /// Initializes an `ImageEncoder` using a path for the model and optionally a configuration file.
    /// - Parameters:
    ///   - modelPath: The path to the directory containing the machine learning model.
    ///   - configPath: Optional. The path to the configuration file. Defaults to config.json in the model directory.
    public init(modelPath: String, configPath: String? = nil, computeUnits: MLComputeUnits = .all) throws {
        let finalConfigPath = configPath ?? modelPath + "/config.json"
        self.model = try readModel(fromPath: modelPath, computeUnits: computeUnits)
        self.processor = try ImageProcessor(configPath: finalConfigPath)
    }

    /// Initializes an `ImageEncoder` using a model name and an API for fetching models.
    /// - Parameters:
    ///   - modelName: The identifier for the model repository.
    ///   - hubApi: The API object to interact with the model hub. Defaults to a shared instance.
    ///   - computeUnits: The hardware devices to use for model computation. Use `.cpuAndNeuralEngine` for best performance.
    public init(modelName: String, hubApi: HubApi = .shared, computeUnits: MLComputeUnits = .all) async throws {
        let repo = Hub.Repo(id: modelName)
        let encoderMask =
            computeUnits == .cpuAndNeuralEngine ? "image_encoder_neural.mlpackage" : "image_encoder.mlpackage"
        let modelURL = try await hubApi.snapshot(from: repo, matching: ["\(encoderMask)/*", "config.json"])
        let configPath = modelURL.appendingPathComponent("config.json").path
        self.model = try readModel(
            fromURL: modelURL.appendingPathComponent(encoderMask, isDirectory: true),
            computeUnits: computeUnits
        )
        self.processor = try ImageProcessor(configPath: configPath)
    }

    /// Processes an image and returns embeddings. Throws an error if processing fails.
    /// - Parameter image: The `CGImage` to encode.
    /// - Returns: An `Embedding` object containing the model output.
    public func encode(_ image: CGImage) throws -> Embedding {
        let inputFeatureProvider = try self.processor.preprocess(image)
        guard let prediction = try? self.model.prediction(from: inputFeatureProvider),
            let predictionFeature = prediction.featureValue(for: "embeddings"),
            let output = predictionFeature.multiArrayValue,
            let embedding = Embedding(from: output)
        else {
            throw EncoderError.modelPredictionFailed("Failed to extract embeddings or unsupported data type.")
        }
        return embedding
    }
}

// MARK: - Processors

/// Handles the preprocessing of text data to be used by a machine learning model.
class TextProcessor {
    let tokenizer: Tokenizer
    let minContextLength: Int
    let maxContextLength: Int

    /// Initializes a `TextProcessor` with specific configuration.
    /// - Parameters:
    ///   - configPath: The path to the configuration file specifying tokenizer and model configurations.
    ///   - tokenizerPath: The path to the tokenizer configuration.
    ///   - model: The machine learning model to be used with this processor.
    /// - Throws: An error if the configuration is invalid or missing necessary components.
    public init(configPath: String, tokenizerPath: String, model: MLModel) throws {
        var configDict = try readConfig(fromPath: configPath)
        let tokenizerDict = try readConfig(fromPath: tokenizerPath)

        // Check if there's a specific 'text_encoder' configuration within the main configuration
        if let textEncoderConfig = configDict["text_encoder"] as? [String: Any] {
            configDict = textEncoderConfig  // Use the specific 'text_encoder' configuration
        }

        // Initialize the tokenizer with its configuration.
        let config = Config(configDict)
        let tokenizerData = Config(tokenizerDict)
        self.tokenizer = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: tokenizerData)

        // Extract the model's input shape constraints.
        guard let inputDescription = model.modelDescription.inputDescriptionsByName["input_ids"],
            let multiArrayConstraint = inputDescription.multiArrayConstraint
        else {
            throw EncoderError.invalidInput("Cannot obtain shape information from the model.")
        }

        // Determine the context length constraints based on the model's input shape constraint.
        let shapeConstraint = multiArrayConstraint.shapeConstraint
        switch shapeConstraint.type {
        case .enumerated:
            minContextLength = shapeConstraint.enumeratedShapes[0][1].intValue
            maxContextLength = minContextLength
        case .range:
            guard let range = shapeConstraint.sizeRangeForDimension[1] as? NSRange else {
                throw EncoderError.unknownError("Model input shape has a range constraint that cannot be interpreted.")
            }
            minContextLength = range.location
            maxContextLength = range.length
        case .unspecified:
            throw EncoderError.unknownError("Model input shape is unspecified.")
        @unknown default:
            throw EncoderError.unknownError("Unknown model input shape constraint type.")
        }
    }

    /// Preprocesses a string of text into a format suitable for model prediction.
    /// - Parameter text: The text to preprocess.
    /// - Returns: A `MLFeatureProvider` containing the processed text ready for the model.
    /// - Throws: An error if the text encoding fails.
    public func preprocess(_ text: String) throws -> MLFeatureProvider {
        let inputIDs = self.tokenizer.encode(text: text)
        return TextInput(inputIDs: inputIDs, sequenceLength: self.maxContextLength)
    }
}

/// Handles the preprocessing of image data to be used by a machine learning model.
class ImageProcessor {
    let imageSize: Int
    let mean: [Float]
    let std: [Float]

    /// Initializes an `ImageProcessor` with specific configuration.
    /// - Parameter configPath: The path to the configuration file specifying image size, mean, and std.
    init(configPath: String) throws {
        var configDict = try readConfig(fromPath: configPath)
        if let imageEncoderConfig = configDict["image_encoder"] as? [String: Any] {
            configDict = imageEncoderConfig
        }

        let config = Config(configDict)
        guard let imageSize = config.imageSize?.value as? Int else {
            throw EncoderError.invalidInput("Invalid or missing image size.")
        }
        self.imageSize = imageSize

        guard let meanArray = config.normalizationMeans?.value as? [Any],
            let stdArray = config.normalizationDeviations?.value as? [Any]
        else {
            throw EncoderError.invalidInput("Normalization means or deviations are missing.")
        }

        self.mean = try meanArray.compactMap({
            guard let doubleValue = $0 as? Double else {
                throw EncoderError.invalidInput("Normalization means should be an array of floats.")
            }
            return Float(doubleValue)
        })

        self.std = try stdArray.compactMap({
            guard let doubleValue = $0 as? Double else {
                throw EncoderError.invalidInput("Normalization deviations should be an array of floats.")
            }
            return Float(doubleValue)
        })

        // Check if the arrays have 3 values for the 3 channels
        if self.mean.count != 3 || self.std.count != 3 {
            throw EncoderError.invalidInput("Normalization means should contain 3 values.")
        }
    }

    /// Preprocesses a `CGImage` into a format suitable for model prediction.
    /// - Parameter cgImage: The image to preprocess.
    /// - Returns: An `MLFeatureProvider` containing the preprocessed image data.
    func preprocess(_ cgImage: CGImage) throws -> MLFeatureProvider {
        guard let cropped = resizeAndCrop(image: cgImage, toSideLength: self.imageSize),
            let normalized = exportToTensorAndNormalize(image: cropped, mean: self.mean, std: self.std)
        else {
            throw EncoderError.invalidInput("Image preprocessing failed.")
        }
        let featureValue = MLFeatureValue(multiArray: normalized)
        return try ImageInput(precomputedFeature: featureValue)
    }

    private func resizeAndCrop(image: CGImage, toSideLength imageSize: Int) -> CGImage? {
        let originalWidth = CGFloat(image.width)
        let originalHeight = CGFloat(image.height)

        let widthRatio = CGFloat(imageSize) / originalWidth
        let heightRatio = CGFloat(imageSize) / originalHeight
        let scaleFactor = max(widthRatio, heightRatio)

        let scaledWidth = originalWidth * scaleFactor
        let scaledHeight = originalHeight * scaleFactor

        let dx = (scaledWidth - CGFloat(imageSize)) / 2.0
        let dy = (scaledHeight - CGFloat(imageSize)) / 2.0
        guard
            let context = CGContext(
                data: nil,
                width: imageSize,
                height: imageSize,
                bitsPerComponent: image.bitsPerComponent,
                bytesPerRow: 0,
                space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: image.bitmapInfo.rawValue
            )
        else { return nil }

        // Draw the scaled and cropped image in the context
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: -dx, y: -dy, width: scaledWidth, height: scaledHeight))
        return context.makeImage()
    }

    private func exportToTensorAndNormalize(image: CGImage, mean: [Float], std: [Float]) -> MLMultiArray? {
        let width = image.width
        let height = image.height

        // Prepare the bitmap context for drawing the image.
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard
            let context = CGContext(
                data: &pixelData,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: 4 * width,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        else { return nil }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // While normalizing the pixels, let's also transpose them from HWC to CHW
        let channelSize = width * height
        var floatPixels = [Float](repeating: 0, count: channelSize * 3)
        for i in 0 ..< channelSize {
            floatPixels[channelSize * 0 + i] = (Float(pixelData[i * 4 + 0]) / 255.0 - mean[0]) / std[0]
            floatPixels[channelSize * 1 + i] = (Float(pixelData[i * 4 + 1]) / 255.0 - mean[1]) / std[1]
            floatPixels[channelSize * 2 + i] = (Float(pixelData[i * 4 + 2]) / 255.0 - mean[2]) / std[2]
        }

        // We need to wrap the constructor that may fail
        do {
            let tensor = try MLMultiArray(
                shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
                dataType: .float32
            )
            for i in 0 ..< floatPixels.count {
                tensor[i] = NSNumber(value: floatPixels[i])
            }
            return tensor
        }
        catch {
            return nil
        }
    }
}

// MARK: - Feature Providers

/// Provides features for text input to a machine learning model, handling padding and attention mask generation.
class TextInput: MLFeatureProvider {
    var inputIDs: [Int]
    var sequenceLength: Int
    var paddingID: Int

    /// Initializes a new instance for providing text input features.
    /// - Parameters:
    ///   - inputIDs: Array of integer IDs representing the encoded text.
    ///   - sequenceLength: The fixed length to which the input sequence should be padded.
    ///   - paddingID: The integer ID used for padding shorter sequences. Defaults to 0.
    init(inputIDs: [Int], sequenceLength: Int, paddingID: Int = 0) {
        self.inputIDs = inputIDs
        self.sequenceLength = sequenceLength
        self.paddingID = paddingID
    }

    var featureNames: Set<String> {
        return Set(["input_ids", "attention_mask"])
    }

    /// Returns the feature value for the specified feature name.
    /// - Parameter featureName: The name of the feature for which the value is requested.
    /// - Returns: An optional `MLFeatureValue` containing the data for the specified feature.
    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "input_ids", "attention_mask":
            return createFeatureValue(for: featureName)
        default:
            return nil
        }
    }

    /// Creates the feature value for input IDs or attention mask based on the specified feature name.
    /// - Parameter featureName: The name of the feature.
    /// - Returns: An `MLFeatureValue` if the array can be created, otherwise nil.
    private func createFeatureValue(for featureName: String) -> MLFeatureValue? {
        let count = min(inputIDs.count, sequenceLength)
        let totalElements = sequenceLength
        guard let multiArray = try? MLMultiArray(shape: [1, NSNumber(value: totalElements)], dataType: .int32) else {
            return nil
        }

        if featureName == "input_ids" {
            for i in 0 ..< count {
                multiArray[i] = NSNumber(value: inputIDs[i])
            }
            for i in count ..< totalElements {
                multiArray[i] = NSNumber(value: paddingID)
            }
        }
        else if featureName == "attention_mask" {
            for i in 0 ..< count {
                multiArray[i] = NSNumber(value: 1)
            }
            for i in count ..< totalElements {
                multiArray[i] = NSNumber(value: 0)
            }
        }

        return MLFeatureValue(multiArray: multiArray)
    }
}

/// Provides a precomputed feature for image inputs to a machine learning model.
class ImageInput: MLFeatureProvider {
    var precomputedFeature: MLFeatureValue

    /// Initializes a new instance with a precomputed feature.
    /// - Parameter precomputedFeature: The `MLFeatureValue` containing the precomputed feature data.
    /// - Throws: An error if the precomputed feature is not valid for the model.
    init(precomputedFeature: MLFeatureValue) throws {
        self.precomputedFeature = precomputedFeature
    }

    var featureNames: Set<String> {
        return Set(["images"])
    }

    /// Returns the feature value for the specified feature name.
    /// - Parameter featureName: The name of the feature for which the value is requested.
    /// - Returns: An optional `MLFeatureValue` containing the data for the specified feature.
    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "images":
            return precomputedFeature
        default:
            return nil
        }
    }
}
