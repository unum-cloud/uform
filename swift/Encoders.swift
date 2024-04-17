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


enum EncoderError: Error {
    case configLoadingError(String)
    case modelLoadingError(String)
    case unsupportedDataType
    case invalidInput
    case unsupportedShapeConstraint
    case modelPredictionFailed(String)
}


public enum Embedding {
    case i32s([Int32])
    case f16s([Float16])
    case f32s([Float32])
    case f64s([Float64])

    init?(from multiArray: MLMultiArray) {
        switch multiArray.dataType {
        case .float64:
            self = .f64s(
                Array(
                    UnsafeBufferPointer(
                        start: multiArray.dataPointer.assumingMemoryBound(to: Float64.self),
                        count: Int(truncating: multiArray.shape[1])
                    )
                )
            )
        case .float32:
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
            return nil  // return nil for unsupported data types
        }
    }

    public func asFloats() -> [Float] {
        switch self {
        case .f32s(let array):
            return array
        case .i32s(let array):
            return array.map { Float($0) }
        case .f16s(let array):
            return array.map { Float($0) }
        case .f64s(let array):
            return array.map { Float($0) }
        }
    }
}

// MARK: - Helpers

func readConfig(fromPath path: String) throws -> [String: Any] {
    // If it's not an absolute path, let's assume it's a path relative to the current working directory
    let absPath = path.hasPrefix("/") ? path : FileManager.default.currentDirectoryPath + "/" + path
    let data = try Data(contentsOf: URL(fileURLWithPath: absPath))
    return try JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]
}

func readModel(fromURL modelURL: URL) throws -> MLModel {
    let compiledModelURL = try MLModel.compileModel(at: modelURL)
    return try MLModel(contentsOf: compiledModelURL)
}

func readModel(fromPath path: String) throws -> MLModel {
    // If it's not an absolute path, let's assume it's a path relative to the current working directory
    let absPath = path.hasPrefix("/") ? path : FileManager.default.currentDirectoryPath + "/" + path
    let modelURL = URL(fileURLWithPath: absPath, isDirectory: true)
    return try readModel(fromURL: modelURL)
}

// MARK: - Encoders

public class TextEncoder {
    let model: MLModel
    let processor: TextProcessor

    public init(modelPath: String, configPath: String? = nil, tokenizerPath: String? = nil) throws {
        let finalConfigPath = configPath ?? modelPath + "/config.json"
        let finalTokenizerPath = tokenizerPath ?? modelPath + "/tokenizer.json"
        self.model = try readModel(fromPath: modelPath)
        self.processor = try TextProcessor(configPath: finalConfigPath, tokenizerPath: finalTokenizerPath, model: self.model)
    }

    
    public init(modelName: String, hubApi: HubApi = .shared) async throws {
        let repo = Hub.Repo(id: modelName)
        let modelURL = try await hubApi.snapshot(from: repo, matching: ["text.mlpackage/*", "config.json", "tokenizer.json"])
        let configPath = modelURL.appendingPathComponent("config.json").path
        let tokenizerPath = modelURL.appendingPathComponent("tokenizer.json").path
        self.model = try readModel(fromURL: modelURL.appendingPathComponent("text.mlpackage", isDirectory: true))
        self.processor = try TextProcessor(configPath: configPath, tokenizerPath: tokenizerPath, model: self.model)
    }

    public func forward(with text: String) throws -> Embedding {
        let inputFeatureProvider = try self.processor.preprocess(text)
        let prediction = try self.model.prediction(from: inputFeatureProvider)
        guard let predictionFeature = prediction.featureValue(for: "embeddings"),
            let output = predictionFeature.multiArrayValue,
            let embedding = Embedding(from: output)
        else {
            throw NSError(
                domain: "TextEncoder",
                code: 0,
                userInfo: [NSLocalizedDescriptionKey: "Failed to extract embeddings or unsupported data type."]
            )
        }
        return embedding
    }
}

public class ImageEncoder {
    let model: MLModel
    let processor: ImageProcessor

    public init(modelPath: String, configPath: String? = nil) throws {
        let finalConfigPath = configPath ?? modelPath + "/config.json"
        self.model = try readModel(fromPath: modelPath)
        self.processor = try ImageProcessor(configPath: finalConfigPath)
    }

    public init(modelName: String, hubApi: HubApi = .shared) async throws {
        let repo = Hub.Repo(id: modelName)
        let modelURL = try await hubApi.snapshot(from: repo, matching: ["image.mlpackage/*", "config.json"])
        let configPath = modelURL.appendingPathComponent("config.json").path
        self.model = try readModel(fromURL: modelURL.appendingPathComponent("image.mlpackage", isDirectory: true))
        self.processor = try ImageProcessor(configPath: configPath)
    }
    
    public func forward(with image: CGImage) throws -> Embedding {
        let inputFeatureProvider = try self.processor.preprocess(image)
        let prediction = try self.model.prediction(from: inputFeatureProvider)
        guard let predictionFeature = prediction.featureValue(for: "embeddings"),
            let output = predictionFeature.multiArrayValue,
            let embedding = Embedding(from: output)
        else {
            throw NSError(
                domain: "ImageEncoder",
                code: 0,
                userInfo: [NSLocalizedDescriptionKey: "Failed to extract embeddings or unsupported data type."]
            )
        }
        return embedding
    }
}

// MARK: - Processors

class TextProcessor {
    let tokenizer: Tokenizer
    let minContextLength: Int
    let maxContextLength: Int

    public init(configPath: String, tokenizerPath: String, model: MLModel) throws {
        var configDict = try readConfig(fromPath: configPath)
        let tokenizerDict = try readConfig(fromPath: tokenizerPath)

        // Check if there's a specific 'text_encoder' configuration within the main configuration
        if let textEncoderConfig = configDict["text_encoder"] as? [String: Any] {
            configDict = textEncoderConfig  // Use the specific 'text_encoder' configuration
        }

        let config = Config(configDict)
        let tokenizerData = Config(tokenizerDict)
        self.tokenizer = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: tokenizerData)

        let inputDescription = model.modelDescription.inputDescriptionsByName["input_ids"]
        guard let shapeConstraint = inputDescription?.multiArrayConstraint?.shapeConstraint else {
            fatalError("Cannot obtain shape information")
        }

        switch shapeConstraint.type {
        case .enumerated:
            minContextLength = shapeConstraint.enumeratedShapes[0][1].intValue
            maxContextLength = minContextLength
        case .range:
            let range = inputDescription?.multiArrayConstraint?.shapeConstraint.sizeRangeForDimension[1] as? NSRange
            minContextLength = range?.location ?? 1
            maxContextLength = range?.length ?? 128
        case .unspecified:
            minContextLength = 128
            maxContextLength = 128
        @unknown default:
            minContextLength = 128
            maxContextLength = 128
        }
    }

    public func preprocess(_ text: String) throws -> MLFeatureProvider {
        let inputIDs = self.tokenizer.encode(text: text)
        return TextInput(inputIDs: inputIDs, sequenceLength: self.maxContextLength)
    }
}

class ImageProcessor {
    let imageSize: Int
    let mean: [Float] = [0.485, 0.456, 0.406]  // Common mean values for normalization
    let std: [Float] = [0.229, 0.224, 0.225]  // Common std values for normalization

    init(configPath: String) throws {
        var configDict = try readConfig(fromPath: configPath)
        // Check if there's a specific 'image_encoder' configuration within the main configuration
        if let imageEncoderConfig = configDict["image_encoder"] as? [String: Any] {
            configDict = imageEncoderConfig
        }
        
        let config = Config(configDict)
        self.imageSize = config.imageSize!.intValue!
    }

    func preprocess(_ cgImage: CGImage) throws -> MLFeatureProvider {
        // Populate a tensor of size 3 x `imageSize` x `imageSize`,
        // by resizing the image, then performing a center crop.
        // Then normalize with the `mean` and `std` and export as a provider.
        let cropped = resizeAndCrop(image: cgImage, toSideLength: self.imageSize)!
        let normalized = exportToTensorAndNormalize(image: cropped, mean: self.mean, std: self.std)!
        let featureValue = MLFeatureValue(multiArray: normalized)
        return try ImageInput(precomputedFeature: featureValue)
    }

    private func resizeAndCrop(image: CGImage, toSideLength imageSize: Int) -> CGImage? {
        let originalWidth = CGFloat(image.width)
        let originalHeight = CGFloat(image.height)

        // Calculate new size preserving the aspect ratio
        let widthRatio = CGFloat(imageSize) / originalWidth
        let heightRatio = CGFloat(imageSize) / originalHeight
        let scaleFactor = max(widthRatio, heightRatio)

        let scaledWidth = originalWidth * scaleFactor
        let scaledHeight = originalHeight * scaleFactor

        // Calculate the crop rectangle
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
        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 4 * width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Normalize the pixel data
        var floatPixels = [Float](repeating: 0, count: width * height * 3)
        for c in 0 ..< 3 {
            for i in 0 ..< (width * height) {
                floatPixels[i * 3 + c] = (Float(pixelData[i * 4 + c]) / 255.0 - mean[c]) / std[c]
            }
        }

        // Create the tensor array
        var tensor = [Float](repeating: 0, count: 3 * width * height)
        for i in 0 ..< (width * height) {
            for c in 0 ..< 3 {
                tensor[c * width * height + i] = floatPixels[i * 3 + c]
            }
        }

        let multiArray = try? MLMultiArray(
            shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )
        for i in 0 ..< tensor.count {
            multiArray?[i] = NSNumber(value: tensor[i])
        }
        return multiArray
    }

}

// MARK: - Feature Providers

class TextInput: MLFeatureProvider {
    var inputIDs: [Int]
    var sequenceLength: Int
    var paddingID: Int

    init(inputIDs: [Int], sequenceLength: Int, paddingID: Int = 0) {
        self.inputIDs = inputIDs
        self.sequenceLength = sequenceLength
        self.paddingID = paddingID
    }

    var featureNames: Set<String> {
        return Set(["input_ids", "attention_mask"])
    }

    // The model expects the input IDs to be an array of integers
    // of length `sequenceLength`, padded with `paddingID` if necessary
    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "input_ids", "attention_mask":
            return createFeatureValue(for: featureName)
        default:
            return nil
        }
    }

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

class ImageInput: MLFeatureProvider {
    var precomputedFeature: MLFeatureValue

    init(precomputedFeature: MLFeatureValue) throws {
        self.precomputedFeature = precomputedFeature
    }

    var featureNames: Set<String> {
        return Set(["input"])
    }

    // The model expects the input IDs to be an array of integers
    // of length `sequenceLength`, padded with `paddingID` if necessary
    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "input":
            return precomputedFeature
        default:
            return nil
        }
    }
}
