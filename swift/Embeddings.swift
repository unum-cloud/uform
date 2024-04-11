//
//  Embeddings.swift
//
//
//  Created by Ash Vardanian on 3/27/24.
//
import Foundation
import CoreGraphics
import Accelerate
import CoreML

import Hub // `Config`
import Tokenizers // `AutoTokenizer`

// MARK: - Helpers

func readConfig(fromPath path: String) throws -> [String: Any] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]
}

func readModel(fromPath path: String) throws -> MLModel {
    // If compilation succeeds, you can then load the compiled model
    let modelURL = URL(fileURLWithPath: path, isDirectory: true)
    let compiledModelURL = try MLModel.compileModel(at: modelURL)
    return try MLModel(contentsOf: compiledModelURL)
}

// MARK: - Encoders

public class TextEncoder {
    let model: MLModel
    let processor: TextProcessor
    
    public init(modelPath: String, configPath: String, tokenizerPath: String) throws {
        self.model = try readModel(fromPath: modelPath)
        self.processor = try TextProcessor(configPath: configPath, tokenizerPath: tokenizerPath, model: self.model)
    }
    
    public func forward(with text: String) throws -> [Float32] {
        let inputFeatureProvider = try self.processor.preprocess(text)
        let prediction = try self.model.prediction(from: inputFeatureProvider)
        let predictionFeature = prediction.featureValue(for: "embeddings")
        // The `predictionFeature` is an MLMultiArray, which can be converted to an array of Float32
        let output = predictionFeature!.multiArrayValue!
        return Array(UnsafeBufferPointer(start: output.dataPointer.assumingMemoryBound(to: Float32.self), count: Int(truncating: output.shape[1])))
    }
}


public class ImageEncoder {
    let model: MLModel
    let processor: ImageProcessor
    
    public init(modelPath: String, configPath: String) throws {
        self.model = try readModel(fromPath: modelPath)
        self.processor = try ImageProcessor(configPath: configPath)
    }
    
    public func forward(with image: CGImage) throws -> [Float32] {
        let inputFeatureProvider = try self.processor.preprocess(image)
        let prediction = try self.model.prediction(from: inputFeatureProvider)
        let predictionFeature = prediction.featureValue(for: "embeddings")
        // The `predictionFeature` is an MLMultiArray, which can be converted to an array of Float32
        let output = predictionFeature!.multiArrayValue!
        return Array(UnsafeBufferPointer(start: output.dataPointer.assumingMemoryBound(to: Float32.self), count: Int(truncating: output.shape[1])))
    }
    
}

// MARK: - Processors

class TextProcessor {
    let tokenizer: Tokenizer
    let minContextLength: Int
    let maxContextLength: Int
    
    public init(configPath: String, tokenizerPath: String, model: MLModel) throws {
        let configDict = try readConfig(fromPath: configPath)
        let tokenizerDict = try readConfig(fromPath: tokenizerPath)
        
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
    let mean: [Float] = [0.485, 0.456, 0.406] // Common mean values for normalization
    let std: [Float] = [0.229, 0.224, 0.225] // Common std values for normalization
    
    init(configPath: String) throws {
        let configDict = try readConfig(fromPath: configPath)
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
        
        let widthRatio = CGFloat(imageSize) / originalWidth
        let heightRatio = CGFloat(imageSize) / originalHeight
        let scaleFactor = max(widthRatio, heightRatio)
        
        let scaledWidth = originalWidth * scaleFactor
        let scaledHeight = originalHeight * scaleFactor
        
        let dx = (scaledWidth - CGFloat(imageSize)) / 2.0
        let dy = (scaledHeight - CGFloat(imageSize)) / 2.0
        let insetRect = CGRect(x: dx, y: dy, width: CGFloat(imageSize) - dx*2, height: CGFloat(imageSize) - dy*2)
        
        // Create a new context (off-screen canvas) with the desired dimensions
        guard let context = CGContext(
            data: nil,
            width: imageSize, height: imageSize, bitsPerComponent: image.bitsPerComponent, bytesPerRow: 0,
            space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(), bitmapInfo: image.bitmapInfo.rawValue) else { return nil }
        
        // Draw the image in the context with the specified inset (cropping as necessary)
        context.interpolationQuality = .high
        context.draw(image, in: insetRect, byTiling: false)
        
        // Extract the new image from the context
        return context.makeImage()
    }
    
    private func exportToTensorAndNormalize(image: CGImage, mean: [Float], std: [Float]) -> MLMultiArray? {
        let width = image.width
        let height = image.height
        
        // Prepare the bitmap context for drawing the image.
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: &pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 4 * width, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert pixel data to float and normalize
        let totalCount = width * height * 4
        var floatPixels = [Float](repeating: 0, count: totalCount)
        vDSP_vfltu8(pixelData, 1, &floatPixels, 1, vDSP_Length(totalCount))
        
        // Scale the pixel values to [0, 1]
        var divisor = Float(255.0)
        vDSP_vsdiv(floatPixels, 1, &divisor, &floatPixels, 1, vDSP_Length(totalCount))
        
        // Normalize the pixel values
        for c in 0..<3 {
            var slice = [Float](repeating: 0, count: width * height)
            for i in 0..<(width * height) {
                slice[i] = (floatPixels[i * 4 + c] - mean[c]) / std[c]
            }
            floatPixels.replaceSubrange(c*width*height..<(c+1)*width*height, with: slice)
        }
        
        // Rearrange the array to C x H x W
        var tensor = [Float](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<3 {
                    tensor[c * width * height + y * width + x] = floatPixels[y * width * 4 + x * 4 + c]
                }
            }
        }
        
        // Reshape the tensor to 1 x 3 x H x W and pack into a rank-3 `MLFeatureValue`
        let multiArray = try? MLMultiArray(shape: [1, 3, NSNumber(value: self.imageSize), NSNumber(value: self.imageSize)], dataType: .float32)
        for i in 0..<(width * height * 3) {
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
            for i in 0..<count {
                multiArray[i] = NSNumber(value: inputIDs[i])
            }
            for i in count..<totalElements {
                multiArray[i] = NSNumber(value: paddingID)
            }
        } else if featureName == "attention_mask" {
            for i in 0..<count {
                multiArray[i] = NSNumber(value: 1)
            }
            for i in count..<totalElements {
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

