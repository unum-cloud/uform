import UForm

import XCTest
import CoreGraphics
import ImageIO

final class TokenizerTests: XCTestCase {
    
    
    func cosineSimilarity<T: FloatingPoint>(between vectorA: [T], and vectorB: [T]) -> T {
        guard vectorA.count == vectorB.count else {
            fatalError("Vectors must be of the same length.")
        }

        let dotProduct = zip(vectorA, vectorB).reduce(T.zero) { $0 + ($1.0 * $1.1) }
        let magnitudeA = sqrt(vectorA.reduce(T.zero) { $0 + $1 * $1 })
        let magnitudeB = sqrt(vectorB.reduce(T.zero) { $0 + $1 * $1 })

        // Avoid division by zero
        if magnitudeA == T.zero || magnitudeB == T.zero {
            return T.zero
        }

        return dotProduct / (magnitudeA * magnitudeB)
    }

    
    func testTextEmbeddings() async throws {
        let model = try TextEncoder(
            modelPath: "uform/uform-vl-english-small-text.mlpackage",
            configPath: "uform/config.json",
            tokenizerPath: "uform/tokenizer.json"
        )
        
        let texts = [
            "sunny beach with clear blue water",
            "crowded sandbeach under the bright sun",
            "dense forest with tall green trees",
            "quiet park in the morning light"
        ]
        
        var embeddings: [[Float32]] = []
        for text in texts {
            let embedding: [Float32] = try model.forward(with: text)
            embeddings.append(embedding)
        }
        
        // Now let's compute the cosine similarity between the embeddings
        let similarityBeach = cosineSimilarity(between: embeddings[0], and: embeddings[1])
        let similarityForest = cosineSimilarity(between: embeddings[2], and: embeddings[3])
        let dissimilarityBetweenScenes = cosineSimilarity(between: embeddings[0], and: embeddings[2])
        
        // Assert that similar texts have higher similarity scores
        XCTAssertTrue(similarityBeach > dissimilarityBetweenScenes, "Beach texts should be more similar to each other than to forest texts.")
        XCTAssertTrue(similarityForest > dissimilarityBetweenScenes, "Forest texts should be more similar to each other than to beach texts.")
    }
    
    func testImageEmbeddings() async throws {
        let model = try ImageEncoder(
            modelPath: "uform/uform-vl-english-small-image.mlpackage",
            configPath: "uform/config_image.json"
        )
        
        let imageURLs = [
            "https://github.com/ashvardanian/ashvardanian/blob/master/demos/bbq-on-beach.jpg?raw=true",
            "https://github.com/ashvardanian/ashvardanian/blob/master/demos/cat-in-garden.jpg?raw=true",
            "https://github.com/ashvardanian/ashvardanian/blob/master/demos/girl-and-rain.jpg?raw=true",
            "https://github.com/ashvardanian/ashvardanian/blob/master/demos/light-bedroom-furniture.jpg?raw=true",
            "https://github.com/ashvardanian/ashvardanian/blob/master/demos/louvre-at-night.jpg?raw=true",
        ]

        var embeddings: [[Float32]] = []
        for imageURL in imageURLs {
            guard let url = URL(string: imageURL),
                  let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
                throw NSError(domain: "ImageError", code: 100, userInfo: [NSLocalizedDescriptionKey: "Could not load image from URL: \(imageURL)"])
            }
            
            let embedding: [Float32] = try model.forward(with: cgImage)
            embeddings.append(embedding)
        }

        // Now let's compute the cosine similarity between the embeddings
        let similarityGirlAndBeach = cosineSimilarity(between: embeddings[2], and: embeddings[0])
        let similarityGirlAndLouvre = cosineSimilarity(between: embeddings[2], and: embeddings[4])
        let similarityBeachAndLouvre = cosineSimilarity(between: embeddings[0], and: embeddings[4])

        // Assert that similar images have higher similarity scores
        XCTAssertTrue(similarityGirlAndBeach > similarityGirlAndLouvre, "");
        XCTAssertTrue(similarityGirlAndBeach > similarityBeachAndLouvre, "");
    }
    
    
}
