import CoreGraphics
import ImageIO
import UForm
import XCTest

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

        let root = "/uform/"
        let textModel = try TextEncoder(
            modelPath: root + "uform-vl-english-large-text.mlpackage",
            configPath: root + "uform-vl-english-large-text.json",
            tokenizerPath: root + "uform-vl-english-large-text.tokenizer.json"
        )

        let texts = [
            "sunny beach with clear blue water",
            "crowded sandbeach under the bright sun",
            "dense forest with tall green trees",
            "quiet park in the morning light",
        ]

        var textEmbeddings: [[Float32]] = []
        for text in texts {
            let embedding: [Float32] = try textModel.forward(with: text)
            textEmbeddings.append(embedding)
        }

        // Now let's compute the cosine similarity between the textEmbeddings
        let similarityBeach = cosineSimilarity(between: textEmbeddings[0], and: textEmbeddings[1])
        let similarityForest = cosineSimilarity(between: textEmbeddings[2], and: textEmbeddings[3])
        let dissimilarityBetweenScenes = cosineSimilarity(between: textEmbeddings[0], and: textEmbeddings[2])

        // Assert that similar texts have higher similarity scores
        XCTAssertTrue(
            similarityBeach > dissimilarityBetweenScenes,
            "Beach texts should be more similar to each other than to forest texts."
        )
        XCTAssertTrue(
            similarityForest > dissimilarityBetweenScenes,
            "Forest texts should be more similar to each other than to beach texts."
        )
    }

    func testImageEmbeddings() async throws {

        let root = "/uform/"
        let textModel = try TextEncoder(
            modelPath: root + "uform-vl-english-large-text.mlpackage",
            configPath: root + "uform-vl-english-large-text.json",
            tokenizerPath: root + "uform-vl-english-large-text.tokenizer.json"
        )
        let imageModel = try ImageEncoder(
            modelPath: root + "uform-vl-english-large-image.mlpackage",
            configPath: root + "uform-vl-english-large-image.json"
        )

        let texts = [
            "A group of friends enjoy a barbecue on a sandy beach, with one person grilling over a large black grill, while the other sits nearby, laughing and enjoying the camaraderie.",
            "A white and orange cat stands on its hind legs, reaching towards a wicker basket filled with red raspberries on a wooden table in a garden, surrounded by orange flowers and a white teapot, creating a serene and whimsical scene.",
            "A young girl in a yellow dress stands in a grassy field, holding an umbrella and looking at the camera, amidst rain.",
            "This serene bedroom features a white bed with a black canopy, a gray armchair, a black dresser with a mirror, a vase with a plant, a window with white curtains, a rug, and a wooden floor, creating a tranquil and elegant atmosphere.",
            "The image captures the iconic Louvre Museum in Paris, illuminated by warm lights against a dark sky, with the iconic glass pyramid in the center, surrounded by ornate buildings and a large courtyard, showcasing the museum's grandeur and historical significance.",
        ]
        let imageURLs = [
            "https://github.com/ashvardanian/ashvardanian/blob/master/demos/bbq-on-beach.jpg?raw=true",
            "https://github.com/ashvardanian/ashvardanian/blob/master/demos/cat-in-garden.jpg?raw=true",
            "https://github.com/ashvardanian/ashvardanian/blob/master/demos/girl-and-rain.jpg?raw=true",
            "https://github.com/ashvardanian/ashvardanian/blob/master/demos/light-bedroom-furniture.jpg?raw=true",
            "https://github.com/ashvardanian/ashvardanian/blob/master/demos/louvre-at-night.jpg?raw=true",
        ]

        var textEmbeddings: [[Float32]] = []
        var imageEmbeddings: [[Float32]] = []
        for (text, imageURL) in zip(texts, imageURLs) {
            guard let url = URL(string: imageURL),
                let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
                let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
            else {
                throw NSError(
                    domain: "ImageError",
                    code: 100,
                    userInfo: [NSLocalizedDescriptionKey: "Could not load image from URL: \(imageURL)"]
                )
            }

            let textEmbedding: [Float32] = try textModel.forward(with: text)
            textEmbeddings.append(textEmbedding)
            let imageEmbedding: [Float32] = try imageModel.forward(with: cgImage)
            imageEmbeddings.append(imageEmbedding)
        }

        // Now let's make sure that the cosine distance between image and respective text embeddings is low.
        // Make sure that the similarity between image and text at index `i` is higher than with other texts and images.
        for i in 0 ..< texts.count {
            let pairSimilarity = cosineSimilarity(between: textEmbeddings[i], and: imageEmbeddings[i])
            let otherTextSimilarities = (0 ..< texts.count).filter { $0 != i }.map {
                cosineSimilarity(between: textEmbeddings[$0], and: imageEmbeddings[i])
            }
            let otherImageSimilarities = (0 ..< texts.count).filter { $0 != i }.map {
                cosineSimilarity(between: textEmbeddings[i], and: imageEmbeddings[$0])
            }

            XCTAssertTrue(
                pairSimilarity > otherTextSimilarities.max()!,
                "Text should be more similar to its corresponding image than to other images."
            )
            XCTAssertTrue(
                pairSimilarity > otherImageSimilarities.max()!,
                "Text should be more similar to its corresponding image than to other texts."
            )
        }
    }

}
