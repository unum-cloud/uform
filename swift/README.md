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
let textModel = try await TextEncoder(modelName: "unum-cloud/uform3-image-text-english-small")
let text = "A group of friends enjoy a barbecue on a sandy beach, with one person grilling over a large black grill, while the other sits nearby, laughing and enjoying the camaraderie."
let textEmbedding: Embedding = try textModel.encode(text)
let textVector: [Float32] = textEmbedding.asFloats()
```

### Image Embeddings

```swift
let imageModel = try await ImageEncoder(modelName: "unum-cloud/uform3-image-text-english-small")
let imageURL = "https://github.com/ashvardanian/ashvardanian/blob/master/demos/bbq-on-beach.jpg?raw=true"
guard let url = URL(string: imageURL),
    let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
    let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) {
    throw Exception("Could not load image from URL: \(imageURL)")
}

var imageEmbedding: Embedding = try imageModel.encode(cgImage)
var imageVector: [Float32] = embedding.asFloats()
```

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
