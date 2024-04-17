# UForm for Swift

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
let textEmbedding: Embedding = try textModel.forward(with: text)
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

var imageEmbedding: Embedding = try imageModel.forward(with: cgImage)
var imageVector: [Float32] = embedding.asFloats()
```


### Computing Distances