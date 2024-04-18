import * as ort from 'onnxruntime-web';
import { AutoTokenizer, PreTrainedTokenizer } from '@xenova/transformers';

type ModelConfig = {
    modelPath: string;
    tokenizerPath: string;
};

class TextEncoder {
    private session: ort.InferenceSession;
    private tokenizer: PreTrainedTokenizer;

    constructor(private config: ModelConfig) {}

    async init(): Promise<void> {
        this.tokenizer = await AutoTokenizer.from_pretrained(this.config.tokenizerPath);
        this.session = await ort.InferenceSession.create(this.config.modelPath);
    }

    async forward(text: string): Promise<{ features: Uint8Array, embeddings: Uint8Array }> {
        // Tokenization
        const { input_ids } = await this.tokenizer(text);
        const tensorInputIds = new ort.Tensor('float32', Float32Array.from(input_ids), [1, input_ids.length]);
        const tensorAttentionMask = new ort.Tensor('float32', new Float32Array(input_ids.length).fill(1), [1, input_ids.length]);

        // Model inference
        const feeds = { input_ids: tensorInputIds, attention_mask: tensorAttentionMask };
        const results = await this.session.run(feeds);

        // Assume output tensors are in results['features'] and results['embeddings']
        const features = results['features'].data as Uint8Array!
        const embeddings = results['embeddings'].data as Uint8Array!

        return { features, embeddings };
    }
}

// Usage
async function main() {
    const textEncoder = new TextEncoder({
        modelPath: './text_encoder.onnx',
        tokenizerPath: 'Xenova/bert-base-uncased'
    });

    await textEncoder.init();
    const result = await textEncoder.forward('I love transformers!');
    console.log('Features:', result.features);
    console.log('Embeddings:', result.embeddings);
}

main();




