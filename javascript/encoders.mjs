import { readFileSync } from 'fs';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { getCheckpoint, Modality } from "./hub.mjs";
import { PreTrainedTokenizer } from '@xenova/transformers';


class TextProcessor {

    constructor(configPath, tokenizerPath) {
        this.configPath = configPath;
        this.tokenizerPath = tokenizerPath;

        this.maxSeqLen = 0;
        this.padTokenIdx = 0;
        this.tokenizer = null;
    }

    async init() {
        const config = JSON.parse(readFileSync(this.configPath, { encoding: 'utf8' }));
        this.maxSeqLen = config.text_encoder.max_position_embeddings;
        this.padTokenIdx = config.text_encoder.padding_idx;

        const tokenizerConfig = JSON.parse(readFileSync(this.tokenizerPath, { encoding: 'utf8' }));
        this.tokenizer = new PreTrainedTokenizer(tokenizerConfig, config.text_encoder);
        this.tokenizer.model_max_length = this.maxSeqLen;
        this.tokenizer.pad_token_id = this.padTokenIdx;
    }

    async process(texts) {

        const encoded = await this.tokenizer(texts, {
            addSpecialTokens: true,
            returnAttentionMask: true,
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

class TextEncoder {

    constructor(configPath, modelPath, tokenizerPath) {
        this.configPath = configPath;
        this.modelPath = modelPath;
        this.tokenizerPath = tokenizerPath;

        this.session = null;
    }

    async init() {
        this.session = await InferenceSession.create(this.modelPath);
    }

    async forward(inputs) {
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

export { TextProcessor, TextEncoder };
