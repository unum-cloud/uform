import { readFileSync } from 'fs';
import { InferenceSession } from 'onnxruntime-web';

import { getCheckpoint, Modality } from "./hub.mjs";

import { AutoTokenizer } from '@xenova/transformers';


class TextProcessor {

    async init(configPath, tokenizerPath) {
        const config = JSON.parse(readFileSync(configPath, { encoding: 'utf8' }));
        this.maxSeqLen = config.text_encoder.max_position_embeddings;
        this.padTokenIdx = config.text_encoder.padding_idx;
        this.tokenizer = await AutoTokenizer.from_pretrained(tokenizerPath);
    }

    async processTexts(texts) {
        if (typeof texts === 'string') {
            texts = [texts];
        }

        const encoded = await this.tokenizer.encodeBatch(texts, {
            addSpecialTokens: true,
            returnAttentionMask: true,
            padding: 'max_length',
            max_length: this.maxSeqLen,
            truncation: true,
            return_tensors: 'np'
        });

        const inputIds = encoded.map(e => e.input_ids);
        const attentionMask = encoded.map(e => e.attention_mask);
        return { inputIds, attentionMask };
    }
}

export { TextProcessor };
