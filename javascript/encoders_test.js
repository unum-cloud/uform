import { existsSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

import { getCheckpoint, Modality } from "./hub.mjs";
import { TextProcessor, TextEncoder, ImageEncoder, ImageProcessor } from "./encoders.mjs";

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

// Check if the HuggingFace Hub API token is set in the environment variable.
let hf_token = process.env.HUGGINGFACE_HUB_TOKEN;
if (!hf_token) {
    const dirname = path.dirname(fileURLToPath(import.meta.url));
    const tokenPath = path.join(dirname, '../', '.hf_token');
    if (existsSync(tokenPath)) {
        hf_token = readFileSync(tokenPath, 'utf8').trim();
    }
}

async function tryGettingCheckpoint(modelId, modalities) {
    const { configPath, modalityPaths, tokenizerPath } = await getCheckpoint(
        modelId,
        modalities,
        hf_token,
        '.onnx'
    );

    assert(configPath !== null, "Config path should not be null");
    assert(modalityPaths !== null, "Modality paths should not be null");
    assert(tokenizerPath !== null, "Tokenizer path should not be null");

    // Check if the file actually exists
    assert(existsSync(configPath), `Config file should exist at ${configPath}`);
    assert(existsSync(tokenizerPath), `Tokenizer file should exist at ${tokenizerPath}`);
    for (const modalityPath of Object.values(modalityPaths)) {
        assert(existsSync(modalityPath), `Modality file should exist at ${modalityPath}`);
    }
}

async function testGetCheckpoint() {
    console.log("- `testGetCheckpoint`: Start");

    try {
        const modalities = [Modality.TextEncoder, Modality.ImageEncoder];

        for (const modelId of [
            'unum-cloud/uform3-image-text-english-small',
            'unum-cloud/uform3-image-text-english-base',
            'unum-cloud/uform3-image-text-english-large',
            'unum-cloud/uform3-image-text-multilingual-base',
        ]) {
            await tryGettingCheckpoint(modelId, modalities, hf_token);
        }

        console.log("- `testGetCheckpoint`: Success");
    } catch (error) {
        console.error("- `testGetCheckpoint`: Failed", error);
    }
}

async function tryTextEncoderForwardPass(modelId) {
    const modalities = [Modality.TextEncoder];
    const { configPath, modalityPaths, tokenizerPath } = await getCheckpoint(
        modelId,
        modalities,
        hf_token,
        '.onnx'
    );

    const textProcessor = new TextProcessor(configPath, tokenizerPath);
    await textProcessor.init();
    const processedTexts = await textProcessor.process("Hello, world!");

    const textEncoder = new TextEncoder(modalityPaths.text_encoder, textProcessor);
    await textEncoder.init();
    const textOutput = await textEncoder.forward(processedTexts);
    assert(textOutput.embeddings.dims.length === 2, "Output should be 2D");

    await textEncoder.dispose();
}

async function tryImageEncoderForwardPass(modelId) {
    const modalities = [Modality.ImageEncoder];
    const { configPath, modalityPaths } = await getCheckpoint(
        modelId,
        modalities,
        hf_token,
        '.onnx'
    );

    const imageProcessor = new ImageProcessor(configPath);
    await imageProcessor.init();
    const processedImages = await imageProcessor.process("assets/unum.png");

    const imageEncoder = new ImageEncoder(modalityPaths.image_encoder, imageProcessor);
    await imageEncoder.init();
    const imageOutput = await imageEncoder.forward(processedImages);
    assert(imageOutput.embeddings.dims.length === 2, "Output should be 2D");

    await imageEncoder.dispose();
}

async function testEncoders() {
    console.log("- `testEncoders`: Start");

    try {

        // Go through the bi-modal models
        for (const modelId of [
            'unum-cloud/uform3-image-text-english-small',
            'unum-cloud/uform3-image-text-english-base',
            'unum-cloud/uform3-image-text-english-large',
            'unum-cloud/uform3-image-text-multilingual-base',
        ]) {
            await tryTextEncoderForwardPass(modelId, hf_token);
            await tryImageEncoderForwardPass(modelId, hf_token);
        }

        console.log("- `testEncoders`: Success");
    } catch (error) {
        console.error("- `testEncoders`: Failed", error);
    }
}

testGetCheckpoint();
testEncoders();
