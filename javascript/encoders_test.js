import { existsSync } from 'fs';

import { getCheckpoint, Modality } from "./hub.mjs";
import { TextProcessor, TextEncoder, ImageEncoder, ImageProcessor } from "./encoders.mjs";

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

async function testGetCheckpoint() {
    console.log("Test getCheckpoint: Start");

    try {
        const modelId = 'unum-cloud/uform3-image-text-english-small';
        const token = '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';
        const modalities = [Modality.TextEncoder, Modality.ImageEncoder];

        const { configPath, modalityPaths, tokenizerPath } = await getCheckpoint(
            modelId,
            modalities,
            token,
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

        console.log("Test getCheckpoint: Success");
    } catch (error) {
        console.error("Test getCheckpoint: Failed", error);
    }
}

async function testEncoders() {
    console.log("Test testEncoders: Start");
    let textEncoder = null;
    let imageEncoder = null;

    try {
        const modelId = 'unum-cloud/uform3-image-text-english-small';
        const token = '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';
        const modalities = [Modality.TextEncoder, Modality.ImageEncoder];

        const { configPath, modalityPaths, tokenizerPath } = await getCheckpoint(
            modelId,
            modalities,
            token,
            '.onnx'
        );

        assert(configPath !== null, "Config path should not be null");
        assert(modalityPaths !== null, "Modality paths should not be null");
        assert(tokenizerPath !== null, "Tokenizer path should not be null");

        const textProcessor = new TextProcessor(configPath, tokenizerPath);
        await textProcessor.init();
        const processedTexts = await textProcessor.process("Hello, world!");

        textEncoder = new TextEncoder(modalityPaths.text_encoder, textProcessor);
        await textEncoder.init();
        const textOutput = await textEncoder.forward(processedTexts);
        console.log(textOutput.embeddings.dims);

        const imageProcessor = new ImageProcessor(configPath);
        await imageProcessor.init();
        const processedImages = await imageProcessor.process("assets/unum.png");

        imageEncoder = new ImageEncoder(modalityPaths.image_encoder, imageProcessor);
        await imageEncoder.init();
        const imageOutput = await imageEncoder.forward(processedImages);
        console.log(imageOutput.embeddings.dims);

        console.log("Test testEncoders: Success");
    } catch (error) {
        console.error("Test testEncoders: Failed", error);
    } finally {
        await textEncoder.dispose();
        await imageEncoder.dispose();
    }
}

testGetCheckpoint();
testEncoders();
