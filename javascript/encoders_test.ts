import { getCheckpoint } from "./encoders.mts";
import { Modality } from "./encoders.mts";

// Simple function to assert conditions
function assert(condition: boolean, message: string) {
    if (!condition) {
        throw new Error(message);
    }
}

// Test case for getCheckpoint function
async function testGetCheckpoint() {
    console.log("Test getCheckpoint: Start");

    try {
        const modelId = 'uform3-image-text-english-small';  // Example model ID
        const token = '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';  // Example token
        const modalities = [Modality.TextEncoder, Modality.ImageEncoder];

        const { configPath, modalityPaths, tokenizerPath } = await getCheckpoint(
            modelId,
            modalities,
            token,
            '.onnx'
        );

        // Asserts to check if the paths are not null (indicating successful file retrieval)
        assert(configPath !== null, "Config path should not be null");
        assert(modalityPaths !== null, "Modality paths should not be null");
        assert(tokenizerPath !== null, "Tokenizer path should not be null");

        console.log("Test getCheckpoint: Success");
    } catch (error) {
        console.error("Test getCheckpoint: Failed", error);
    }
}

// Run the test
testGetCheckpoint();
