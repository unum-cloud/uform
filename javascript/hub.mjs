import { downloadFile, listFiles } from "@huggingface/hub";

const Modality = {
    TextEncoder: "text_encoder",
    ImageEncoder: "image_encoder",
    VideoEncoder: "video_encoder",
    TextDecoder: "text_decoder",
};

function isModality(value) {
    return Object.values(Modality).includes(value);
}

function normalizeModalities(modalities) {
    return modalities.map(x => {
        if (typeof x === "string") {
            if (isModality(x)) {
                return x;
            } else {
                throw new Error(`Invalid modality: ${x}`);
            }
        }
        return x;
    });
}

async function getCheckpoint(
    modelId, modalities, token = null, format = '.onnx',
) {
    modalities = normalizeModalities(modalities);

    const configNames = ['config.json'];
    const tokenizerNames = ['tokenizer.json'];
    const modelFileNames = modalities.map(modality => `${modality}${format}`);
    const allowedPatterns = [...modelFileNames, ...configNames, ...tokenizerNames];

    const repo = { type: "model", name: modelId };
    const credentials = token ? { accessToken: token } : undefined;

    let configPath = null;
    let tokenizerPath = null;
    const modalityPaths = {};

    const fileIterator = listFiles({ repo, recursive: true, credentials });
    for await (const file of fileIterator) {
        const fileName = file.path.split('/').pop();
        if (fileName && allowedPatterns.includes(fileName)) {
            const filePath = file.path;
            if (configNames.includes(fileName)) {
                configPath = filePath;
            } else if (tokenizerNames.includes(fileName)) {
                tokenizerPath = filePath;
            } else {
                const modalityName = fileName.split('.')[0];
                modalityPaths[modalityName] = filePath;
            }

            const response = await downloadFile({ repo, path: filePath, credentials });
            if (response) {
                console.log(`Downloaded ${fileName} successfully to ${response.json()}`);
            }
        }
    }

    return { configPath, modalityPaths, tokenizerPath };
}

export { getCheckpoint, Modality };
