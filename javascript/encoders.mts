import { downloadFile, listFiles, RepoDesignation, Credentials } from "@huggingface/hub";

export enum Modality {
    TextEncoder = "text_encoder",
    ImageEncoder = "image_encoder",
    VideoEncoder = "video_encoder",
    TextDecoder = "text_decoder",
}

function isModality(key: any): key is keyof typeof Modality {
    return Object.keys(Modality).includes(key);
}

function normalizeModalities(modalities: Array<string | Modality>): Array<Modality> {
    return modalities.map(x => {
        if (typeof x === "string") {
            if (isModality(Modality[x as keyof typeof Modality])) {
                return Modality[x as keyof typeof Modality];
            } else {
                throw new Error(`Invalid modality: ${x}`);
            }
        }
        return x;
    });
}

export async function getCheckpoint(
    modelId: string,
    modalities: Array<string | Modality>,
    token: string | null = null,
    format: '.pt' | '.onnx' = '.onnx'
): Promise<{ configPath: string | null, modalityPaths: Record<string, string> | null, tokenizerPath: string | null }> {
    modalities = normalizeModalities(modalities);

    const configNames = ['config.json'];
    const tokenizerNames = ['tokenizer.json'];
    const modelFileNames = modalities.map(modality => `${modality}${format}`);
    const allowedPatterns = [...modelFileNames, ...configNames, ...tokenizerNames];

    const repo: RepoDesignation = { type: "model", name: modelId };
    const credentials: Credentials | undefined = token ? { accessToken: token } : undefined;

    let configPath: string | null = null;
    let tokenizerPath: string | null = null;
    const modalityPaths: Record<string, string> = {};

    // List files and directly process
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

            // Download the file
            const response = await downloadFile({ repo, path: filePath, credentials });
            if (response) {
                // Handle file response, save locally or process in-memory as needed
                // Example: Save to a local file or process the file contents
                console.log(`Downloaded ${fileName} successfully.`);
            }
        }
    }

    return { configPath, modalityPaths, tokenizerPath };
}

