# Contributing to UForm

We welcome contributions to UForm!

## Python

Before submitting any changes, please make sure that the tests pass.

```sh
pip install -e ".[dev]"         # For development dependencies
pip install -e ".[torch]"       # For PyTorch
pip install -e ".[onnx]"        # For ONNX on CPU
pip install -e ".[onnx-gpu]"    # For ONNX on GPU, available for some platforms
pip install -e ".[torch,onnx,onnx-gpu,dev]"  # For all

pytest python/scripts/ -s -x -Wd -v
pytest python/scripts/ -s -x -Wd -v -k onnx # To run only ONNX tests without loading Torch
```

## Swift

To build and test the Swift package, use the following command:

```bash
swift build
swift test
```

Swift formatting is enforced with `swift-format` default utility from Apple.
To install and run it on all the files in the project, use the following command:

```bash
brew install swift-format
swift-format . -i -r
```

The style is controlled by the `.swift-format` JSON file in the root of the repository.
As there is no standard for Swift formatting, even Apple's own `swift-format` tool and Xcode differ in their formatting rules, and available settings.

## JavaScript

For rapid development you can avoid the TypeScript precompilation step:

```sh
npm install -g ts-node
ts-node javascript/embeddings.mts
```

Before submitting any changes, please make sure that the tests pass.

```sh
npm install
npm run test
```

## Benchmarking

If you want to double check, how fast the model may work on your hardware, you can clone the library and repeat the benchmarks locally.
The following benchmark will exclude PyTorch backend, CUDA-capable devices, and all the `-base` and `-large` models, running only the ONNX benchmarks on the CPU.

```sh
git clone https://github.com/unum-cloud/uform --depth 1 # Clone the repository
cd uform && pip install -e ".[torch,onnx,onnx-gpu,dev]" # Install all dependencies
python python/scripts/bench_encoders.py --filter-out "torch|cuda|base|large"
```

