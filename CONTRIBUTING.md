# Contributing to UForm

We welcome contributions to UForm!

## Python

Before submitting any changes, please make sure that the tests pass.

```sh
pip install -e .                # For core dependencies

pip install -e ".[torch]"       # For PyTorch
pip install -e ".[onnx]"        # For ONNX on CPU
pip install -e ".[onnx-gpu]"    # For ONNX on GPU, available for some platforms
pip install -e ".[torch,onnx]"  # For PyTorch and ONNX Python tests

pytest python/scripts/ -s -x -Wd -v
pytest python/scripts/ -s -x -Wd -v -k onnx # To run only ONNX tests without loading Torch
```

## Swift

Swift formatting is enforced with `swift-format` default utility from Apple.
To install and run it on all the files in the project, use the following command:

```bash
brew install swift-format
swift-format . -i -r
```

The style is controlled by the `.swift-format` JSON file in the root of the repository.
As there is no standard for Swift formatting, even Apple's own `swift-format` tool and Xcode differ in their formatting rules, and available settings.

## JavaScript

Before submitting any changes, please make sure that the tests pass.

```sh
npm install
npm run test
```
