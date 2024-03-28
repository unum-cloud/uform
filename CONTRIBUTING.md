# Contributing to UForm

We welcome contributions to UForm!
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