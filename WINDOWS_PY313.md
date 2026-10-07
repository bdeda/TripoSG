# Windows / Python 3.13 / CUDA 13 build notes

Verified: Windows 11, Python 3.13, torch 2.14.1+cu132, CUDA toolkit 13.3, VS 2022 (MSVC 14.42), RTX 4060 Ti.
`scripts/inference_triposg.py` generates a watertight mesh from `assets/example_data/hjswed.png`, with and without `--faces`.

## Install

```powershell
py -3.13 -m venv venv
venv\Scripts\python -m pip install --upgrade pip setuptools wheel
# Install torch from the CUDA index FIRST and check it is a +cu build; later installs can
# silently replace it with the CPU wheel from PyPI.
venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Everything in requirements.txt except diso (built below) installs from wheels:
venv\Scripts\python -m pip install diffusers transformers einops huggingface_hub opencv-python trimesh omegaconf scikit-image numpy peft jaxtyping typeguard pymeshlab
```

## Building `diso` (CUDA extension, no prebuilt wheel)

Run from an MSVC x64 environment (`vcvars64.bat`), with the venv active:

```bat
set "DISTUTILS_USE_SDK=1"
set "CL=/std:c++20 /Zc:preprocessor"
set "TORCH_CUDA_ARCH_LIST=8.9"
pip install --no-build-isolation diso
```

- `DISTUTILS_USE_SDK=1`: torch's `BuildExtension` refuses to run in an activated VC environment without it.
- `/std:c++20`: torch 2.14 headers need C++20 (`std::strong_ordering`, designated initializers).
- `/Zc:preprocessor`: CUDA 13's CCCL headers reject MSVC's traditional preprocessor.
- Set `TORCH_CUDA_ARCH_LIST` to your GPU (8.9 = RTX 40-series).
- Quote `set "VAR=value"`; an unquoted `set VAR=1 &&` stores a trailing space.

Verify with `python -c "from diso import DiffDMC"`, not the pip exit code.

## Code changes

`triposg/inference_utils.py`: the mesh-extraction `try/except` blocks used to print the error and return
`(None, None)`, which then crashed later with an unrelated `AttributeError`. They now re-raise.
