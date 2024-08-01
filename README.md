# Grounded-SAM-2
Grounded SAM 2: Ground and Track Anything with Grounding DINO and SAM 2


## Contents


## Installation

Since we need the CUDA compilation environment to compile the `Deformable Attention` operator used in Grounding DINO, we need to check whether the CUDA environment variables have been set correctly (which you can refer to [Grounding DINO Installation](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) for more details). You can set the environment variable manually as follows if you want to build a local GPU environment for Grounding DINO to run Grounded SAM 2:

```bash
export CUDA_HOME=/path/to/cuda-12.1/
```

Install `segment-anything-2`:

```bash
pip install -e .
```

Install `grounding dino`:

```bash
pip install --no-build-isolation -e grounding_dino
```
