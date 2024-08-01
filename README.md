# Grounded-SAM-2
Grounded SAM 2: Ground and Track Anything with Grounding DINO, Grounding DINO 1.5 and SAM 2


## Contents
- [Installation](#installation)
- [Grounded-SAM-2 Demo](#grounded-sam-2-demo)
  - [Grounded-SAM-2 Image Demo](#grounded-sam-2-image-demo-with-grounding-dino)
  - [Grounded-SAM-2 Image Demo (with Grounding DINO 1.5)](#grounded-sam-2-image-demo-with-grounding-dino-15--16)

## Installation

Since we need the CUDA compilation environment to compile the `Deformable Attention` operator used in Grounding DINO, we need to check whether the CUDA environment variables have been set correctly (which you can refer to [Grounding DINO Installation](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) for more details). You can set the environment variable manually as follows if you want to build a local GPU environment for Grounding DINO to run Grounded SAM 2:

```bash
export CUDA_HOME=/path/to/cuda-12.1/
```

Install `Segment Anything 2`:

```bash
pip install -e .
```

Install `Grounding DINO`:

```bash
pip install --no-build-isolation -e grounding_dino
```

Downgrade the version of the `supervision` library to `0.6.0` to use its original API for visualization (we will update our code to be compatible with the latest version of `supervision` in the future release):

```bash
pip install supervision==0.6.0
```

Download the pretrained `SAM 2` checkpoints:

```bash
cd checkpoints
bash download_ckpts.sh
```

Download the pretrained `Grounding DINO` checkpoints:

```bash
cd gdino_checkpoints
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

## Grounded-SAM-2 Demo
### Grounded-SAM-2 Image Demo (with Grounding DINO)
Note that `Grounding DINO` has already been supported in [Huggingface](https://huggingface.co/IDEA-Research/grounding-dino-tiny), so we provide two choices for running `Grounded-SAM-2` model:
- Use huggingface API to inference Grounding DINO (which is simple and clear)

```bash
python grounded_sam2_hf_model_demo.py
```

- Load local pretrained Grounding DINO checkpoint and inference with Grounding DINO original API (make sure you've already downloaded the pretrained checkpoint)

```bash
python grounded_sam2_local_demo.py
```

### Grounded-SAM-2 Image Demo (with Grounding DINO 1.5 & 1.6)

We've already released our most capable open-set detection model [Grounding DINO 1.5 & 1.6](https://github.com/IDEA-Research/Grounding-DINO-1.5-API), which can be combined with SAM 2 for stronger open-set detection and segmentation capability. You can apply the API token first and run Grounded-SAM-2 with Grounding DINO 1.5 as follows:

Install the latest DDS cloudapi:

```bash
pip install dds-cloudapi-sdk
```

Apply your API token from our official website here: [request API token](https://deepdataspace.com/request_api).

```bash
python grounded_sam2_gd1.5_demo.py
```