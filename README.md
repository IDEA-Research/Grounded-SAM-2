# 🌊 SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

[[Arxiv]](https://arxiv.org/abs/2411.11922) [[Project Page]](https://yangchris11.github.io/samurai/) 

This repository is the official implementation of SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

https://github.com/user-attachments/assets/9d368ca7-2e9b-4fed-9da0-d2efbf620d88

## Getting Started

#### SAMURAI Installation 

SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://github.com/facebookresearch/sam2?tab=readme-ov-file) to install both PyTorch and TorchVision dependencies. You can install **the SAMURAI version** of SAM 2 on a GPU machine using:
```
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```

Please see [INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) from the original SAM 2 repository for FAQs on potential issues and solutions.
```
pip install requirements.txt
```

#### SAM 2.1 Checkpoint Download

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

#### Data Preparation

Please prepare the data in the following format:
```
data/LaSOT
├── airplane/
│   ├── airplane-1/
│   │   ├── full_occlusion.txt
│   │   ├── groundtruth.txt
│   │   ├── img
│   │   ├── nlp.txt
│   │   └── out_of_view.txt
│   ├── airplane-2/
│   ├── airplane-3/
│   ├── ...
├── basketball
├── bear
├── bicycle
...
├── training_set.txt
└── testing_set.txt
```

#### Main Inference
```
python scripts/main_inference.py 
```

## Acknowledgment

SAMURAI is built on top of [SAM 2](https://github.com/facebookresearch/sam2?tab=readme-ov-file) by Meta FAIR.

The VOT evaluation code is modifed from [VOT Toolkit](https://github.com/votchallenge/toolkit) by Luka Čehovin Zajc.

## Citation

Please consider citing our paper and the wonderful `SAM 2` if you found our work interesting and useful.
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

@misc{yang2024samurai,
      title={SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory}, 
      author={Cheng-Yen Yang and Hsiang-Wei Huang and Wenhao Chai and Zhongyu Jiang and Jenq-Neng Hwang},
      year={2024},
      eprint={2411.11922},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.11922}, 
}
```
