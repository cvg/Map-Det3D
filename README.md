<div align="center">

# Map-Det3D: Metric Feed-Forward 3D Reconstruction Prior for Multi-view 3D Object Detection from Streaming Inputs

<a href="https://arxiv.org/abs/2608.12179"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://royyang0714.github.io/Map-Det3D'><img src='https://img.shields.io/badge/Project-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/RoyYang0714/Map-Det3D'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href='https://huggingface.co/RoyYang0714/Map-Det3D'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

</div>

<div>
  <img src="assets/overview.png" width="100%" alt="Banner 2" align="center">
</div>

<div>
  <p></p>
</div>

> [**Map-Det3D: Metric Feed-Forward 3D Reconstruction Prior for Multi-view 3D Object Detection from Streaming Inputs**](https://royyang0714.github.io/Map-Det3D) \
> Yung-Hsu Yang, Luigi Piccinelli, Samuel Rota Bulò, Sunghwan Hong, Denis Rozumny, Johannes Schönberger, Zuria Bauer, Hermann Blum, Peter Kontschieder, and Marc Pollefeys \
> ECCV 2026,
> *Paper at [arXiv 2608.12179](https://arxiv.org/pdf/2608.12179)*


## News and ToDo

- [x] `17.08.2026`: Release code and models.
- [x] `18.06.2026`: Map-Det3D is accepted at ECCV 2026!

## Getting Started

Try our [HuggingFace Demo](https://huggingface.co/spaces/RoyYang0714/Map-Det3D) without installation and try your own data directly!

### Installation

We support Python 3.11+ and PyTorch 2.8.0+.
Please install the correct PyTorch version according to your own hardware settings.

```bash
conda create -n mapdet3d python=3.11 -y

conda activate mapdet3d

# Install PyTorch
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install CUDA ops
pip install git+https://github.com/SysCV/vis4d_cuda_ops.git --no-build-isolation --no-cache-dir

# Install Map-Det3D
pip install -v -e .
```

### Model

We host our model on [HuggingFace](https://huggingface.co/RoyYang0714/Map-Det3D) and provide the [`demo.py`](./scripts/demo.py) as the example.

```python
import torch

from mapdet3d.model.mapdet3d import MapDet3D
from mapdet3d.op.mapdet3d.head import RoI2Det

device = "cuda" if torch.cuda.is_available() else "cpu"

# TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("highest")

# Init model
model = MapDet3D.from_pretrained("RoyYang0714/Map-Det3D").to(device)

# (Optional) Enable tracking
model.track_whole_scene = True
model.roi2det = RoI2Det(nms=True, score_threshold=0.25, iou_threshold=0.5)

# Inference
model.eval()

with torch.no_grad():

with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
    predictions: MapDet3DOut = model(
        images=[image],
        intrinsics=[intrinsics],
        extrinsics=[extrinsics],
        frame_ids=[frame_id],
    )
```

The model weight can also be download as [map-det3d-ca1m.pt](https://huggingface.co/RoyYang0714/map-det3d-ca1m).

### Data

#### CA-1M

We use [CA-1M](https://github.com/apple/ml-cubifyanything) as the training and in-domain testing sets.

1. Download the `train.txt` and `val.txt` from [here](https://github.com/apple/ml-cubifyanything/tree/main/data), and put them under `data/CA1M`

2. Use the provided script to download the data and unzip them.

```bash
python scripts/ca1m/download.py --split train
python scripts/ca1m/download.py --split val
```

It will download the full CA1M train and val data under `data/CA1M`.

3. Convert the dataset for training and testing.

```bash
python scripts/ca1m/convert.py --split train
python scripts/ca1m/convert.py --split val
```

It will parse the dataset and save the cached files and HDF5 under `data/ca1m`.

4. (Optional) Mesh file.

The mesh is for the visualization purpose. You can download CA-1M mesh from [BoxFusion](https://huggingface.co/datasets/Kevin1804/BoxFusion/tree/main) or just generate them with `open3d`.

The final data structure should be like this:

```bash
REPO_ROOT
├── data
│   ├── CA1M
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── train
│   │   │   ├── $SEQ_NAME
│   │   │   ├── ...
│   │   └── val
│   └── ca1m
│       ├── cache
│       ├── mesh
│       │   ├── $SEQ_NAME
│       │   │   └── mesh.ply
│       │   ├── ...
│       ├── train
│       │   ├── $SEQ_NAME.hdf5
│       │   ├── ...
│       └── val
├── ...
```

#### ScanNet

We follow [BoxFusion](https://github.com/lanlan96/BoxFusion) and use ScanNet as the out-of-domain testing sets.

1. Download [ScanNet](https://github.com/ScanNet/ScanNet) and use the [script](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) to extract the images, depth, intrinsics, and poses according to the [val.txt](./data/scannet/meta_data/val.txt):

```bash
REPO_ROOT
├── data
│   ├── scannet
│   │   ├── data
│   │   │   └── $SEQ_NAME
│   │   │   │   ├── frames
│   │   │   │   │   ├── colors
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   └── meta_data
├── ...
```

2. Extract annotations.

```bash
python scripts/scannet/batch_load_scannet_data.py
python scripts/scannet/batch_load_scannet_data.py --scannet200
```

3. Convert the data for testing.

```bash
python scripts/scannet/convert.py
python scripts/scannet/convert.py --scannet200
```

The final data structure should be like this:

```bash
REPO_ROOT
├── data
│   ├── scannet
│   │   ├── cache
│   │   ├── data
│   │   ├── meta_data
│   │   ├── scannet_instance_data
│   │   └── scannet200_instance_data
├── ...
```

### Training

```bash
# 2 nodes and 8 gpus each node 
mapdet3d fit --config mapdet3d/zoo/mapdet3d/mapdet3d_ca1m.py --gpus 8 --nodes 2
```

The output will be dumped under `./work_dir/${experiment_name}/${version}`.

You can also enable wandb logging by adding `--wandb`.

### Testing

```bash
# CA1M
mapdet3d test --config mapdet3d/zoo/mapdet3d/mapdet3d_ca1m.py --gpus 1 --ckpt https://huggingface.co/RoyYang0714/map-det3d-ca1m/resolve/main/map-det3d-ca1m.pt

# ScanNet200
mapdet3d test --config mapdet3d/zoo/mapdet3d/mapdet3d_scannet200.py --gpus 1 --ckpt https://huggingface.co/RoyYang0714/map-det3d-ca1m/resolve/main/map-det3d-ca1m.pt

# Tracking
mapdet3d test --config mapdet3d/zoo/mapdet3d/mapdet3d_track_scannet.py --gpus 1 --ckpt https://huggingface.co/RoyYang0714/map-det3d-ca1m/resolve/main/map-det3d-ca1m.pt
```

We provide the rerun visualization.
Enable it with `--vis` flag, and the `.rrd` file will be saved under `./work_dir/${experiment_name}/${version}/rerun_vis` folder.

## Citation

If you find our work useful in your research, please consider citing our publications:
```bibtex
@misc{yang2026mapdet3dmetricfeedforward3d,
      title={Map-Det3D: Metric Feed-Forward 3D Reconstruction Prior for Multi-view 3D Object Detection from Streaming Inputs}, 
      author={Yung-Hsu Yang and Luigi Piccinelli and Samuel Rota Bulò and Sunghwan Hong and Denis Rozumny and Johannes Schönberger and Zuria Bauer and Hermann Blum and Peter Kontschieder and Marc Pollefeys},
      year={2026},
      eprint={2608.12179},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.12179}, 
}
```

## Acknowledgements

This project builds upon [Vis4D](https://github.com/SysCV/vis4d), [3D-MOOD](https://github.com/cvg/3D-MOOD), [BoxFusion](https://github.com/lanlan96/BoxFusion), and [MapAnything](https://map-anything.github.io/).
We thank the authors of these projects for making their code available.
