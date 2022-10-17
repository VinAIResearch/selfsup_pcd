# selfsup_pcd
##### Table of contents
1. [Getting Started](#Getting-Started)
2. [Experiments](#Experiments)
3. [Acknowledgments](#Acknowledgments)
4. [Contacts](#Contacts)

# Self-Supervised Learning with Multi-View Rendering for 3D Point Cloud Analysis (ACCV 2022)

[Bach Tran](https://bachtranxuan.github.io/),
[Binh-Son Hua](https://sonhua.github.io/),
[Anh Tuan Tran](https://sites.google.com/site/anhttranusc/),
[Minh Hoai](https://www3.cs.stonybrook.edu/~minhhoai/)<br>
VinAI Research, Vietnam

> **Abstract:** 
Recently, great progress has been made in 3D deep learning with the emergence of deep neural networks specifically designed for 3D point clouds. These networks are often trained from scratch or from pre-trained models learned purely from point cloud data. Inspired by the success of deep learning in the image domain, we devise a novel pre-training technique for better model initialization by utilizing the multi-view rendering of the 3D data. Our pre-training is self-supervised by a local pixel/point level correspondence loss computed from perspective projection and a global image/point cloud level loss based on knowledge distillation, thus effectively improving upon popular point cloud networks, including PointNet, DGCNN and SR-UNet. 
These improved models outperform existing state-of-the-art methods on various datasets and downstream tasks. We also analyze the benefits of synthetic and real data for pre-training, and observe that pre-training on synthetic data is also useful for high-level downstream tasks.

```bibtex
@inproceedings{dinh2021hyperinverter,
    title={Self-Supervised Learning with Multi-View Rendering for 3D Point Cloud Analysis},
    author={Bach Tran and Binh-Son Hua and Anh Tuan Tran and Minh Hoai},
    booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
    year={2022}
}
```
**Please CITE** our paper whenever our model implementation is used to help produce published results or incorporated into other software.

## Getting Started
The codebase is tested on
- Ubuntu
- CUDA 11.0

### Installation

- Clone this repo:
``` 
git clone https://github.com/VinAIResearch/selfsup_pcd.git
cd selfsup_pcd
```

- Install dependencies:
```
conda env create -f environment.yml
```

### Datasets
- **Synthetic data**: we evaluate our pre-trained model on two synthetic datasets that include [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) for classification task and [ShapeNetPart](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html) for part segmentation task with official train and test sets.

- **Real data**: We also evalutate our pre-trained model on real dataset. Particularly, we use [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/) with two variants (without and with background) for classification task, [S3DIS](http://buildingparser.stanford.edu/dataset.html) and [ScanNet](http://www.scan-net.org/) for semantic segmentation task, and [ScanNet](http://www.scan-net.org/) and [SUN RGB-D](https://rgbd.cs.princeton.edu/) for object detection task.

## Experiments
### Pre-trained Models.
We also provide official [pre-trained models](https://drive.google.com/drive/folders/11796nNYvQ77XdFwdEXbIX0IZltnXHn4Z?usp=sharing).


### Pre-training
Please follow the [instruction](./pretrain/README.md).

### Downstream tasks
Please follow the [instruction](./downstream/README.md).

## Acknowledgments
Our source code is developed based on the below codebase:
- [DGCNN](https://github.com/antao97/dgcnn.pytorch.git)
- [PointNet](https://github.com/fxia22/pointnet.pytorch.git)
- [SR-UNet](https://github.com/facebookresearch/PointContrast.git)

Overall, thank you so much.
## Contacts
If you have any questions, please drop an email to _tranxuanbach1412@gmail.com_ or open an issue in this repository.