#### Table of contents
1. [Datasets](#Datasets)
2. [Experiments](#Experiments)

## Datasets
For [S3DIS](http://buildingparser.stanford.edu/dataset.html), [ScanNet](http://www.scan-net.org/) and [SUN RGB-D](https://rgbd.cs.princeton.edu/) data processing, we follow the instruction of [Pointcontrast](https://github.com/facebookresearch/PointContrast.git)
## Experiments
### Classification

Run the following command to evaluate pre-trained models with different backbones.

```bash
cd downstream/DgCnn # to evaluate with DgCnn 
python train_cls.py \
--log_dir path_to_results_folder \
--dataset_type modelnet40 \ # type of datase such as modelnet40, scanobjectnn
--dataset_path path_to_folder_dataset \
--model_path path_to_pre_trained_model \ # path to pre_trained models

cd downstream/PointNet #  to evaluate with PointNet
python train_cls.py \
--log_dir path_to_results_folder \
--dataset_type modelnet40 \ # type of datase such as modelnet40, scanobjectnn
--dataset_path path_to_folder_dataset \
--model_path path_to_pre_trained_model \ # path to pre_trained models
--data_aug
```

### Part segmentation

Run the following command to evaluate pre-trained models with different backbones.

```bash
cd downstream/DgCnn # to evaluate with DgCnn 
python train_part_segmentation.py \
--log_dir path_to_results_folder \
--dataset_path path_to_folder_dataset \
--model_path path_to_pre_trained_model \ # path to pre_trained models

cd downstream/PointNet #  to evaluate with PointNet
python train_part_segmentation.py \
--log_dir path_to_results_folder \
--dataset_path path_to_folder_dataset \
--model_path path_to_pre_trained_model \ # path to pre_trained models
```

### Semantic segmentation

Run the following command to evaluate pre-trained models with different backbones.

```bash
cd downstream/DgCnn # to evaluate with DgCnn 
python train_segmentation.py \
--log_dir path_to_results_folder \
--dataset_path path_to_folder_dataset \
--model_path path_to_pre_trained_model \ # path to pre_trained models
--test_area 6 \ # test area id, 1,2,3,4,5,6

cd downstream/PointNet #  to evaluate with PointNet
python train_segmentation.py \
--log_dir path_to_results_folder \
--dataset_path path_to_folder_dataset \
--model_path path_to_pre_trained_model \ # path to pre_trained models
--test_area 6 \ # test area id, 1,2,3,4,5,6

cd downstream/SRUNet/semseg
# setting for ScanNet
DATAPATH=path_to_folder_dataset 
PRETRAIN=path_to_pre_trained_model
MODEL=Res16UNet34C
BATCH_SIZE=${BATCH_SIZE:-12}
LOG_DIR=path_to_results_folder

python ddp_main.py \
    train.train_phase=train \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=1 \
    train.val_freq=500 \
    train.save_freq=500 \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    augmentation.normalize_color=True \
    data.dataset=ScannetVoxelization2cmDataset \
    data.batch_size=$BATCH_SIZE \
    data.num_workers=1 \
    data.num_val_workers=1 \
    data.scannet_path=${DATAPATH} \
    data.return_transformation=False \
    test.test_original_pointcloud=False \
    test.save_prediction=False \
    optimizer.lr=0.8 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=20000 \
    misc.log_dir=${LOG_DIR} \
    distributed=local \
    distributed.distributed_world_size=4\
    net.weights=${PRETRAIN} \

# setting for S3DIS
DATAPATH=path_to_folder_dataset 
PRETRAIN=path_to_pre_trained_model
MODEL=Res16UNet34C
BATCH_SIZE=${BATCH_SIZE:-18}
LOG_DIR=LOG_DIR=path_to_results_folder

python ddp_main.py \
    train.train_phase=train \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=1 \
    train.val_freq=200 \
    train.save_freq=100 \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    data.dataset=StanfordArea5Dataset \
    data.batch_size=$BATCH_SIZE \
    data.voxel_size=0.05 \
    data.num_workers=1 \
    data.stanford3d_path=${DATAPATH} \
    augmentation.data_aug_color_trans_ratio=0.05 \
    augmentation.data_aug_color_jitter_std=0.005 \
    optimizer.lr=0.1 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=20000 \
    misc.log_dir=${LOG_DIR} \
    distributed=local \
    distributed.distributed_world_size=1\
    net.weights=$PRETRAIN \
```
### Object dectection

Run the following command to evaluate pre-trained models with SR-UNet backbone.
```bash
cd downstream/SRUNet/votenet_det_new
PRETRAIN=path_to_pre_trained_model
export LOGDIR=path_to_results_folder
mkdir -p $LOGDIR
# command for SUN RGB-D
python ddp_main.py \
  net.is_train=True \
  net.backbone=sparseconv \
  data.dataset=sunrgbd \
  data.num_workers=8 \
  data.batch_size=64 \
  data.no_height=True \
  data.voxelization=True \
  data.voxel_size=0.025 \
  optimizer.learning_rate=0.001 \
  misc.log_dir=$LOGDIR \
  net.weights=$PRETRAIN \
# command for ScanNet
python ddp_main.py \
  net.backbone=sparseconv \
  data.dataset=scannet \
  data.num_workers=8 \
  data.batch_size=32 \
  data.num_points=40000 \
  data.no_height=True \
  optimizer.learning_rate=0.001 \
  data.voxelization=True \
  data.voxel_size=0.025 \
  misc.log_dir=$LOGDIR \
  net.is_train=True \
  net.weights=$PRETRAIN \
```