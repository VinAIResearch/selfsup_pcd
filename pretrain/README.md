## <h1 align="center">Pre-training</h1>

#### Table of contents
1. [Datasets](#Datasets)
2. [Pre-training](#Pre-training)

## Datasets
For [ModelNet40](https://modelnet.cs.princeton.edu/), we use [Blender](https://www.blender.org/) with fixed camera parameters to generate multi-view of each object. After that, we produce pixel-point correspondences by projecting sampled points to images based on the camera parameters.

[ScanNet](http://www.scan-net.org/), we use two view images when their corresponding point cloud pairs have at least 30% overlapping. To define pixelpoint pairs, we reconstruct a point cloud from the first depth image in an image pair, then project it to two color images to get pixel-point correspondences.

## Pre-training
Run the following command to pre-train models with different backbones.
```bash
python train.py \
--num_views 12 \ # number of view used for each object
--num_point_contrast 512 \ # number of positive pair for contrastive loss
--num_points 1024 \ # number of points for each object
--dataset path_to_folder_dataset \
--log_dir path_to_result_model \
--path_model path_to_pre_trained_2d_image \
--model pointnet \ # pre-training backbone pointnet or dgcnn
```


