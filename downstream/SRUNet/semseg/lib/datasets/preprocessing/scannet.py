# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
# from lib.pc_utils import read_plyfile, save_point_cloud
from concurrent.futures import ProcessPoolExecutor
SCANNET_RAW_PATH = Path('/lustre/scratch/client/vinai/users/bachtx12/ScanNetv2/')
SCANNET_OUT_PATH = Path('/lustre/scratch/client/vinai/users/bachtx12/scannet_pointcontrast/')
TRAIN_DEST = 'train'
TEST_DEST = 'test'
SUBSETS = {TRAIN_DEST: 'scans', TEST_DEST: 'scans_test'}
POINTCLOUD_FILE = '_vh_clean_2.ply'
BUGS = {
    'train/scene0270_00.ply': 50,
    'train/scene0270_02.ply': 50,
    'train/scene0384_00.ply': 149,
}
print('start preprocess')
# Preprocess data.

def read_plyfile(filepath):
  """Read ply file and return it as numpy array. Returns None if emtpy."""
  with open(filepath, 'rb') as f:
    plydata = PlyData.read(f)
  if plydata.elements:
    return pd.DataFrame(plydata.elements[0].data).values


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
  """Save an RGB point cloud as a PLY file.

  Args:
    points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
        the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
  """
  assert points_3d.ndim == 2
  if with_label:
    assert points_3d.shape[1] == 7
    python_types = (float, float, float, int, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1'), ('label', 'u1')]
  else:
    if points_3d.shape[1] == 3:
      gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
      points_3d = np.hstack((points_3d, gray_concat))
    assert points_3d.shape[1] == 6
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
  if binary is True:
    # Format into NumPy structured array
    vertices = []
    for row_idx in range(points_3d.shape[0]):
      cur_point = points_3d[row_idx]
      vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(vertices_array, 'vertex')

    # Write
    PlyData([el]).write(filename)
  else:
    # PlyData([el], text=True).write(filename)
    with open(filename, 'w') as f:
      f.write('ply\n'
              'format ascii 1.0\n'
              'element vertex %d\n'
              'property float x\n'
              'property float y\n'
              'property float z\n'
              'property uchar red\n'
              'property uchar green\n'
              'property uchar blue\n'
              'property uchar alpha\n'
              'end_header\n' % points_3d.shape[0])
      for row_idx in range(points_3d.shape[0]):
        X, Y, Z, R, G, B = points_3d[row_idx]
        f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
  if verbose is True:
    print('Saved point cloud to: %s' % filename)
def handle_process(path):
  f = Path(path.split(',')[0])
  phase_out_path = Path(path.split(',')[1])
  pointcloud = read_plyfile(f)
  # Make sure alpha value is meaningless.
  assert np.unique(pointcloud[:, -1]).size == 1
  # Load label file.
  label_f = f.parent / (f.stem + '.labels' + f.suffix)
  if label_f.is_file():
    label = read_plyfile(label_f)
    # Sanity check that the pointcloud and its label has same vertices.
    assert pointcloud.shape[0] == label.shape[0]
    assert np.allclose(pointcloud[:, :3], label[:, :3])
  else:  # Label may not exist in test case.
    label = np.zeros_like(pointcloud)
  out_f = phase_out_path / (f.name[:-len(POINTCLOUD_FILE)] + f.suffix)
  processed = np.hstack((pointcloud[:, :6], np.array([label[:, -1]]).T))
  save_point_cloud(processed, out_f, with_label=True, verbose=False)


path_list = []
for out_path, in_path in SUBSETS.items():
  phase_out_path = SCANNET_OUT_PATH / out_path
  phase_out_path.mkdir(parents=True, exist_ok=True)
  for f in (SCANNET_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE):
    path_list.append(str(f) + ',' + str(phase_out_path))

pool = ProcessPoolExecutor(max_workers=20)
result = list(pool.map(handle_process, path_list))

# Fix bug in the data.
for files, bug_index in BUGS.items():
  print(files)

  for f in SCANNET_OUT_PATH.glob(files):
    pointcloud = read_plyfile(f)
    bug_mask = pointcloud[:, -1] == bug_index
    print(f'Fixing {f} bugged label {bug_index} x {bug_mask.sum()}')
    pointcloud[bug_mask, -1] = 0
    save_point_cloud(pointcloud, f, with_label=True, verbose=False)
