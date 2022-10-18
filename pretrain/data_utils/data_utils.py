import numpy as np


def rotate_point_cloud(batch_data):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
    rotated_data = np.dot(batch_data, rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
    """
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(*batch_data.shape), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def normalize_point_cloud(pc):
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """Rotate the point cloud along up direction with certain angle.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def center_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc


def farthest_pair_sample(point, dict_pair, num_pair=512):
    point_id = list(dict_pair.keys())
    N = len(point_id)
    xyz = point[point_id]
    centroids = np.zeros((num_pair,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    id = 0
    list_id = []
    curr_pair = 0
    while True:
        centroids[id] = farthest
        num_pair_in = len(dict_pair[point_id[farthest]])
        if curr_pair + num_pair_in >= num_pair:
            num_need_pair = num_pair - curr_pair
            list_id += np.random.choice(dict_pair[point_id[farthest]], num_need_pair).tolist()
            break
        list_id += dict_pair[point_id[farthest]]
        curr_pair += num_pair_in

        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
        id += 1
    return list_id


def separate_point_sample(point, dict_pair, num_pair=512):
    point_id = list(dict_pair.keys())
    np.random.shuffle(point_id)
    N = len(point_id)
    xyz = point[point_id]
    if N > num_pair:
        # print('fps')
        centroids = np.zeros((num_pair,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        id = 0
        list_id = []
        curr_pair = 0
        while True:
            if curr_pair >= num_pair:
                return list_id
            centroids[id] = farthest
            list_id.append(dict_pair[point_id[farthest]][0])
            curr_pair += 1

            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
            id += 1
    else:
        curr_pair = 0
        list_id = []
        while True:
            for key in point_id:
                try:
                    list_id.append(dict_pair[key].pop())
                except Exception:
                    continue
                curr_pair += 1
                if curr_pair >= num_pair:
                    return list_id


def farthest_point_sample(point, npoint=1024):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    fo = farthest
    print(fo)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype("float32")
    return translated_pointcloud
