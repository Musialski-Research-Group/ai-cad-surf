# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import open3d as o3d

from datasets.sampling import POINT_SAMPLING_STATEGIES


class ReconstructionDataset(data.Dataset):
    """Generates clean and noisy point clouds sampled  + samples on a grid with their distance to the surface"""
    def __init__(self, file_path, config):

        self.config = config
        self.o3d_point_cloud = o3d.io.read_point_cloud(file_path)

        # extract center and scale points and normals
        self.points, self.mnfld_n = self.get_mnfld_points()
        self.bbox = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)]).transpose()
        self.nonmnfld_points = self.get_nonmnfld_points()
        self.sample_gaussian_noise_around_shape()
        
    def get_mnfld_points(self):
        """Get points on the manifold"""

        points = np.asarray(self.o3d_point_cloud.points, dtype=np.float32)
        normals = np.asarray(self.o3d_point_cloud.normals, dtype=np.float32)
        if normals.shape[0] == 0:
            normals = np.zeros_like(points)

        # center and scale point cloud
        self.cp = points.mean(axis=0)
        points = points - self.cp[None, :]
        self.scale = np.abs(points).max()
        points = points / self.scale

        return points, normals
    

    def sample_gaussian_noise_around_shape(self):
        kd_tree = spatial.KDTree(self.points)
        # query each point for sigma
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]
        self.sigmas = sigmas
        return
    

    def get_nonmnfld_points(self):
        nonmnfld_point = POINT_SAMPLING_STATEGIES.get(self.config.non_manifold_sampling, None)
        if not nonmnfld_point:
            raise NotImplementedError("Unsupported non manfold sampling type")
        return nonmnfld_point


    def __getitem__(self, index):
        manifold_idxes_permutation = np.random.permutation(self.points.shape[0])
        mnfld_idx = manifold_idxes_permutation[:self.config.n_points]
        manifold_points = self.points[mnfld_idx]
        manifold_normals = self.mnfld_n[mnfld_idx]

        nonmnfld_points = np.random.uniform(-self.config.grid_range, self.config.grid_range,
                            size=(self.config.n_points, 3)).astype(np.float32)

        near_points = (manifold_points + self.sigmas[mnfld_idx] * np.random.randn(manifold_points.shape[0],
                                                                                  manifold_points.shape[1])).astype(np.float32)
        
        return {
            'points': manifold_points,
            'mnfld_n': manifold_normals,
            'nonmnfld_points': nonmnfld_points,
            'near_points': near_points,
            }
    
    def __len__(self):
        return self.config.n_samples
