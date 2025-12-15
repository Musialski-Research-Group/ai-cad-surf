import numpy as np


def grid_point_sampling(config):
    x, y, z = np.linspace(-config.grid_range, config.grid_range, config.grid_res).astype(np.float32), \
                np.linspace(-config.grid_range, config.grid_range, config.grid_res).astype(np.float32),\
                np.linspace(-config.grid_range, config.grid_range, config.grid_res).astype(np.float32)
    xx, yy, zz = np.meshgrid(x, y, z)
    xx, yy, zz = xx.ravel(), yy.ravel(), zz.ravel()
    return np.stack([xx, yy, zz], axis=1).astype('f')


def uniform_point_sampling(config):
    nonmnfld_points = np.random.uniform(
        -config.grid_range, config.grid_range,
        size=(config.grid_res * config.grid_res * config.grid_res, 3)).astype(np.float32)
    return nonmnfld_points


def gaussian_noisy_points_sampling(config):
    n_noisy_points = int(np.round(config.grid_res * config.grid_res / config.n_points))
    noise = np.random.multivariate_normal(
        [0, 0, 0],[
        [config.sampling_std, 0, 0],
        [0, config.sampling_std, 0],
        [0, 0, config.sampling_std]],
        size=(
            config.points.shape[0],
            n_noisy_points)
        ).astype(np.float32)
    nonmnfld_points = np.tile(config.points[:, None, :], [1, n_noisy_points, 1]) + noise
    nonmnfld_points = nonmnfld_points.reshape([nonmnfld_points.shape[0], -1, nonmnfld_points.shape[-1]])
    return nonmnfld_points


def gaussian_point_sampling(config):
    nonmnfld_points = gaussian_noisy_points_sampling()
    idx = np.random.choice(range(nonmnfld_points.shape[1]), config.grid_res * config.grid_res)
    sample_idx = np.random.choice(range(nonmnfld_points.shape[0]), config.grid_res * config.grid_res)
    nonmnfld_points = nonmnfld_points[sample_idx, idx]
    return nonmnfld_points
    

def combined_point_sampling(config):
    nonmnfld_points1 = gaussian_noisy_points_sampling()
    nonmnfld_points2 = config.grid_points
    idx1 = np.random.choice(range(nonmnfld_points1.shape[1]), int(np.ceil(config.grid_res * config.grid_res / 2)))
    idx2 = np.random.choice(range(nonmnfld_points2.shape[0]), int(np.floor(config.grid_res * config.grid_res / 2)))
    sample_idx = np.random.choice(range(nonmnfld_points1.shape[0]), int(np.ceil(config.grid_res * config.grid_res / 2)))

    nonmnfld_points = np.concatenate([nonmnfld_points1[sample_idx, idx1], nonmnfld_points2[idx2]], axis=0)
    return nonmnfld_points


POINT_SAMPLING_STATEGIES = {
    'grid': grid_point_sampling,
    'uniform': uniform_point_sampling,
    'gaussian': gaussian_point_sampling,
    'combined': combined_point_sampling
}
