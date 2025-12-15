# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import io
import hashlib
import random
import json
from PIL import Image
from omegaconf import OmegaConf
import yaml

from plyfile import PlyData
import trimesh

import torch
import torch.backends.cudnn as cudnn
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def backup_code(logdir, dir_list=[], file_list=[]):
    """backup models saving"""
    code_bkp_dir = os.path.join(logdir, 'code_bkp')
    os.makedirs(code_bkp_dir, exist_ok=True)
    for dir_name in dir_list:
        print("copying directory {} to {}".format(dir_name, code_bkp_dir))
        os.system('cp -r %s %s' % (dir_name, code_bkp_dir))  # backup the current model code
    for file_name in file_list:
        print("copying file {} to {}".format(file_name, code_bkp_dir))
        os.system('cp %s %s' % (file_name, code_bkp_dir))


def get_cuda_ifavailable(torch_obj, device=None):
    """get a cuda obeject if cuda is available"""
    if (torch.cuda.is_available()):
        return torch_obj.cuda(device=device)
    else:
        return torch_obj


def plotly_fig2array(fig):
    """convert Plotly fig to  an array"""
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf).convert('RGB')
    img = np.asarray(img)
    return img


def read_vnf_ply(filename):
    """read vertices and normal vectors from a ply file"""
    plydata = PlyData.read(filename)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    nx = np.asarray(plydata.elements[0].data['nx'])
    ny = np.asarray(plydata.elements[0].data['ny'])
    nz = np.asarray(plydata.elements[0].data['nz'])
    return np.stack([x, y, z], axis=1), np.stack([nx, ny, nz], axis=1), plydata['face'].data


def load_reconstruction_data(file_path, n_points=30000, sample_type='vertices'):
    extension = file_path.split('.')[-1]
    if extension == 'xyz':
        points = np.loadtxt(file_path)
    elif extension == 'ply':
        mesh = trimesh.load_mesh(file_path)

        if hasattr(mesh, 'faces') and not sample_type == 'vertices':
            # sample points if its a triangle mesh
            points = trimesh.sample.sample_surface(mesh, n_points)[0]
        else:
            # use the vertices if its a point cloud
            points = mesh.vertices
    # Center and scale points
    # cp = points.mean(axis=0)
    # points = points - cp[None, :]
    # scale = np.abs(points).max()
    # points = points / scale
    return np.array(points).astype('float32')


def convert_xyz_to_ply_with_noise(file_path, noise=None):
    """convert ply file in xyznxnynz format to ply file"""
    points = np.loadtxt(file_path)
    if noise is None:
        mesh = trimesh.Trimesh(points[:, :3], [], vertex_normals=points[:, 3:])
        mesh.export(file_path.split('.')[0] + '.ply', vertex_normal=True)
    else:
        for std in noise:
            bbox_scale = np.abs(points).max()
            var = std*std
            cov_mat = bbox_scale * np.array([[var, 0., 0.], [0., var, 0.], [0., 0., var]])
            noise = np.random.multivariate_normal([0., 0., 0.], cov_mat, size=points.shape[0], check_valid='warn', tol=1e-8)
            mesh = trimesh.Trimesh(points[:, :3] + noise, [], vertex_normals=points[:, 3:])
            mesh.export(file_path.split('.')[0] + '_' +str(std) + '.ply', vertex_normal=True)


def count_parameters(model):
    """count the number of parameters in a given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def load_config_from_path(config_path):
#     spec = importlib.util.spec_from_file_location("config_module", config_path)
#     config_module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(config_module)
    
#     # Assuming the config file defines a function get_config()
#     config_data = config_module.get_config()
    
#     # Convert to ConfigDict
#     return ConfigDict(config_data)


def hash_config(config):
    config_str = json.dumps(config.to_dict(), sort_keys=True, separators=(',', ':'))
    return hashlib.md5(config_str.encode()).hexdigest()


def hash_to_nbits(hash_, num_bits):
    full_hash_int = int(hash_, 16)

    max_value = (1 << num_bits) - 1
    shortened_hash = full_hash_int & max_value

    num_hex_digits = num_bits // 4
    return f"{shortened_hash:0{num_hex_digits}x}"


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def get_data_paths(config, path='data/',cat_name=None):
    # Setups dataset model paths
    assert not OmegaConf.is_missing(config.data, 'dataset')
    assert config.data.dataset == 'file_path'
    assert not OmegaConf.is_missing(config.data, 'file_path')
    assert not OmegaConf.is_missing(config.data, 'gt_path')
    point_cloud_path = f'{config.data.file_path}'
    mesh_path = f'{config.data.gt_path}'
    data_path = point_cloud_path
    gt_path = mesh_path
    return data_path, gt_path

