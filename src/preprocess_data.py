import os
import sys

import tqdm
import open3d as o3d
import trimesh

from utils.multiproc import start_process_pool
from utils.evaluation import normalize_mesh_export 


def data_no_filter(mesh_path, input_path, sample_pt_num=10000):
    print(mesh_path)

    try:
        mesh = trimesh.load_mesh(mesh_path, process=False, force='mesh')
        mesh = normalize_mesh_export(mesh)
        # unit to [-0.5, 0.5]
        pts, idx = trimesh.sample.sample_surface(mesh, count=sample_pt_num)
        normals = mesh.face_normals[idx]
        pts_o3d = o3d.geometry.PointCloud()
        pts_o3d.points = o3d.utility.Vector3dVector(pts)
        pts_o3d.normals = o3d.utility.Vector3dVector(normals)

        f_name = os.path.splitext(mesh_path.split('/')[-1])[0]
        o3d.io.write_point_cloud(os.path.join(input_path, f_name + '.ply'), pts_o3d)

        return
    except Exception as e:
        print(e)
        print('error', mesh_path)


def preprocess_data(gt_path: str, input_path: str, num_processes:int = 16, sample_pt_num:int = 1000):
    os.makedirs(input_path, exist_ok=True)

    call_params = list()
    test=os.listdir(gt_path)
    for f in tqdm.tqdm(sorted(os.listdir(gt_path))):
        if os.path.splitext(f)[1] not in ['.ply', '.obj', '.off','xyz']:
            continue

        mesh_path = os.path.join(gt_path, f)
        call_params.append((mesh_path, input_path, sample_pt_num))

    start_process_pool(data_no_filter, call_params, num_processes)


def main(argv):
    del argv
    unload_results(FLAGS.model, FLAGS.output)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('gt', None, 'Path to the gt mesh file.')
    flags.DEFINE_string('input', None, 'Path to the input ply file.')
    flags.mark_flags_as_required(['model', 'output'])
    app.run(main)
