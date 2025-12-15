import torch
import numpy as np
from sklearn import neighbors
import trimesh
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree
import logging
import mcubes

from utils.setup import get_cuda_ifavailable
from utils.math import get_3d_grid

logger = logging.getLogger('flatcad')


def implicit2mesh(decoder, mods, grid_res, translate=[0., 0., 0.], 
                  scale=1, get_mesh=True, device=None, 
                  bbox=np.array([[-1, 1], [-1, 1], [-1, 1]]), 
                  feat=None, 
                  hash_tree=None, 
                  chunk_size=10000):

    # compute a mesh from the implicit representation in the decoder.
    # Uses marching cubes.
    # reimplemented from SAL get surface trace function : https://github.com/matanatz/SAL/blob/master/code/utils/plots.py
    logger.info('[Implicit2Mesh]')
    logger.info('Neural-SDF Inference')
    logger.info(f'Grid Resolution: {grid_res}')
    logger.info(f'Translation: {translate}')
    logger.info(f'Scale: {scale}')
    logger.info(f'Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}]')
    mesh = None
    grid_dict = get_3d_grid(resolution=grid_res, bbox=bbox, device=device)
    logger.info('Finished getting grid_dict')
    cell_width = grid_dict['xyz'][0][2] - grid_dict['xyz'][0][1]
    pnts = grid_dict["grid_points"]

    z = []
    # for point in tqdm(torch.split(pnts, 10000, dim=0)):
    for point in tqdm(torch.split(pnts, chunk_size, dim=0)):
        # point: (100000, 3)
        point = get_cuda_ifavailable(point, device=device)
        if feat is not None:
            if point.dim() == 2:
                point = point.unsqueeze(0)
            query_feat = decoder.encoder.query_feature(feat, point)
            z.append(decoder.decoder(point, query_feat).detach().squeeze(0).cpu().numpy())
        elif hash_tree is not None:
            if point.dim() == 2:
                point = point.unsqueeze(0)
            # query_feat = decoder.encoder.query_feature(feat, point)
            query_feat = decoder.encoder.query_feature(hash_tree, feat, point)
            z.append(decoder.decoder(point, query_feat).detach().squeeze(0).cpu().numpy())
        else:
            z.append(decoder(point.type(torch.float32), mods).detach().squeeze(0).cpu().numpy())
    z = np.concatenate(z, axis=0).reshape(grid_dict['xyz'][1].shape[0], grid_dict['xyz'][0].shape[0],
                                          grid_dict['xyz'][2].shape[0]).transpose([1, 0, 2]).astype(np.float64)

    logger.info(f'Z axis min: {z.min()}')
    logger.info(f'Z axis max: {z.max()}')

    threshs = [0.00]
    mesh_list = list()
    for thresh in threshs:
        if (np.sum(z > 0.0) < np.sum(z < 0.0)):
            thresh = -thresh
        # verts, faces, normals, values = measure.marching_cubes(volume=z, level=thresh,
        #                                                        spacing=(cell_width, cell_width, cell_width),
        #                                                        method='lewiner')  # method:'lewiner' or 'lorensen'
        verts, faces = mcubes.marching_cubes(z, 0)

        verts = verts*float(cell_width) + np.array([grid_dict['xyz'][0][0], grid_dict['xyz'][1][0], grid_dict['xyz'][2][0]])    #this is the one fixed by shen
        #verts = verts + np.array([grid_dict['xyz'][0][0], grid_dict['xyz'][1][0], grid_dict['xyz'][2][0]])
        verts = verts * (1 / scale) - translate

        # print(verts)
        # print(faces)
        if get_mesh:
            # mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values, validate=True)
            mesh = trimesh.Trimesh(verts, faces, validate=True)
        mesh_list.append(mesh)
    return mesh_list[0]


def recon_metrics(
        pc1, pc2, one_sided=False,
        alphas=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
        percentiles=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
        k=[10, 25, 50], return_all=False
        ):
    """
    Compute reconstruction benchmarc evaluation metrics:
     - chamfer and hausdorff distance metrics between two point clouds pc1 and pc2 [nx3]
     - percentage of distance points metric (not used in the paper)
     - meal local chamfer variance
    pc1 is the reconstruction and pc2 is the gt data
    """
    scale = np.abs(pc2).max()

    # compute one side
    pc1_kd_tree = KDTree(pc1)
    one_distances, one_vertex_ids = pc1_kd_tree.query(pc2, n_jobs=4)
    cd12 = np.mean(one_distances)
    hd12 = np.max(one_distances)
    cdmed12 = np.median(one_distances)
    cd21 = None
    hd21 = None
    cdmed21 = None
    cdp2 = None
    chamfer_distance = cd12
    hausdorff_distance = hd12

    # compute chamfer distance percentiles cdp
    cdp1 = np.percentile(one_distances, percentiles, interpolation='lower')

    # compute PoD
    pod1 = []
    for alpha in alphas:
        pod1.append((one_distances < alpha * scale).sum() / one_distances.shape[0])


    if not one_sided:
        # compute second side
        pc2_kd_tree = KDTree(pc2)
        two_distances, two_vertex_ids = pc2_kd_tree.query(pc1, n_jobs=4)
        cd21 = np.mean(two_distances)
        hd21 = np.max(two_distances)
        cdmed21 = np.median(two_distances)
        chamfer_distance = 0.5*(cd12 + cd21)
        hausdorff_distance = np.max((hd12, hd21))
        # compute chamfer distance percentiles cdp
        cdp2 = np.percentile(two_distances, percentiles)
        # compute PoD
        pod2 = []
        for alpha in alphas:
            pod2.append((two_distances < alpha*scale).sum() / two_distances.shape[0])

     # compute double sided pod
    pod12 = []
    for alpha in alphas:
        pod12.append( ((one_distances < alpha * scale).sum() + (two_distances < alpha*scale).sum()) /
                      (one_distances.shape[0] + two_distances.shape[0]))
    cdp12 = np.percentile(np.concatenate([one_distances, two_distances]), percentiles) # compute chamfer distance percentiles cdp

    nn1_dist, local_idx2 = pc1_kd_tree.query(pc1, max(k), n_jobs=-1)
    nn1_dist_2pc2 = two_distances[local_idx2]
    malcv = [(nn1_dist_2pc2[:, :k0]/ nn1_dist.mean(axis=1, keepdims=True)).var(axis=1).mean() for k0 in k]


    if return_all:
        return chamfer_distance, hausdorff_distance, (cd12, cd21, cdmed12, cdmed21, hd12, hd21), (pod1, pod2, pod12), \
            (cdp1.tolist(), cdp2.tolist(), cdp12.tolist()), malcv, (one_distances, two_distances)


    return chamfer_distance, hausdorff_distance, (cd12, cd21, cdmed12, cdmed21, hd12, hd21), (pod1, pod2, pod12), \
           (cdp1.tolist(), cdp2.tolist(), cdp12.tolist()), malcv


def eval_reconstruct_gt(rec_mesh: trimesh.Trimesh, gt_pts: trimesh.Trimesh, is_final_res=False, file_name=None, sample_num=100000):
    def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
        ''' Computes minimal distances of each point in points_src to points_tgt.

        Args:
            points_src (numpy array): source points
            normals_src (numpy array): source normals
            points_tgt (numpy array): target points
            normals_tgt (numpy array): target normals
        '''
        kdtree = KDTree(points_tgt)
        dist, idx = kdtree.query(points_src, workers=-1)  # workers=-1 means all workers will be used

        if normals_src is not None and normals_tgt is not None:
            normals_src = \
                normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
            normals_tgt = \
                normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

            #        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
            #        # Handle normals that point into wrong direction gracefully
            #        # (mostly due to mehtod not caring about this in generation)
            #        normals_dot_product = np.abs(normals_dot_product)

            normals_dot_product = np.abs(normals_tgt[idx] * normals_src)
            normals_dot_product = normals_dot_product.sum(axis=-1)
        else:
            normals_dot_product = np.array(
                [np.nan] * points_src.shape[0], dtype=np.float32)
        return dist, normals_dot_product

    def get_threshold_percentage(dist, thresholds):
        ''' Evaluates a point cloud.
        Args:
            dist (numpy array): calculated distance
            thresholds (numpy array): threshold values for the F-score calculation
        '''
        in_threshold = [
            (dist <= t).mean() for t in thresholds
        ]
        return in_threshold

    def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None, thresholds=[0.005]):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness ** 2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy ** 2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()
        # logger.info(completeness,accuracy,completeness2,accuracy2)
        # Chamfer distance
        logger.info('--calculating the chamferL2--')
        chamferL2 = 0.5 * (completeness2 + accuracy2)

        logger.info('--calculating the normals_correctness--')
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        logger.info('--calculating the F1-Score--')
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]
        return normals_correctness, chamferL1, chamferL2, F[0]

    def get_ecd_ef1(
            pts_rec, pts_gt, normals_rec, normals_gt,
            ef1_radius=0.004, ef1_dotproduct_threshold=0.2,
            ef1_threshold=0.005
            ):

        # sample gt edge points
        gt_tree = neighbors.KDTree(pts_gt)
        indslist = gt_tree.query_radius(pts_gt, ef1_radius)
        flags = np.zeros([len(pts_gt)], bool)
        for p in range(len(pts_gt)):
            inds = indslist[p]
            if len(inds) > 0:
                this_normals = normals_gt[p:p + 1]
                neighbor_normals = normals_gt[inds]
                dotproduct = np.abs(np.sum(this_normals * neighbor_normals, axis=1))
                if np.any(dotproduct < ef1_dotproduct_threshold):
                    flags[p] = True
        gt_edge_points = np.ascontiguousarray(pts_gt[flags])

        # sample pred edge points
        pred_tree = neighbors.KDTree(pts_rec)
        indslist = pred_tree.query_radius(pts_rec, ef1_radius)
        flags = np.zeros([len(pts_rec)], bool)
        for p in range(len(pts_rec)):
            inds = indslist[p]
            if len(inds) > 0:
                this_normals = normals_rec[p:p + 1]
                neighbor_normals = normals_rec[inds]
                dotproduct = np.abs(np.sum(this_normals * neighbor_normals, axis=1))
                if np.any(dotproduct < ef1_dotproduct_threshold):
                    flags[p] = True
        pred_edge_points = np.ascontiguousarray(pts_rec[flags])

        # ecd ef1
        if len(pred_edge_points) == 0:
             pred_edge_points = np.zeros([486, 3], np.float32)
        if len(gt_edge_points) == 0:
            ecd = 0
            ef1 = 1
        else:
            # from gt to pred
            tree = KDTree(pred_edge_points)
            dist, inds = tree.query(gt_edge_points, k=1)
            recall = np.sum(dist < ef1_threshold) / float(len(dist))
            dist = np.square(dist)
            gt2pred_mean_ecd = np.mean(dist)

            # from pred to gt
            tree = KDTree(gt_edge_points)
            dist, inds = tree.query(pred_edge_points, k=1)
            precision = np.sum(dist < ef1_threshold) / float(len(dist))
            dist = np.square(dist)
            pred2gt_mean_ecd = np.mean(dist)

            ecd = gt2pred_mean_ecd + pred2gt_mean_ecd
            if recall + precision > 0:
                ef1 = 2 * recall * precision / (recall + precision)
            else:
                ef1 = 0

        return ecd, ef1

    # logger.info(file_name)
    gt_pts = normalize_mesh_export(gt_pts)
    rec_mesh = normalize_mesh_export(rec_mesh)

    # sample point for rec
    pts_rec, idx = rec_mesh.sample(sample_num, return_index=True)
    normals_rec = rec_mesh.face_normals[idx]
    # sample point for gt
    pts_gt = None
    normals_gt = None
    if isinstance(gt_pts, trimesh.PointCloud):
        if gt_pts.shape[0] < sample_num:
            sample_num = gt_pts.shape[0]
        idx = np.random.choice(gt_pts.vertices.shape[0], sample_num, replace=False)
        pts_gt = gt_pts.vertices[idx]
        normals_gt = None
    elif isinstance(gt_pts, trimesh.Trimesh):
        pts_gt, idx = gt_pts.sample(sample_num, return_index=True)
        normals_gt = gt_pts.face_normals[idx]

    normals_correctness, chamferL1, chamferL2, f1_mu = eval_pointcloud(pts_rec, pts_gt, normals_rec, normals_gt)

    euler_num = gt_pts.euler_number - rec_mesh.euler_number
    euler_num = np.abs(euler_num)

    out_dict = dict()
    out_dict['nc'] = normals_correctness
    out_dict['cdL1'] = chamferL1
    out_dict['cdL2'] = chamferL2
    out_dict['f1_mu'] = f1_mu
    out_dict['eul'] = euler_num

    if is_final_res:
        # CD and f1 for the points on the edges
        logger.info('--calculating the ecd and ef1--')
        ecd, ef1 = get_ecd_ef1(pts_rec, pts_gt, normals_rec, normals_gt)

        out_dict['ecd'] = ecd
        out_dict['ef1'] = ef1

    return out_dict


def normalize_mesh_export(mesh, file_out=None):
    # unit to [-0.5, 0.5]
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    if file_out is not None:
        mesh.export(file_out)
    return mesh
