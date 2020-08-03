import argparse

import numpy as np

import os

import shutil

import subprocess

import sqlite3

import torch

import torch.nn.functional as func

import types

from tqdm import tqdm

from matchers import (mutual_nn_matcher, lisrd_matcher,
                      adalam_matcher, sequential_adalam_matcher)

from camera import Camera

from utils import quaternion_to_rotation_matrix, camera_center_to_translation

import sys
IS_PYTHON3 = sys.version_info[0] >= 3

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def recover_database_images_and_ids(paths, args):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    # Recover database images and ids.
    images = {}
    cameras = {}
    cursor.execute("SELECT name, image_id, camera_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]
        cameras[row[0]] = row[2]

    # Close the connection to the database.
    cursor.close()
    connection.close()

    return images, cameras


def preprocess_reference_model(paths, args):
    print('Preprocessing the reference model...')
    
    # Recover intrinsics.
    with open(os.path.join(paths.reference_model_path, 'aachen_v_1_1/database_intrinsics_v1_1.txt')) as f:
        raw_intrinsics = f.readlines()
    
    camera_parameters = {}

    for intrinsics in raw_intrinsics:
        intrinsics = intrinsics.strip('\n').split(' ')
        
        image_name = intrinsics[0]
        
        camera_model = intrinsics[1]

        intrinsics = [float(param) for param in intrinsics[2 :]]

        camera = Camera()
        camera.set_intrinsics(camera_model=camera_model, intrinsics=intrinsics)

        camera_parameters[image_name] = camera
    
    # Recover poses.
    with open(os.path.join(paths.reference_model_path, 'aachen_v_1_1/aachen_v_1_1.nvm')) as f:
        raw_extrinsics = f.readlines()

    # Skip the header.
    n_cameras = int(raw_extrinsics[2])
    raw_extrinsics = raw_extrinsics[3 : 3 + n_cameras]

    for extrinsics in raw_extrinsics:
        extrinsics = extrinsics.strip('\n').split(' ')

        image_name = extrinsics[0]

        # Skip the focal length. Skip the distortion and terminal 0.
        qw, qx, qy, qz, cx, cy, cz = [float(param) for param in extrinsics[2 : -2]]

        qvec = np.array([qw, qx, qy, qz])
        c = np.array([cx, cy, cz])
        
        # NVM -> COLMAP.
        t = camera_center_to_translation(c, qvec)

        camera_parameters[image_name].set_pose(qvec=qvec, t=t)
    
    return camera_parameters


def generate_empty_reconstruction(images, cameras, camera_parameters, paths, args):
    print('Generating the empty reconstruction...')

    if not os.path.exists(paths.empty_model_path):
        os.mkdir(paths.empty_model_path)
    
    with open(os.path.join(paths.empty_model_path, 'cameras.txt'), 'w') as f:
        for image_name in images:
            image_id = images[image_name]
            camera_id = cameras[image_name]
            try:
                camera = camera_parameters[image_name]
            except:
                continue
            f.write('%d %s %s\n' % (
                camera_id, 
                camera.camera_model, 
                ' '.join(map(str, camera.intrinsics))
            ))

    with open(os.path.join(paths.empty_model_path, 'images.txt'), 'w') as f:
        for image_name in images:
            image_id = images[image_name]
            camera_id = cameras[image_name]
            try:
                camera = camera_parameters[image_name]
            except:
                continue
            f.write('%d %s %s %d %s\n\n' % (
                image_id, 
                ' '.join(map(str, camera.qvec)), 
                ' '.join(map(str, camera.t)), 
                camera_id,
                image_name
            ))

    with open(os.path.join(paths.empty_model_path, 'points3D.txt'), 'w') as f:
        pass


def import_features(images, paths, args):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    # Import the features.
    print('Importing features...')
    
    for image_name, image_id in tqdm(images.items(), total=len(images.items())):
        features_path = os.path.join(paths.image_path, '%s.%s' % (image_name, args.method_name))

        data = np.load(features_path)
        keypoints = data['keypoints']
        scores = data['scores']
        n_keypoints = min(args.num_kp, keypoints.shape[0])
        keypoints = keypoints[np.argsort(scores)[-n_keypoints:]]
        
        # Keep only x, y coordinates.
        keypoints = keypoints[:, : 2]
        # Add placeholder scale, orientation.
        keypoints = np.concatenate([keypoints, np.ones((n_keypoints, 1)), np.zeros((n_keypoints, 1))], axis=1).astype(np.float32)
        
        keypoints_str = keypoints.tostring()
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                       (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_str))
        connection.commit()
    
    # Close the connection to the database.
    cursor.close()
    connection.close()


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def match_features(images, paths, args):
    def keypoint_list_to_grid_points(keypoints, img_size, device):
        """
        Convert a 2darray [N, 2] of keypoints into a grid in [-1, 1]²
        that can be used in torch.nn.functional.interpolate.
        """
        n_points = keypoints.shape[0]
        cuda_keypoints = torch.tensor(keypoints, dtype=torch.float,
                                      device=device)
        grid_points = cuda_keypoints * 2. / torch.tensor(
            img_size, dtype=torch.float, device=device) - 1.
        grid_points = grid_points[:, [1, 0]].view(1, n_points, 1, 2)
        return grid_points

    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    # Match the features and insert the matches in the database.
    print('Matching...')

    with open(paths.match_list_path, 'r') as f:
        raw_pairs = f.readlines()
    
    image_pair_ids = set()
    for raw_pair in tqdm(raw_pairs, total=len(raw_pairs)):
        image_name1, image_name2 = raw_pair.strip('\n').split(' ')
        
        features_path1 = os.path.join(paths.image_path, '%s.%s' % (image_name1, args.method_name))
        features_path2 = os.path.join(paths.image_path, '%s.%s' % (image_name2, args.method_name))

        # Extract local descriptors
        data1 = np.load(features_path1)
        descriptors1 = data1['descriptors']
        scores1 = data1['scores']
        n_keypoints1 = min(args.num_kp, len(descriptors1))
        sorted_idx1 = np.argsort(scores1)
        descriptors1 = descriptors1[sorted_idx1[-n_keypoints1:]]
        descriptors1 = torch.from_numpy(descriptors1).to(device).float()

        data2 = np.load(features_path2)
        descriptors2 = data2['descriptors']
        scores2 = data2['scores']
        n_keypoints2 = min(args.num_kp, len(descriptors2))
        sorted_idx2 = np.argsort(scores2)
        descriptors2 = descriptors2[sorted_idx2[-n_keypoints2:]]
        descriptors2 = torch.from_numpy(descriptors2).to(device).float()

        # Extract grid points
        if 'grid_points' in data1:
            grid_points1 = np.tile(
                data1['grid_points'][:, sorted_idx1[-n_keypoints1:], :, :],
                (4, 1, 1, 1))
            grid_points1 = torch.from_numpy(grid_points1).to(device).float()
            grid_points2 = np.tile(
                data2['grid_points'][:, sorted_idx2[-n_keypoints2:], :, :],
                (4, 1, 1, 1))
            grid_points2 = torch.from_numpy(grid_points2).to(device).float()
        else:
            assert 'img_size' in data1
            grid_points1 = []
            kp1 = data1['keypoints'][sorted_idx1[-n_keypoints1:]]
            scales1 = data1['scales'][sorted_idx1[-n_keypoints1:]]
            n_scales1 = np.amax(scales1)
            for s in range(n_scales1 + 1):
                grid_points1.append(keypoint_list_to_grid_points(
                    kp1[scales1 == s], data1['img_size'],
                    device).repeat(4, 1, 1, 1))

            grid_points2 = []
            kp2 = data2['keypoints'][sorted_idx2[-n_keypoints2:]]
            scales2 = data2['scales'][sorted_idx2[-n_keypoints2:]]
            n_scales2 = np.amax(scales2)
            for s in range(n_scales2 + 1):
                grid_points2.append(keypoint_list_to_grid_points(
                    kp2[scales2 == s], data2['img_size'],
                    device).repeat(4, 1, 1, 1))

        if 'meta_descriptors' in data1:
            meta_descriptors1 = data1['meta_descriptors']
            meta_descriptors1 = torch.from_numpy(
                meta_descriptors1).to(device).float()
            meta_descriptors2 = data2['meta_descriptors']
            meta_descriptors2 = torch.from_numpy(
                meta_descriptors2).to(device).float()

            if len(meta_descriptors1.shape) == 4:
                meta_descriptors1 = func.normalize(
                    func.grid_sample(meta_descriptors1, grid_points1),
                    dim=1).squeeze(3).permute(2, 0, 1)
                del grid_points1
                meta_descriptors2 = func.normalize(
                    func.grid_sample(meta_descriptors2, grid_points2),
                    dim=1).squeeze(3).permute(2, 0, 1)
                del grid_points2
            else:
                meta_desc1 = torch.empty(
                    n_keypoints1, 4, meta_descriptors1.shape[2],
                    dtype=torch.float, device=device)
                for s in range(n_scales1 + 1):
                    meta_desc1[scales1 == s] = func.normalize(
                        func.grid_sample(
                            meta_descriptors1[s], grid_points1[s]),
                        dim=1).squeeze(3).permute(2, 0, 1)
                meta_descriptors1 = meta_desc1
                del grid_points1
                meta_desc2 = torch.empty(
                    n_keypoints2, 4, meta_descriptors2.shape[2],
                    dtype=torch.float, device=device)
                for s in range(n_scales2 + 1):
                    meta_desc2[scales2 == s] = func.normalize(
                        func.grid_sample(
                            meta_descriptors2[s], grid_points2[s]),
                        dim=1).squeeze(3).permute(2, 0, 1)
                meta_descriptors2 = meta_desc2
                del grid_points2

            with torch.no_grad():
                if args.adalam:
                    matches = adalam_matcher(
                        torch.from_numpy(kp1).to(device).float(),
                        torch.from_numpy(kp2).to(device).float(),
                        descriptors1, descriptors2,
                        meta_descriptors1, meta_descriptors2,
                        data1['img_size'], data2['img_size']).astype(np.uint32)
                else:
                    matches = lisrd_matcher(
                        descriptors1, descriptors2,
                        meta_descriptors1, meta_descriptors2).astype(np.uint32)
            del descriptors1, descriptors2, meta_descriptors1, meta_descriptors2
        else:
            matches = mutual_nn_matcher(descriptors1,
                                        descriptors2).astype(np.uint32)

        image_id1, image_id2 = images[image_name1], images[image_name2]
        image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
        if image_pair_id in image_pair_ids:
            continue
        image_pair_ids.add(image_pair_id)

        if image_id1 > image_id2:
            matches = matches[:, [1, 0]]
        
        matches_str = matches.tostring()
        cursor.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                       (image_pair_id, matches.shape[0], matches.shape[1], matches_str))
        connection.commit()
    
    # Close the connection to the database.
    cursor.close()
    connection.close()


def geometric_verification(paths, args):
    print('Running geometric verification...')

    subprocess.call([os.path.join(args.colmap_path, 'colmap'), 'matches_importer',
                     '--database_path', paths.database_path,
                     '--match_list_path', paths.match_list_path,
                     '--match_type', 'pairs'])


def reconstruct(paths, args):
    if not os.path.isdir(paths.database_model_path):
        os.mkdir(paths.database_model_path)
    
    # Reconstruct the database model.
    subprocess.call([os.path.join(args.colmap_path, 'colmap'), 'point_triangulator',
                     '--database_path', paths.database_path,
                     '--image_path', paths.image_path,
                     '--input_path', paths.empty_model_path,
                     '--output_path', paths.database_model_path,
                     '--Mapper.ba_refine_focal_length', '0',
                     '--Mapper.ba_refine_principal_point', '0',
                     '--Mapper.ba_refine_extra_params', '0'])


def register_queries(paths, args):
    if not os.path.isdir(paths.final_model_path):
        os.mkdir(paths.final_model_path)
    
    # Register the query images.
    subprocess.call([os.path.join(args.colmap_path, 'colmap'), 'image_registrator',
                     '--database_path', paths.database_path,
                     '--input_path', paths.database_model_path,
                     '--output_path', paths.final_model_path,
                     '--Mapper.ba_refine_focal_length', '0',
                     '--Mapper.ba_refine_principal_point', '0',
                     '--Mapper.ba_refine_extra_params', '0'])


def recover_query_poses(paths, args):
    print('Recovering query poses...')

    if not os.path.isdir(paths.final_txt_model_path):
        os.mkdir(paths.final_txt_model_path)

    # Convert the model to TXT.
    subprocess.call([os.path.join(args.colmap_path, 'colmap'), 'model_converter',
                     '--input_path', paths.final_model_path,
                     '--output_path', paths.final_txt_model_path,
                     '--output_type', 'TXT'])
    
    # Recover query names.
    query_image_list_path = os.path.join(args.dataset_path, 'queries/night_time_queries_with_intrinsics.txt')
    
    with open(query_image_list_path) as f:
        raw_queries = f.readlines()
    
    query_names = set()
    for raw_query in raw_queries:
        raw_query = raw_query.strip('\n').split(' ')
        query_name = raw_query[0]
        query_names.add(query_name)

    with open(os.path.join(paths.final_txt_model_path, 'images.txt')) as f:
        raw_extrinsics = f.readlines()

    f = open(paths.prediction_path, 'w')

    # Skip the header.
    for extrinsics in raw_extrinsics[4 :: 2]:
        extrinsics = extrinsics.strip('\n').split(' ')

        image_name = extrinsics[-1]

        if image_name in query_names:
            # Skip the IMAGE_ID ([0]), CAMERA_ID ([-2]), and IMAGE_NAME ([-1]).
            f.write('%s %s\n' % (image_name.split('/')[-1], ' '.join(extrinsics[1 : -2])))

    f.close()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True,
                        help='Path to the dataset')
    parser.add_argument('--colmap_path', required=True,
                        help='Path to the COLMAP executable folder')
    parser.add_argument('--method_name', required=True,
                        help='Name of the method')
    parser.add_argument('--num_kp', type=int, default=10000,
                        help='Number of keypoints to use')
    parser.add_argument('--adalam', action='store_true',
                        help='Use Adalam filtering')
    args = parser.parse_args()

    # Torch settings for the matcher.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Create the extra paths.
    paths = types.SimpleNamespace()
    paths.dummy_database_path = os.path.join(args.dataset_path, 'database_v1_1.db')
    paths.database_path = os.path.join(args.dataset_path, args.method_name + '_v1_1.db')
    paths.image_path = os.path.join(args.dataset_path, 'images', 'images_upright')
    paths.features_path = os.path.join(args.dataset_path, args.method_name)
    paths.reference_model_path = os.path.join(args.dataset_path, '3D-models')
    paths.match_list_path = os.path.join(args.dataset_path, 'image_pairs_to_match_v1_1.txt')
    paths.empty_model_path = os.path.join(args.dataset_path, 'sparse-%s-empty' % args.method_name)
    paths.database_model_path = os.path.join(args.dataset_path, 'sparse-%s-database' % args.method_name)
    paths.final_model_path = os.path.join(args.dataset_path, 'sparse-%s-final' % args.method_name)
    paths.final_txt_model_path = os.path.join(args.dataset_path, 'sparse-%s-final-txt' % args.method_name)
    paths.prediction_path = os.path.join(args.dataset_path, 'Aachen_v1_1_eval_%s.txt' % args.method_name)
    
    # Create a copy of the dummy database.
    if os.path.exists(paths.database_path):
        raise FileExistsError('The database file already exists for method %s.' % args.method_name)
    shutil.copyfile(paths.dummy_database_path, paths.database_path)
    
    # Reconstruction pipeline.
    camera_parameters = preprocess_reference_model(paths, args)
    images, cameras = recover_database_images_and_ids(paths, args)
    generate_empty_reconstruction(images, cameras, camera_parameters, paths, args)
    import_features(images, paths, args)
    match_features(images, paths, args)
    geometric_verification(paths, args)
    reconstruct(paths, args)
    register_queries(paths, args)
    recover_query_poses(paths, args)
