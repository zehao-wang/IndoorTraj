import torch
import os
import numpy as np
import random
import json
import math
import copy
import cv2
import argparse
import open3d as o3d
random.seed(0)

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def scale_world_space_and_dump(meta_train, meta_test, point_cloud, out_path, scale=4.0):
    avglen = 0.
    nframes = len(meta_train['frames']) + len(meta_test['frames'])
    for f in meta_train['frames']:
        avglen += np.linalg.norm(np.array(f["transform_matrix"])[0:3,3])

    for f in meta_test['frames']:
        avglen += np.linalg.norm(np.array(f["transform_matrix"])[0:3,3])
         
    avglen /= nframes
    for f in meta_train['frames']:
        mat = np.array(f["transform_matrix"])
        mat[0:3, 3] *= scale / avglen
        f["transform_matrix"] = mat.tolist()

    for f in meta_test['frames']:
        mat = np.array(f["transform_matrix"])
        mat[0:3, 3] *= scale / avglen
        f["transform_matrix"] = mat.tolist()
		
    json.dump(meta_train, open(os.path.join(out_path, 'transforms_train.json'), 'w'), indent=2)
    json.dump(meta_test, open(os.path.join(out_path, 'transforms_test.json'), 'w'), indent=2)

    print(avglen)   
    raw_points = np.asarray(point_cloud.points)  * scale / avglen
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points)
    o3d.io.write_point_cloud(os.path.join(out_path, 'points3d.ply'), pcd)

def main(args):
    split_ratio = 0.1 # test set ratio
    step = args.step_size
    if args.scenes is not None:
        splits = args.scenes
        print(f"Processing scenes: {splits}")
    else:
        raise ValueError("No scene list provided. Please provide a list of scenes to process.")
    
    for split in splits:
        source_path = os.path.join(args.input, split)
        intrinsic_path = os.path.join(source_path,  'intrinsic', 'intrinsic_color.txt')
        pose_dir = os.path.join(source_path, "pose")

        image_dir = os.path.join(source_path, "color")
        img_dir_sorted = list(sorted(os.listdir(image_dir), key=lambda x: int(x.split(".")[0])))
        pose_dir_sorted = list(sorted(os.listdir(pose_dir), key=lambda x: int(x.split(".")[0])))
        

        K = np.loadtxt(intrinsic_path)
        first_img = cv2.imread(os.path.join(image_dir, img_dir_sorted[0]))  
        h, w, _ = first_img.shape

        fy = K[1][1]
        fx = K[0][0]
        cy = K[1][2]
        cx = K[0][2]
        FoVy = focal2fov(fy, h)
        FoVx = focal2fov(fx, w)

        poses  = [] 
        image_filenames = []
        
        for i, (img, pose) in enumerate(zip(img_dir_sorted, pose_dir_sorted)):
            if i%step != 0:
                continue
            pose = np.loadtxt(os.path.join(pose_dir, pose))
            pose = np.array(pose).reshape(4, 4)
            pose[:3, 1] *= -1
            pose[:3, 2] *= -1
            pose = torch.from_numpy(pose).float()
            if np.isinf(pose).any():
                continue
            
            poses.append(pose)
            image_filenames.append(os.path.join(os.path.abspath(image_dir), img))

        indices = [i for i in range(len(poses))]

        # generate train/test split
        random.shuffle(indices)
        split_index = int(len(indices) * split_ratio)
        test_indices = sorted(indices[:split_index])
        train_indices = sorted(indices[split_index:])

        print(f"Scene: {split}")
        print("Training split:", len(train_indices))
        print("Testing split:", len(test_indices))
        print()

        meta = {
            "camera_angle_x": FoVx,
            "camera_angle_y": FoVy,
            "fl_x": fx,
            "fl_y": fy,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": 4, 
            "frames": []
        }
        meta_test = copy.deepcopy(meta)

        for i, (pose, image_file) in enumerate(zip(poses, image_filenames)):
            if i in train_indices:
                meta["frames"].append({
                    "file_path": image_file,
                    "transform_matrix": pose.tolist()
                })
            else:
                meta_test["frames"].append({
                    "file_path": image_file,
                    "transform_matrix": pose.tolist()
                })
        
        
        out_path = os.path.join(args.output, f"scannet-{split}") 
        os.makedirs(out_path, exist_ok=True)

        point_cloud = o3d.io.read_point_cloud(os.path.join(source_path, 'points3d.ply'))
        
        scale_world_space_and_dump(meta, meta_test, point_cloud, out_path)

        # json.dump(meta, open(os.path.join(out_path, 'transforms_train.json'), 'w'), indent=2)
        # json.dump(meta_test, open(os.path.join(out_path, 'transforms_test.json'), 'w'), indent=2)
        # shutil.copyfile(
        #     os.path.join(source_path, 'points3d.ply'), 
        #     os.path.join(out_path, 'points3d.ply')
        # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Root folder of input scenes')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--scenes', nargs='+', required=True, help='Scene list to process')
    parser.add_argument('--step_size', type=int, default=2)
    args = parser.parse_args()
    main(args)