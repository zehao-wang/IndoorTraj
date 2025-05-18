import argparse
import os
import json
import numpy as np
import open3d as o3d
import copy
from utils import readColmapSceneInfo, convertRTtoTransformMatrix

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
    for split in args.scenes:
        source_path = os.path.join(args.input, split)
        cam_infos, ply_path = readColmapSceneInfo(source_path) 

        FoVx = cam_infos[0].FovX
        FoVy = cam_infos[0].FovY
        fx = cam_infos[0].intr.params[0]
        fy = cam_infos[0].intr.params[1]
        cx = cam_infos[0].intr.params[2]
        cy = cam_infos[0].intr.params[3]
        w = cam_infos[0].intr.width
        h = cam_infos[0].intr.height

        llffhold = 8 # follow standard mipnerf360 split
        train_indices = [idx for idx in range(len(cam_infos)) if idx % llffhold != 0]

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

        for i, cam in enumerate(cam_infos):
    
            image_file = os.path.join(os.path.abspath(cam.image_path))
            
            transform_matrix = convertRTtoTransformMatrix(cam.R, cam.T)

            if i in train_indices:
                meta["frames"].append({
                    "file_path": image_file,
                    "transform_matrix": transform_matrix.tolist()
                })
            else:
                meta_test["frames"].append({
                    "file_path": image_file,
                    "transform_matrix": transform_matrix.tolist()
                })
        
        
        out_path = os.path.join(args.output, f"mp-colmaps-{split}") 
        os.makedirs(out_path, exist_ok=True)

        scale_world_space_and_dump(meta, meta_test, o3d.io.read_point_cloud(ply_path), out_path)

        # json.dump(meta, open(os.path.join(out_path, 'transforms_train.json'), 'w'), indent=2)
        # json.dump(meta_test, open(os.path.join(out_path, 'transforms_test.json'), 'w'), indent=2)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Root folder of input scenes')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--scenes', nargs='+', required=True, help='Scene list to process')
    args = parser.parse_args()
    main(args)

