# Diversity-Driven View Subset Selection for Indoor Novel View Synthesis
![Python](https://img.shields.io/badge/Python-3.7-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)

This repository accompanies the paper "Diversity-Driven View Subset Selection for Indoor Novel View Synthesis." The project focuses on formulating the objective of view subset selection and exploring a range of value functions that can guide effective and efficient sampling toward this goal.

## Table of Contents
- [0. Environment Setup](#0-environment-setup)
- [1. Data Preparation](#1-data-preparation)
- [2. Experiments](#2-experiments)
- [3. Citation](#citation)


## 0 Environment Setup
Setup environment following [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), and install packages listed in the ```requirements.txt```. 

Compile ```histogram_batched``` for uniform coverage value function following [link](https://gitlab.inria.fr/fungraph/progressive-camera-placement).
```bash
cd externalLib/histogram_batched
python setup.py install 
```

<details> 

<summary>Note: The standard gaussian-splatting dataloader does not support different extension and evolved several times. Please first checkout to rgb only version ...</summary>

```bash
cd gaussian-splatting
git checkout b2ada78a779ba0455dfdc2b718bdf1726b05a1b6
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

Then, modify the ```readCamerasFromTransforms``` and ```fetchPly``` in ```scene/dataset_readers.py```
```python
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        print(f'\033[1;32m [INFO]\033[0m Loading from {path} with size {len(frames)}')
        for idx, frame in enumerate(frames):
            if frame["file_path"].endswith('jpg'):
                extension='.jpg'
                if path in frame["file_path"]:
                    cam_name = frame["file_path"]
                else:
                    cam_name = os.path.join(path, frame["file_path"])
            elif frame["file_path"].endswith('JPG'):
                extension='.JPG'
                if path in frame["file_path"]:
                    cam_name = frame["file_path"]
                else:
                    cam_name = os.path.join(path, frame["file_path"])
            elif frame["file_path"].endswith('png'):
                extension='.png'
                if path in frame["file_path"]:
                    cam_name = frame["file_path"]
                else:
                    cam_name = os.path.join(path, frame["file_path"])
            else:
                cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            if path in cam_name:
                image_path = cam_name
            else:
                image_path = os.path.join(path, cam_name)
            
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if 'camera_angle_y' in contents.keys():
                FovY = contents['camera_angle_y']
                # print(focal2fov(fov2focal(fovx, image.size[0]), image.size[1]), FovY)
            else:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        print('\033[1;31m [Warning]\033[0m', "CANNOT find normals in the point cloud, initialized by zeros, also randomize the color")
        normals = np.zeros_like(positions)
        shs = np.random.random((len(positions), 3)) / 255.0
        colors = SH2RGB(shs)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
```
</details> 


## 1 Data Preparation

For our processed IndoorTraj and Replica scenes, you can download them from [link](https://drive.google.com/drive/folders/1-vYWz3apLcXT1w7sFH2HM2yO7zxw3JKK). The Replica dataset we used is based on the 2000 frames version provided by [NICE-SLAM](https://github.com/cvg/nice-slam) project. For ScanNet and MipNerf360, we provide processing scripts in the ```preprocessing``` folder.

The ```data``` directory should look like the following:
```bash
data/
├── processed
│   ├── indoortraj
│   │   ├── kitchen_1
│   │   ├── kitchen_2
│   │   ├── living_1
│   │   └── openplan_1
│   ├── mipnerf360
│   │   ├── mp-colmaps-bonsai
│   │   ├── mp-colmaps-counter
│   │   ├── mp-colmaps-kitchen
│   │   └── mp-colmaps-room
│   ├── replica
│   │   └── replica_part_test
│   │       ├── office2
│   │       ├── office3
│   │       ├── room1
│   │       └── room2
│   └── ScanNet
│       ├── scannet-scene0050_00
│       ├── scannet-scene0073_01
│       ├── scannet-scene0085_00
│       └── scannet-scene0134_02
└── raw
    ├── MiP-NeRF360
    │   ├── bonsai
    │   ├── counter
    │   ├── kitchen
    │   └── room
    └── ScanNet
```


## 2 Experiments

Sample subsets by different strategies, you can also change the output dir by modifying argument ```--out-dir```.

```bash
sh scripts/run_sample_5perc_indoortraj.sh # for instance, sample 5% subset of indoortraj dataset
```

Train and evaluate subsets on 3DGS.

```bash
sh scripts/train_5perc_indoortraj_3dgs.sh
```

## 3 Citation
```bibtex
@article{
    wang2025diversitydriven,
    title={Diversity-Driven View Subset Selection for Indoor Novel View Synthesis},
    author={Zehao Wang and Han Zhou and Matthew B. Blaschko and Tinne Tuytelaars and Minye Wu},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=F42CRfcp3D},
    note={}
}
```
