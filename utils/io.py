import json
import os
import shutil
import torch

def rename_filepath(input_dir, file):
    meta = json.load(open(file))
    for frame in meta['frames']:
        if input_dir not in frame['file_path']:
            frame['file_path'] = os.path.join(input_dir, frame['file_path'])
    json.dump(meta, open(file, 'w'), indent=2)

def dump_meta(input_dir, out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, 'transforms_test.json')):
        os.remove(os.path.join(out_dir, 'transforms_test.json'))
    shutil.copy(os.path.join(input_dir, 'transforms_test.json'), os.path.join(out_dir, 'transforms_test.json'))

    if os.path.exists(os.path.join(input_dir, 'points3d.ply')):
        shutil.copy(os.path.join(input_dir, 'points3d.ply'), os.path.join(out_dir, 'points3d.ply'))

    json.dump(meta, open(os.path.join(out_dir, 'transforms_train.json'), 'w'), indent=2)

    rename_filepath(input_dir, os.path.join(out_dir, 'transforms_train.json'))
    rename_filepath(input_dir, os.path.join(out_dir, 'transforms_test.json'))

def load_meta(input_dir):
    meta = json.load(open(os.path.join(input_dir, 'transforms_train.json')))

    poses = [torch.tensor(frame['transform_matrix']) for frame in meta['frames']]
    return meta, poses