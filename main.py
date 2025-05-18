import argparse
from utils.io import load_meta, dump_meta
import os
from tqdm import tqdm
from PIL import Image
from easydict import EasyDict as edict
import copy
import numpy as np
from subset_sampling import avail_measures, avail_sampler

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", dest="input_dir", default='')

parser.add_argument("--out-dir", dest="out_dir", default='')
parser.add_argument("--cache-dir", dest="cache_dir", default='')
parser.add_argument("--sigma", default=0.2, type=float)

parser.add_argument("--sample_ratio", default=0.05, type=float)
parser.add_argument("--scale_down", default=2, type=int)
parser.add_argument("--measure", default='jointm', type=str, choices=['jointm'])
parser.add_argument("--dmatrix_alpha", default=0.7, type=float)
parser.add_argument("--dmatrx_beta", default=0.2, type=float)
parser.add_argument("--sample_strategy", default='random', type=str, choices=['df', 'dpp', 'cf', 'random', 'uniform'])
args = parser.parse_args()

def main():
    meta_raw, poses = load_meta(args.input_dir)
    sample_size=int(len(poses) * args.sample_ratio)

    out_template = "_{}_{}_ratio{}"
    out_dir = args.out_dir + out_template.format(args.sample_strategy, args.measure, args.sample_ratio)
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = args.cache_dir + "_cache_alpha{}_beta{}".format(args.dmatrix_alpha, args.dmatrx_beta)
    os.makedirs(cache_dir, exist_ok=True)

    h, w = int(meta_raw['h']), int(meta_raw['w'])
    images = []
    for frame in tqdm(meta_raw['frames']):
        if os.path.exists(os.path.join(cache_dir, 'clip_sim.npy')):
            images.append(np.zeros((int(w/args.scale_down),int(h/args.scale_down),3)))
        else:
            img = Image.open(os.path.join(args.input_dir, frame['file_path']))
            images.append(img.resize((int(w/args.scale_down), int(h/args.scale_down))))

    config_L = {
        "cache_dir": cache_dir,
        "out_dir": os.path.abspath(out_dir),
        "sigma": args.sigma, 
        "alpha": args.dmatrix_alpha, 
        "beta": args.dmatrx_beta,
    }
    config_L = edict(config_L)

    sample_stg = avail_measures[args.measure](images, poses, config_L)
    measure = sample_stg.M_ensemble
    
    D = [i for i in range(len(images))]
    sampler = avail_sampler[args.sample_strategy](D, measure)

    if args.sample_strategy == 'cf':
        fovx = meta_raw['camera_angle_x']
        selected_index = sampler.sample(sample_size, poses, fovx, w, h)
    else:
        selected_index = sampler.sample(sample_size)

    print("Selected index: ", selected_index)
    print("Selected index (sorted): ", sorted(selected_index))
    print("Number of samples: ", len(selected_index))
    print()

    meta = copy.deepcopy(meta_raw)
    meta['frames'] = [datum for i, datum in enumerate(meta['frames']) if i in selected_index]

    assert args.input_dir != out_dir
    dump_meta(os.path.abspath(args.input_dir), out_dir=out_dir, meta=meta)


if __name__ == '__main__':
    main()
