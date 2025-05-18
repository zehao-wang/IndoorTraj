import open_clip
import numpy as np
import os
import torch 
from tqdm import tqdm 
from utils.math import angular_distance,dist_gaussian_kernel
from utils.vis import visualize_matrix
from utils.time import execution_time

__all__ = ['MeasureJointM']

class MeasureBase(object):
    def __init__(self, images, poses, config_L) -> None:
        # initialize L matrix
        self.images = images
        self.images_np = [np.asarray(img)[:,:,:3]/255. for img in images]
        self.M_ensemble = self.get_M(poses, config_L)
        # visualize_matrix(self.M_ensemble, config_L.out_dir, file_name='M_ensemble.png')

    def get_M(self, poses, config):
        raise ValueError()

class MeasureJointM(MeasureBase):
    """
    include angular cosine sim, 3D poses sim, and clip image sim
    """

    @execution_time
    def get_M(self, poses, config):
        cache_dir = config.cache_dir

        alpha = config.alpha
        beta = config.beta

        # NOTE: image sim
        if os.path.exists(os.path.join(cache_dir, 'clip_sim.npy')):
            clip_sim = np.load(os.path.join(cache_dir, 'clip_sim.npy'))
        else:
            # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')
            model.eval()
            model = model.cuda()
            
            img_feats = []
            with torch.no_grad():
                for img in tqdm(self.images):
                    image = preprocess(img).unsqueeze(0)
                    image_features = model.encode_image(image.cuda())
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    img_feats.append(image_features)

                img_feats = torch.cat(img_feats, dim=0)
                try:
                    matrix = img_feats @ img_feats.T
                except Exception as e:
                    print(e)
                    import ipdb;ipdb.set_trace() # breakpoint 47
                    print()
                
                clip_sim = matrix.cpu().numpy()
                np.save(os.path.join(cache_dir, 'clip_sim.npy'), clip_sim)

        sigma=0.5

        ang_sim = angular_distance(torch.from_numpy(np.stack(poses))[:,:3,:3])
        ang_sim = ang_sim.cpu().numpy()

        pt_pos = np.stack([pose[:3, 3] for pose in poses])
        eu_dist_metric = dist_gaussian_kernel(pt_pos, sigma=sigma)
        
        joint_m = eu_dist_metric * alpha + ang_sim * beta + clip_sim * (1-alpha-beta)
        np.save(os.path.join(cache_dir, 'measure_joint_m.npy'), joint_m)

        return joint_m