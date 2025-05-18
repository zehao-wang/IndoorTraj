import numpy as np
import torch
from PIL import Image
from dppy.finite_dpps import FiniteDPP
import random
import os
from utils.uniform_coverage_utils import estimate_scene_aabb, sample_nodes_uniformgrid, compute_KSMetric_PerNode, create_cameras
from utils.time import execution_time
from tqdm import tqdm

__all__ = ['DPPSampler', 'MaxMinDistSampler', 'UniformCoverageSampler', 'UniformSampler', 'RandomSampler']

class MaxMinDistSampler:
    def __init__(self, D, measure):
        """
        Args:
            D: full dataset index
            measure: pair-wise distance
            z: value function
        """
        self.idx_full = D
        self.measure = measure 

    @execution_time
    def sample(self, K):
        print(f'\033[1;32m [INFO]\033[0m Sample by max-min distance value function')
        candidates = self.idx_full[:]
        data_size = self.measure.shape[0]

        # NOTE: random a start idx
        selected_index = [candidates.pop(np.random.randint(0, data_size-1))]
        print(f'\033[1;32m [INFO]\033[0m The subset starts from {selected_index}')

        while len(selected_index) < min(K, data_size):
            best_score = -np.inf
            local_idx = -1
            for i, idx in enumerate(candidates):
                # equation_score = -max(self.measure[idx][selected_index]) # equivalent version
                equation_score = min(1-self.measure[idx][selected_index]) 
 
                if equation_score > best_score:
                    best_score = equation_score
                    local_idx = i
            assert local_idx != -1
            selected_index.append(candidates.pop(local_idx))

        assert len(selected_index) == K

        return selected_index
     
class DPPSampler:
    def __init__(self, D, measure):
        """
        Args:
            D: full dataset index
            measure: pair-wise distance
        """
        self.idx_full = D
        self.measure = measure 

    def project_to_semi_definite(self, matrix):
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals[eigvals < 0] = 0  # Set small negative eigenvalues to zero
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
     
    @execution_time
    def sample(self, K):
        print(f'\033[1;32m [INFO]\033[0m Sample by DPP value function')
        L = self.measure.astype(np.float64)
        evals, evecs = np.linalg.eig(L)
        print("Eigen value min: ", evals.min())
        print("Number of negative: ", (evals<0).sum())

        if evals.min()<0:
            # NOTE: solve numerical precision issue
            L = self.project_to_semi_definite(L)

        dpp = FiniteDPP('likelihood', **{'L': L.astype(np.float64)})  
        try:
            dpp.sample_exact_k_dpp(size=K) 
        except Exception as e:
            import ipdb;ipdb.set_trace() # breakpoint 182 
            print(e)
        selected_index = dpp.list_of_samples[-1]
        return selected_index

class UniformCoverageSampler:
    def __init__(self, D, measure=None):
        """
        Args:
            D: full dataset index
        """
        self.idx_full = D

    def greedy_sample_cam_ori(self, poses, fovx, w, h, k, step_size, extension):
        scene_aabb_min, scene_aabb_max = estimate_scene_aabb(poses, extension=extension)
        xyz, grid_resolution = sample_nodes_uniformgrid(scene_aabb_min, scene_aabb_max, step_size=step_size)
        xyz = xyz.float().cuda()
        nodes = xyz

        selected_cameras = []
        candidate_cams = create_cameras(poses, fovx, h=h, w=w, downsample=4)

        for i in tqdm(range(k)):
            metrics = []
            for j, cam in enumerate(candidate_cams):
                metric_gk, spatial_distance, angular_distance = compute_KSMetric_PerNode(selected_cameras + [cam], nodes)
                metrics.append(metric_gk.sum())
            metrics = torch.tensor(metrics)

            selected_idx = torch.tensor(metrics).argmax()
            selected_cam = candidate_cams.pop(selected_idx)
            selected_cameras.append(selected_cam)

        selected_index = [cam.cam_idx for cam in selected_cameras]
        return selected_index
 
    @execution_time
    def sample(self, K, poses, fovx, w, h):
        print(f'\033[1;32m [INFO]\033[0m Sample by uniform coverage value function')
        selected_index = self.greedy_sample_cam_ori(poses, fovx, w, h, k=K, step_size=0.5, extension=20)
        return selected_index

class RandomSampler: 
    def __init__(self, D, measure=None):
        """
        Args:
            D: full dataset index
        """
        self.idx_full = D

    @execution_time
    def sample(self, K):
        print(f'\033[1;32m [INFO]\033[0m Sample randomly in temporal domain')
        selected_index = random.sample(self.idx_full, K)
        return selected_index

class UniformSampler: 
    def __init__(self, D, measure=None):
        """
        Args:
            D: full dataset index
        """
        self.idx_full = D
        
    @execution_time
    def sample(self, K):
        print(f'\033[1;32m [INFO]\033[0m Sample uniformly in temporal domain')
        selected_frames_uni = np.linspace(0, len(self.idx_full)-1, K).astype(int).tolist()
        assert len(set(selected_frames_uni)) == K
        selected_index = (np.array(selected_frames_uni) + np.random.randint(0,100)) % len(self.idx_full)
        return selected_index.tolist()

