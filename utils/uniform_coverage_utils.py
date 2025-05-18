import torch
import math
from histogram_batched._C import histogram_batched # require libc10.so which is in torch, first import torch
import numpy as np
from utils.camera import Camera
import time

def estimate_scene_aabb(poses, extension):
    """
    Args:
        extension: in percent of each edge
    """
    p = torch.stack(poses)
    p = p[:, 0:3, 3]
    scene_aabb_min, idx = torch.min(p, dim=0)
    scene_aabb_max, idx2 = torch.max(p, dim=0)
    
    extend = (scene_aabb_max - scene_aabb_min) * extension/100
    scene_aabb_min = scene_aabb_min - extend
    scene_aabb_max = scene_aabb_max + extend
     
    return scene_aabb_min, scene_aabb_max

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def listify_tensor(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append([i.item() for i in row])
    return matrix_list

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def hfov2vfov(hfov, h, w):
    vfov = 2 * np.arctan(np.tan(hfov/2) * (h/w))
    return vfov

converter = torch.tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).float()
def create_cameras(poses, fovx, h, w, downsample=1):
    t0 = time.time()

    fx = fov2focal(fovx, w)
    vfov = hfov2vfov(fovx, h, w)
    fy = fov2focal(vfov, h)
    assert abs(fx-fy) < 1e-7, f"The pixel is not square, fx: {fx}, fy: {fy}"
    fy = fx
    
    cameras = [Camera(pose @ converter.to(pose.device), fx, fy, h, w, downscale=downsample, cam_idx=i) for i, pose in enumerate(poses)]
    print(f"Time {time.time()-t0}")
    torch.cuda.empty_cache()

    return cameras

def sample_nodes_uniformgrid(scene_aabb_min, scene_aabb_max, step_size):
    rx = int(torch.round((scene_aabb_max[0] - scene_aabb_min[0])/step_size).item())
    ry = int(torch.round((scene_aabb_max[1] - scene_aabb_min[1])/step_size).item())
    rz = int(torch.round((scene_aabb_max[2] - scene_aabb_min[2])/step_size).item())
   
    print("Grid size (x,y,z): " , rx, ry, rz)
    x, y, z = torch.meshgrid(torch.arange(rx), torch.arange(ry), torch.arange(rz))
    x = (x / (rx-1)) * (scene_aabb_max[0] - scene_aabb_min[0]) + scene_aabb_min[0]
    y = (y / (ry-1)) * (scene_aabb_max[1] - scene_aabb_min[1]) + scene_aabb_min[1]
    z = (z / (rz-1)) * (scene_aabb_max[2] - scene_aabb_min[2]) + scene_aabb_min[2]

    xyz = torch.stack((x, y, z), dim=-1).view(-1, 3)
    return xyz, (rx,ry,rz)


def weighted_dist(dists):
    weight = -0.1 * dists + 1
    weight[weight<0] = 0
    return weight

def getDirsOnPolar(dirs, H, W):
    def cartesian_to_polar(xyz):
        ptsnew = torch.cat((xyz, torch.zeros(xyz.shape, device="cuda")), dim=-1)
        xy = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
        ptsnew[..., 3] = torch.sqrt(xy + xyz[..., 2] ** 2)
        ptsnew[..., 4] = torch.arctan2(torch.sqrt(xy), xyz[..., 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[..., 5] = torch.arctan2(xyz[..., 1], xyz[..., 0])
        return ptsnew

    # r, theta, phi
    sph_coord = cartesian_to_polar(dirs)[..., 3:].float()
    sph_coord[..., 1] = sph_coord[..., 1] / math.pi  # from 0 - pi to 0 - 1
    sph_coord[..., 2] = ((sph_coord[..., 2] + math.pi) / (2 * math.pi))  # from -pi - pi to 0 - 1

    # 0 - 1 to 0 - H/W
    y = sph_coord[..., 1] * H
    x = sph_coord[..., 2] * W

    return x,y

def project_points_to_many_cams(cameras, points):
    hom_points = torch.cat((points, torch.tensor([[1.0]], device="cuda").repeat(points.shape[0], 1)), dim=1)
    pose_mats = torch.stack([torch.inverse(cam.pose_mat) for cam in cameras])
    wierd_mats = torch.stack([cam.wierd_mat for cam in cameras])


    camera_space_points = torch.einsum('lkj, ij -> ilk', pose_mats, hom_points.cuda().float())
    # NOTE: add weight w.r.t. distance
    weights = weighted_dist(camera_space_points[:,:,2])

    camera_space_points[:,:,0:2] /= camera_space_points[:,:,2:3]
    camera_space_points = camera_space_points[:,:,:3]
    wierd_points = torch.einsum('fkj, ifk -> ifk', wierd_mats, camera_space_points)

    return wierd_points, weights

def point_camera_mask(cameras, points):

    proj_mats = torch.stack([cam.get_proj_mat() for cam in cameras], dim=0)

    N_POINTS = points.shape[0]

    # Homogenify
    #hom_points = torch.cat((points, torch.tensor([[1.0]]).repeat(N_POINTS, 1)), dim=1)
    # Project Point to all cameras
    #projected_points = torch.einsum('lkj, ij -> ilk', proj_mats, hom_points.cuda().float())

    projected_points, weights = project_points_to_many_cams(cameras, points)

    positive_z_filter = projected_points[:, :, 2:3] > 0
    #projected_points_divz = projected_points/projected_points[:, : , 2:3]

    a = torch.tensor([c.w for c in cameras])
    assert (a - a[0]).sum()==0, f"{a}"
    incamera_filter = (projected_points[:, :, 0:1] > -cameras[0].w/2.0) & (projected_points[:, :, 0:1] < cameras[0].w/2.0) & \
                      (projected_points[:, :, 1:2] > -cameras[0].h/2.0) & (projected_points[:, :, 1:2] < cameras[0].h/2.0)

    return torch.logical_and(positive_z_filter, incamera_filter)

def per_pixel_area_of_sphere(H, W):
    D_theta = math.pi / H
    D_phi = 2 * math.pi / W
    perpixel_area = torch.sin(((torch.arange(H) + 0.5) / H) * math.pi)[:, None].repeat(1, W) * D_theta * D_phi
    return perpixel_area

def compute_KSMetric_PerNode(cameras, nodes):
    origins = torch.stack([cam.world_o for cam in cameras], dim=0)
    point_cam_mask = point_camera_mask(cameras, nodes)

    N_CAMERAS = origins.shape[0]
    N_NODES = nodes.shape[0]
    W = 150 
    H = 150 

    origins = origins.unsqueeze(0).repeat(N_NODES, 1, 1)
    nodes = torch.tensor(nodes).unsqueeze(1).repeat(1, N_CAMERAS, 1)
    NC_vector = nodes - origins

    # Get camera directiosn per node
    x, y = getDirsOnPolar(NC_vector, H, W)

    cdf_resolution = 5

    pdf_u = torch.ones((cdf_resolution, 2 * cdf_resolution), device="cuda") / (cdf_resolution * 2 * cdf_resolution)

    hist_gk = histogram_batched(x.cuda(),
                                y.cuda(),
                                point_cam_mask.int(),
                                cdf_resolution,
                                0, W, 0, H)
    hist_gk_area_weighted = hist_gk/per_pixel_area_of_sphere(cdf_resolution, cdf_resolution*2).unsqueeze(0).cuda()
    pdf_gk = hist_gk_area_weighted/hist_gk_area_weighted.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).clamp_min(0.00001)

    angular_distance = 1 - torch.abs(pdf_gk - pdf_u).sum(dim=-1).sum(dim=-1)/2.0
    
    angular_distance[(pdf_gk.view(pdf_gk.shape[0], -1)==0.0).all(dim=1)] = 0.0

    spatial_distance = point_cam_mask.squeeze(-1).float().sum(dim=1)/len(cameras)

    metric_gk = angular_distance + torch.pow(spatial_distance, 0.1/2.0)

    torch.cuda.empty_cache()

    return metric_gk, spatial_distance, angular_distance

