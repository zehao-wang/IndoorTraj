import torch

class Camera():
    def __init__(self, pose, focal_x, focal_y, h, w, img_path=None, device='cuda', downscale=1, cam_idx=0):

        self.img_path = img_path

        self.focal_x = focal_x
        self.focal_y = focal_y
        self.h = h
        self.w = w
        self.cam_idx = cam_idx

        self.pose_mat = pose.to(device)
        self.world_o = pose[:3, 3].to(device)

        self.intr_mat = torch.tensor([[self.focal_x, 0 , self.w/2.0],
                                      [0, self.focal_y, self.h/2.0],
                                      [0, 0, 1]]).float().to(device)
        self.wierd_mat = torch.tensor([[self.focal_x, 0., 0.],
                                      [0, self.focal_y, 0.],
                                      [0, 0, 1]]).float().to(device)

    def get_proj_mat(self):
        return self.intr_mat @ self.pose_mat[:3,:]

    def project_points(self, points):
        tmp = self.pose_mat[:3, :] @ points
        tmp /= tmp[:, 2]
        final = self.intr_mat@tmp

        return final
