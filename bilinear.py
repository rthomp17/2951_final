import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interpn
import torch
from nerfacc import OccupancyGrid, ray_marching, unpack_info

def test_nerfacc():
    print(torch.cuda.is_available())
    device = "cuda:0"
    batch_size = 128
    rays_o = torch.rand((batch_size, 3), device=device)
    rays_d = torch.randn((batch_size, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    # Ray marching with near far plane.
    ray_indices, t_starts, t_ends = ray_marching(
        rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
    )

    # Ray marching with aabb.
    scene_aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
    ray_indices, t_starts, t_ends = ray_marching(
        rays_o, rays_d, scene_aabb=scene_aabb, render_step_size=1e-3
    )

    # Ray marching with per-ray t_min and t_max.
    t_min = torch.zeros((batch_size,), device=device)
    t_max = torch.ones((batch_size,), device=device)
    ray_indices, t_starts, t_ends = ray_marching(
        rays_o, rays_d, t_min=t_min, t_max=t_max, render_step_size=1e-3
    )

    # Ray marching with aabb and skip areas based on occupancy grid.
    scene_aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
    grid = OccupancyGrid(roi_aabb=[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]).to(device)
    ray_indices, t_starts, t_ends = ray_marching(
        rays_o, rays_d, scene_aabb=scene_aabb, grid=grid, render_step_size=1e-3
    )

    # Convert t_starts and t_ends to sample locations.
    t_mid = (t_starts + t_ends) / 2.0
    sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

# Incomplete, but if needed can implement
def interpolate_fast(image, new_x, new_y, xmin=0, ymin=0, xmax=1, ymax=1):
    dx, dy = (xmax-xmin)/image.shape[1], (ymax-ymin)/image.shape[0]
    x = np.arange(xmin, xmax, dx)
    y = np.arange(-ymax, ymax, dy)

    interp_spline = RectBivariateSpline(y, x, image)

    Z2 = interp_spline(new_y, new_x)

def interpolate(image, points, xmin=0, ymin=0, xmax=1, ymax=1):
    x = np.linspace(xmin, xmax, image.shape[1])
    y = np.linspace(ymin, ymax, image.shape[0])
    coords = (y,x)
    return interpn(coords, image, points)

image = np.array([[[0,1,2],[1,2,3],[2,3,4],[3,4,5]],[[1,2,3],[2,3,4],[3,4,5],[4,5,6]],[[2,3,4],[3,4,5],[4,5,6],[5,6,7]]])
print(interpolate(image, np.array([[0.5,0.5],[1,1]])))
