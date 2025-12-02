import numpy as np
import open3d as o3d
import math


def generate_ground(width, length, resolution, noise_level=0.0, deformation=False):
    """生成地面点云：支持添加噪声和模拟地面沉降形变"""
    x = np.arange(-width / 2, width / 2, resolution)
    y = np.arange(-length / 2, length / 2, resolution)
    xx, yy = np.meshgrid(x, y)

    # 基础地面 z = 0
    zz = np.zeros_like(xx)

    # 模拟地面形变 (模拟论文中的地面沉降或非水平地面)
    if deformation:
        zz = 0.001 * (xx ** 2 + yy ** 2) - 0.5

    points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))

    # 添加噪声
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, points.shape)
        points += noise

    return points


def generate_cone_pile(radius, height, resolution, center=(0, 0), noise_level=0.0):
    """生成圆锥体煤堆。理论体积 V = (1/3) * pi * r^2 * h"""
    theoretical_volume = (1 / 3) * math.pi * (radius ** 2) * height

    x = np.arange(center[0] - radius, center[0] + radius, resolution)
    y = np.arange(center[1] - radius, center[1] + radius, resolution)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = dist <= radius

    zz = np.zeros_like(dist)
    zz[mask] = height * (1 - dist[mask] / radius)

    valid_points = np.column_stack((xx[mask], yy[mask], zz[mask]))

    if noise_level > 0:
        noise = np.random.normal(0, noise_level, valid_points.shape)
        valid_points += noise

    return valid_points, theoretical_volume


def generate_spherical_cap_pile(base_radius, height, resolution, center=(0, 0), noise_level=0.0):
    """生成球缺体煤堆。理论体积 V = (pi * h / 6) * (3 * a^2 + h^2)"""
    theoretical_volume = (math.pi * height / 6) * (3 * base_radius ** 2 + height ** 2)
    R = (base_radius ** 2 + height ** 2) / (2 * height)  # 球半径

    x = np.arange(center[0] - base_radius, center[0] + base_radius, resolution)
    y = np.arange(center[1] - base_radius, center[1] + base_radius, resolution)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = dist <= base_radius

    zz = np.zeros_like(dist)
    sphere_z = np.sqrt(R ** 2 - dist[mask] ** 2)
    zz[mask] = sphere_z - (R - height)

    valid_points = np.column_stack((xx[mask], yy[mask], zz[mask]))

    if noise_level > 0:
        noise = np.random.normal(0, noise_level, valid_points.shape)
        valid_points += noise

    return valid_points, theoretical_volume


def save_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros_like(points)
    colors[:, 2] = points[:, 2] / np.max(points[:, 2]) if np.max(points[:, 2]) > 0 else 0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"已保存: {filename}")


if __name__ == "__main__":
    # 参数设置
    GROUND_SIZE = 50.0
    RESOLUTION = 0.1
    PILE_RADIUS = 10.0
    PILE_HEIGHT = 8.0
    NOISE = 0.02

    # 场景 1: 完美圆锥体
    cone_pts, cone_vol = generate_cone_pile(PILE_RADIUS, PILE_HEIGHT, RESOLUTION)
    save_pcd(cone_pts, "sim_cone_perfect.pcd")
    print(f"场景1 理论体积: {cone_vol:.4f} m3")

    # 场景 2: 地面 + 圆锥体
    ground_pts = generate_ground(GROUND_SIZE, GROUND_SIZE, RESOLUTION)
    scene2_pts = np.vstack((ground_pts, cone_pts))
    save_pcd(scene2_pts, "sim_cone_with_ground.pcd")
    print(f"场景2 理论体积: {cone_vol:.4f} m3")

    # 场景 3: 噪声地面 + 球缺体 + 沉降形变
    cap_pts, cap_vol = generate_spherical_cap_pile(PILE_RADIUS, PILE_HEIGHT, RESOLUTION, noise_level=NOISE)
    ground_def_pts = generate_ground(GROUND_SIZE, GROUND_SIZE, RESOLUTION, noise_level=NOISE, deformation=True)
    scene3_pts = np.vstack((ground_def_pts, cap_pts))
    save_pcd(scene3_pts, "sim_deformation_test.pcd")
    print(f"场景3 理论体积 (需地面拟合补偿): {cap_vol:.4f} m3")