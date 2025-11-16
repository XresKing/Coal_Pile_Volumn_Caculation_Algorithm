import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R





def extract_plane_ransac(pcd, distance_threshold=0.01, ransac_n=8, num_iterations=3000):
    """
    RANSAC
    使用的方程 ax + by + cz + d = 0

    :param pcd: 输入的点云
    :param distance_threshold: 点到平面的最大距离，用于确定内点
    :param ransac_n: RANSAC 使用的点数
    :param num_iterations: RANSAC 最大迭代次数
    :return: 平面参数 (a, b, c, d)
    """
    # 估计法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 使用 RANSAC 找平面
    plane_model, inlier_indices = pcd.segment_plane(distance_threshold=distance_threshold,
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
    # plane_model 是一个包含平面方程的 4 个系数 [a, b, c, d]
    a, b, c, d = plane_model
    print(f"提取的平面方程: {a}x + {b}y + {c}z + {d} = 0")

    # 提取内点
    inlier_cloud = pcd.select_by_index(inlier_indices)
    outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)

    return a, b, c, d, inlier_cloud, outlier_cloud


def rotate_to_horizontal(pcd, a, b, c, d):
    """
    算旋转矩阵，同时进行刚体变化

    :param pcd: 输入点云
    :param a, b, c: 平面方程的参数
    :return: 旋转后的点云
    """
    # 计算法向量
    normal = np.array([a, b, c])
    plane_pt=np.array([0,0,-(d/c)])
    # 目标法向量是 [0, 0, 1]，即水平面
    target_normal = np.array([0, 0, 1])

    if normal.dot(plane_pt) < 0:
        normal = -normal

    # 计算旋转轴：法向量与目标法向量的叉积
    rotation_axis = np.cross(normal, target_normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # 单位化

    # 计算旋转角度：法向量与目标法向量的夹角
    rotation_angle = np.arccos(np.dot(normal, target_normal) / (np.linalg.norm(normal) * np.linalg.norm(target_normal)))

    # 创建旋转矩阵（使用Rodrigues旋转公式）
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

    # 旋转点云
    center = pcd.get_center()
    pcd.rotate(rotation_matrix, center=center)  # 旋转至原点

    return pcd


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(r"./map_900m/map_900m.pcd")

    # 提取平面方程
    a, b, c, d, inlier_cloud, outlier_cloud = extract_plane_ransac(pcd)
    print(f"raw pcd: {pcd}")
    print(f"inlier_cloud is {inlier_cloud}")
    print(f"outlier_cloud is {outlier_cloud}")
    rotated_pcd = rotate_to_horizontal(pcd, a, b, c, d)
    o3d.visualization.draw_geometries([inlier_cloud,rotated_pcd], window_name="Extracted Plane")

    #o3d.visualization.draw_geometries([rotated_pcd], window_name="Rotated to Horizontal")
