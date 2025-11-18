import open3d as o3d
import numpy as np
from CSFpocess import *
from RANSAC import *


class PointCloudFilter:
    def __init__(self, file_path):
        self.pcd = o3d.io.read_point_cloud(file_path)
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.5)
        self.ground = None
        self.filtered_pcd = None
        self.re_filtered_pcd = None
        self.csf_algorithm = CSF_Algorithm()
        print(f"点云加载完成：{file_path}")

    def ransac(self):
        '''
        计算好的平面方程
        0.03394383016525596x + 0.0007017434695681669y + 0.9994234957963593z + -5.196453441143198 = 0
        0.03344523433200128x + -0.025016494596247017y + 0.9991274149469592z + -5.8429253551573055 = 0
        0.03191410843080341x + 0.018158658795456674y + 0.9993256490222879z + -5.880756471591452 = 0
        :return: none
        '''
        b = 5
        while not(-0.025016494596247017<= b <= 0.018158658795456674):
            a, b, c, d, inlier_cloud, outlier_cloud = extract_plane_ransac(self.pcd)
        #a, b, c, d, inlier_cloud, outlier_cloud = extract_plane_ransac(self.pcd)
        print(f"raw pcd: {self.pcd}")
        print(f"inlier_cloud is {inlier_cloud}")
        print(f"outlier_cloud is {outlier_cloud}")
        self.pcd = rotate_to_horizontal(self.pcd, a, b, c, d)
        self.ground = rotate_to_horizontal(inlier_cloud, a, b, c, d)
        o3d.io.write_point_cloud(r"./map_900m/map_900m_ransac.pcd", self.pcd)

    def apply_pass_through_filter(self, min_bound, max_bound):
        """
        :param min_bound: 最小边界 (x_min, y_min, z_min)
        :param max_bound: 最大边界 (x_max, y_max, z_max)
        """
        points = np.asarray(self.pcd.points)
        # 使用条件过滤
        mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        filtered_points = points[mask]

        # 创建过滤后的点云
        self.filtered_pcd = o3d.geometry.PointCloud()
        self.filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # 地点和非地点过滤
        points2 = np.asarray(self.filtered_pcd.points)
        _, non_ground = self.csf_algorithm.csf_seperation(points2)
        re_filtered_points = points2[non_ground]

        self.re_filtered_pcd = o3d.geometry.PointCloud()
        self.re_filtered_pcd.points = o3d.utility.Vector3dVector(re_filtered_points)

        print("点云过滤完成")

    def remove_outliers(self, pcd, nb_neighbors=20, std_ratio=2.0):
        """
        使用统计离群点移除算法过滤点云中没有足够邻居的点。
        :param pcd: 输入点云
        :param nb_neighbors: 每个点的邻域大小（邻居数）
        :param std_ratio: 标准差倍数，用于判定离群点的阈值
        :return: 过滤后的点云
        """
        # 使用统计离群点移除方法
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # 根据返回的索引选择有效点
        filtered_pcd = pcd.select_by_index(ind)

        return filtered_pcd

    def filter_by_normal_z(self, pcd, z_threshold=0.2, radius=0.9, max_nn=30):
        """
        计算点云的法线，使用PCA进行法线计算,通过法线的z分量过滤点云，分离煤堆和墙壁。
        :param pcd: 输入的点云
        :param z_threshold: 法线z分量的阈值
        :param radius: 用于法线计算的邻域半径
        :param max_nn: 用于法线计算的最大邻居点数
        :return: 返回煤堆点云和墙壁点云
        """
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)

        # 计算法线的z分量（dot product with vertical direction）
        z_values = np.abs(normals[:, 2])

        # 过滤掉法线z值小于阈值的点（这些通常是墙壁）
        valid_indices = np.where(z_values >= z_threshold)[0]

        # 提取煤堆点云
        coal_pile_points = pcd.select_by_index(valid_indices)

        # 提取墙壁点云
        invalid_indices = np.setdiff1d(np.arange(len(points)), valid_indices)
        wall_points = pcd.select_by_index(invalid_indices)

        return coal_pile_points, wall_points

    def filter_by_normal_y(self, pcd, z_threshold=0.2, y_threshold=-0.9):
        """
        根据法线方向和与z轴的夹角过滤点云，去除法线朝向 -y 方向且与z轴夹角小于阈值的点。
        :param pcd: 输入点云
        :param angle_threshold: 法线与z轴夹角的阈值（单位：度）
        :param y_threshold: 法线的y分量的阈值，朝向 -y 方向的点会被过滤
        :return: 过滤后的点云
        """
        # 估计法线
        #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # 获取法线数据
        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)

        # 计算法线与z轴的夹角（dot product with vertical direction, which is just the z component）
        z_values = normals[:, 2]
        y_values = normals[:, 1]

        # 计算法线与z轴的夹角
        angles = np.arccos(np.clip(z_values, -1.0, 1.0))  # 法线与z轴的夹角（弧度）

        # 将夹角转换为度
        angles_deg = np.degrees(angles)

        # 过滤掉法线朝向 -y 方向且与z轴夹角小于阈值的点
        valid_indices = np.where((normals[:, 1] > y_threshold) & (np.abs(normals[:, 2]) > z_threshold))[0]
        #valid_indices = np.where((y_values > y_threshold) and (y_values < -y_threshold))[0]

        # 提取有效点云
        filtered_pcd = pcd.select_by_index(valid_indices)

        return filtered_pcd

    def visualize(self):
        if len(self.pcd.points) == 0:
            print("警告：原始点云为空，无法进行可视化。")
            return

        # 坐标轴
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])

        self.pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
        self.filtered_pcd.paint_uniform_color([1, 0, 0])  # 红色
        self.re_filtered_pcd.paint_uniform_color([0, 1, 0]) #蓝色
        if self.filtered_pcd is not None and len(self.filtered_pcd.points) > 0:
            geometries = [self.re_filtered_pcd,self.pcd, axis]
        else:
            geometries = [self.pcd]

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        geometries.append(bbox)

        # 设置摄像头视角
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geom in geometries:
            vis.add_geometry(geom)


        # 获取视图控制器并调整摄像头位置
        view_control = vis.get_view_control()
        # 设置摄像头的视角（可以根据需求调整这些参数）
        view_control.set_front([0.5, 0.5, 0.5])  # 摄像头朝向
        view_control.set_lookat([0, 0, 0])  # 聚焦点
        view_control.set_up([0, 0, 1])  # 摄像头的"上"方向
        view_control.set_zoom(0.1)  # 摄像头的缩放

        # 渲染点云
        vis.run()
        vis.destroy_window()

        print("可视化完成")


# 使用示例
if __name__ == "__main__":
    # 创建点云过滤器对象
    filter = PointCloudFilter(r"./map_900m/map_900m.pcd")
    #filter = PointCloudFilter(r"E:\SLAM\GuangXiSteel\Mine_Shed\Mine_Shed\2025-10-23-14-32-33_RAMY_full_map.pcd")
    filter.ransac()
    # 定义过滤范围
    max_bound = np.array([10.0, 20.0, 14.0])  # 最大边界 [x_max, y_max, z_max]
    min_bound = np.array([-1265.0, -50.0, -10.0])  # 最小边界 [x_min, y_min, z_min]

    # 应用过滤器
    filter.apply_pass_through_filter(min_bound, max_bound)
    filter.re_filtered_pcd,_ = filter.filter_by_normal_z(filter.re_filtered_pcd, z_threshold=0.2, radius=2.114514,max_nn=70)
    filter.re_filtered_pcd = filter.remove_outliers(filter.re_filtered_pcd, nb_neighbors=20, std_ratio=1.0)
    filter.re_filtered_pcd = filter.filter_by_normal_y(filter.re_filtered_pcd, z_threshold=0.2, y_threshold= -0.7)
    print(filter.filtered_pcd)
    print(filter.re_filtered_pcd)

    # 可视化
    filter.visualize()

    o3d.visualization.draw_geometries([filter.re_filtered_pcd], point_show_normal=True)
