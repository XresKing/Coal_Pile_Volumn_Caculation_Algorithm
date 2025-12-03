import open3d as o3d
import numpy as np
from CSFpocess import *
from RANSAC import *
from scipy.spatial import Delaunay
from scipy.optimize import leastsq
from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    # --- 预处理参数 ---
    voxel_size: float = 0.5  # 降采样体素大小

    # --- 范围裁剪 (根据实际场景修改) ---
    min_bound: tuple = (-1265.0, -50.0, -10.0)
    max_bound: tuple = (10.0, 20.0, 14.0)

    # --- 法线滤波参数 ---
    normal_radius: float = 2.11  # 法线估计搜索半径
    normal_max_nn: int = 70  # 法线估计最大邻居数

    # 墙壁/地面过滤阈值
    normal_z_threshold: float = 0.2  # Z轴分量阈值，保留坡度较缓的点（煤堆）
    normal_y_threshold: float = -0.7  # Y轴分量阈值，根据具体朝向过滤

    # --- 离群点移除 ---
    outlier_nb_neighbors: int = 20  # 统计滤波邻居数
    outlier_std_ratio: float = 1.0  # 标准差倍数

    # --- 聚类参数 ---
    cluster_eps: float = 5.0  # DBSCAN 聚类半径 (米)
    cluster_min_points: int = 50  # 聚类最小点数

    # --- 体积计算参数 ---
    # 变形补偿系数。论文建议0.3
    volume_deformation_scale: float = 0.0

    # 三角形最大边长 (米)
    # 防止 Delaunay 三角网在煤堆边缘跨越空洞区域（凸包效应）
    # 比 voxel_size 大 3-5 倍
    max_triangle_edge: float = 2.0


#体积计算类
class CoalVolumeCalculator:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ground_params = None

    def _quadratic_surface(self, params, x, y):
        '''
        f(x,y) = a·x2 +b·y2 +c·x·y +d·x+e·y+f
        '''
        a, b, c, d, e, f_const = params
        return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f_const

    def _error_func(self, params, x, y, z):
        return z - self._quadratic_surface(params, x, y)

    def fit_ground_surface(self, ground_points):
        """
        拟合地面基准面 (二次曲面)
        """
        if len(ground_points) < 100:
            print("错误：地面点数量不足，无法拟合。")
            return None

        x = ground_points[:, 0]
        y = ground_points[:, 1]
        z = ground_points[:, 2]

        # 初始猜测
        initial_guess = [0, 0, 0, 0, 0, np.mean(z)]

        try:
            params, _ = leastsq(self._error_func, initial_guess, args=(x, y, z))
            self.ground_params = params
            print(f"地面拟合参数: {params}")
            return params
        except Exception as e:
            print(f"地面拟合失败: {e}")
            return None

    def _deformation_compensation(self, h):
        """
        应用论文中的对数补偿 (如果配置开启)
        """
        if self.config.volume_deformation_scale <= 0:
            return h

        # 仅对正高度进行补偿
        compensation = np.log(h + 1) * self.config.volume_deformation_scale
        return h + compensation

    def calculate_volume(self, coal_points):
        if self.ground_params is None:
            raise RuntimeError("必须先拟合地面。")

        if len(coal_points) < 4:
            return 0.0

        # 1. 投影到 XY 平面并构建三角网
        points_2d = coal_points[:, :2]
        try:
            tri = Delaunay(points_2d)
        except Exception as e:
            print(f"三角剖分错误: {e}")
            return 0.0

        total_volume = 0.0 #总体积
        skipped_triangles = 0   #过滤的有凸包效应的三角形

        for simplex in tri.simplices:
            p0 = coal_points[simplex[0]]
            p1 = coal_points[simplex[1]]
            p2 = coal_points[simplex[2]]

            # --- 过滤过大的三角形 ---
            # 计算三边长
            edge_a = np.linalg.norm(p0 - p1)
            edge_b = np.linalg.norm(p1 - p2)
            edge_c = np.linalg.norm(p2 - p0)

            # 如果任何一条边超过阈值，说明这个三角形可能跨越了不该算的空区域
            if max(edge_a, edge_b, edge_c) > self.config.max_triangle_edge:
                skipped_triangles += 1
                continue

            # --- 计算投影面积 ---
            area_xy = 0.5 * np.abs(
                (p1[0] - p0[0]) * (p2[1] - p0[1]) -
                (p2[0] - p0[0]) * (p1[1] - p0[1])
            )

            if area_xy < 1e-6:
                continue

            # --- 计算中心点高度 ---
            center_x = (p0[0] + p1[0] + p2[0]) / 3.0
            center_y = (p0[1] + p1[1] + p2[1]) / 3.0

            # 拟合基准面高度
            h_fit = self._quadratic_surface(self.ground_params, center_x, center_y)

            # 实际平均高度
            avg_z = (p0[2] + p1[2] + p2[2]) / 3.0

            # 相对高度
            h_relative = avg_z - h_fit

            # --- 负高度处理 ---
            if h_relative <= 0:
                continue  # 忽略地面以下的噪点体积

            # --- 变形补偿 ---
            h_final = self._deformation_compensation(h_relative)

            total_volume += area_xy * h_final

        if skipped_triangles > 0:
            print(f"  (过滤了 {skipped_triangles} 个边缘长三角形以修正凸包误差)")

        return total_volume


#过滤器类
class PointCloudFilter:
    def __init__(self, file_path, config: ProcessingConfig):
        self.config = config
        self.pcd = o3d.io.read_point_cloud(file_path)
        # 使用配置参数
        self.pcd = self.pcd.voxel_down_sample(voxel_size=self.config.voxel_size)

        self.ground = None
        self.filtered_pcd = None
        self.re_filtered_pcd = None
        self.ground_points_np = None
        self.csf_algorithm = CSF_Algorithm()
        print(f"点云加载完成，降采样 voxel_size={self.config.voxel_size}")

    def ransac(self) -> None:
        #b = 5
        # 增加最大迭代次数防止死循环
        max_iter = 100
        curr_iter = 0
        a, b, c, d, inlier_cloud, outlier_cloud = extract_plane_ransac(self.pcd)
        #while not (-0.025016494596247017 <= b <= 0.018158658795456674):
            #a, b, c, d, inlier_cloud, outlier_cloud = extract_plane_ransac(self.pcd)
            #curr_iter += 1
            #if curr_iter > max_iter:
                #print("RANSAC 迭代次数过多，使用最后一次结果")
                #break

        print(f"RANSAC 完成。参数: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
        self.pcd = rotate_to_horizontal(self.pcd, a, b, c, d)

    def pass_through_filter(self) -> None:
        points = np.asarray(self.pcd.points)

        # 使用配置中的边界
        min_b = np.array(self.config.min_bound)
        max_b = np.array(self.config.max_bound)

        mask = np.all((points >= min_b) & (points <= max_b), axis=1)
        filtered_points = points[mask]

        self.filtered_pcd = o3d.geometry.PointCloud()
        self.filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # CSF 地面分割
        points2 = np.asarray(self.filtered_pcd.points)
        ground_idx, non_ground = self.csf_algorithm.csf_seperation(points2)

        # 保存地面点用于拟合
        self.ground_points_np = points2[ground_idx]
        print(f"分离出地面点: {len(self.ground_points_np)} 个")

        re_filtered_points = points2[non_ground]
        self.re_filtered_pcd = o3d.geometry.PointCloud()
        self.re_filtered_pcd.points = o3d.utility.Vector3dVector(re_filtered_points)

    def process_pipeline(self):
        """执行一系列滤波操作"""
        # 1. 法线 Z 轴过滤
        self.re_filtered_pcd, _ = self.filter_by_normal_z(
            self.re_filtered_pcd,
            z_threshold=self.config.normal_z_threshold,
            radius=self.config.normal_radius,
            max_nn=self.config.normal_max_nn
        )

        # 2. 统计离群点移除
        self.re_filtered_pcd = self.remove_outliers(
            self.re_filtered_pcd,
            nb_neighbors=self.config.outlier_nb_neighbors,
            std_ratio=self.config.outlier_std_ratio
        )

        # 3. 法线 Y 轴过滤，去除墙壁
        self.re_filtered_pcd = self.filter_by_normal_y(
            self.re_filtered_pcd,
            z_threshold=self.config.normal_z_threshold,
            y_threshold=self.config.normal_y_threshold
        )

        return self.re_filtered_pcd

    def filter_by_normal_z(self, pcd, z_threshold, radius, max_nn):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        normals = np.asarray(pcd.normals)
        z_values = np.abs(normals[:, 2])
        valid_indices = np.where(z_values >= z_threshold)[0]
        return pcd.select_by_index(valid_indices), None

    def remove_outliers(self, pcd, nb_neighbors, std_ratio):
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return pcd.select_by_index(ind)

    def filter_by_normal_y(self, pcd, z_threshold, y_threshold):
        normals = np.asarray(pcd.normals)
        valid_indices = np.where((normals[:, 1] > y_threshold) & (np.abs(normals[:, 2]) > z_threshold))[0]
        return pcd.select_by_index(valid_indices)

    def euclidean_clustering(self, pcd):
        labels = np.array(pcd.cluster_dbscan(
            eps=self.config.cluster_eps,
            min_points=self.config.cluster_min_points
        ))
        max_label = labels.max()
        print(f"聚类完成: 发现 {max_label + 1} 个目标")

        clustered_points = []
        for i in range(max_label + 1):
            cluster = pcd.select_by_index(np.where(labels == i)[0])
            # 过滤太小的簇
            if len(cluster.points) > 100:
                clustered_points.append(cluster)
        return clustered_points



if __name__ == "__main__":
    # --- 步骤 0: 在这里集中调整参数 ---
    config = ProcessingConfig(
        voxel_size=0.5,

        # RANSAC大概率不准，此处算法需要结合无人机平面来确定
        min_bound=(-1265.0, -50.0, -10.0),
        max_bound=(10.0, 20.0, 14.0),

        # 体积修正：设置为 0.0 论文测试为0.3导致计算结果偏大
        volume_deformation_scale=0.2,

        # 三角形最大边长
        # 建议设为 1.5 ~ 3.0 米
        max_triangle_edge=2.5,

        # 聚类参数
        cluster_eps=5.0,
        cluster_min_points=50
    )

    print(f"当前体积补偿系数: {config.volume_deformation_scale} (0.0 表示不补偿)")
    print(f"三角形最大边长阈值: {config.max_triangle_edge}m")

    # --- 加载与预处理 ---
    #processor = PointCloudFilter(r"./map_900m/map_900m.pcd", config)
    #processor = PointCloudFilter(r"E:\SLAM\GuangXiSteel\Coal_Algorithm\verificares\sim_cone_perfect.pcd", config)
    #processor = PointCloudFilter(r"E:\SLAM\GuangXiSteel\Coal_Algorithm\verificares\sim_cone_with_ground.pcd", config)
    processor = PointCloudFilter(r"E:\SLAM\GuangXiSteel\Coal_Algorithm\verificares\sim_deformation_test.pcd", config)

    # RANSAC 旋转
    #processor.ransac()

    # 范围裁剪与 CSF 地面提取
    processor.pass_through_filter()

    # 组合滤波 (离群点、法线等)
    processor.process_pipeline()

    # 聚类
    clusters = processor.euclidean_clustering(processor.re_filtered_pcd)

    # --- 体积计算 ---
    print("-" * 50)
    vol_calc = CoalVolumeCalculator(config)

    # 拟合地面
    if processor.ground_points_np is not None:
        print("拟合地面基准面...")
        vol_calc.fit_ground_surface(processor.ground_points_np)

        total_vol = 0.0
        vis_clusters = o3d.geometry.PointCloud()

        for i, cluster in enumerate(clusters):
            points = np.asarray(cluster.points)

            # 计算体积
            vol = vol_calc.calculate_volume(points)

            print(f"Cluster {i + 1}: 点数={len(points)}, 体积={vol:.2f} m^3")
            total_vol += vol

            # 可视化准备
            cluster.paint_uniform_color(np.random.rand(3))
            vis_clusters += cluster

        print(f"\n>>> 总计算体积: {total_vol:.2f} m^3 <<<")

        # 简单的可视化检查
        o3d.visualization.draw_geometries([vis_clusters], window_name="Result")

    else:
        print("错误：未提取到地面点，无法计算体积。")