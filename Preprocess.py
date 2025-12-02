import open3d as o3d
import numpy as np
from CSFpocess import *
from RANSAC import *
from scipy.spatial import Delaunay
from scipy.optimize import leastsq


# --- 嵌入 CoalVolumeCalculator 类 ---
class CoalVolumeCalculator:
    """
    实现论文中的煤堆体积计算与地面变形补偿算法。
    """

    def __init__(self):
        self.ground_params = None

    def _quadratic_surface(self, params, x, y):
        """
        论文公式 (20): 二次曲面模型
        f(x, y) = ax^2 + by^2 + cxy + dx + ey + f
        """
        a, b, c, d, e, f_const = params
        return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f_const

    def _error_func(self, params, x, y, z):
        """
        最小二乘法的误差函数
        """
        return z - self._quadratic_surface(params, x, y)

    def fit_ground_surface(self, ground_points):
        """
        对应论文 Section III-G (1): Ground Fitting
        使用二次函数拟合地面点云。
        """
        if len(ground_points) < 6:
            print("警告：地面点数量太少，无法拟合二次曲面。")
            return None

        x = ground_points[:, 0]
        y = ground_points[:, 1]
        z = ground_points[:, 2]

        # 初始参数猜测 [a, b, c, d, e, f]
        initial_guess = [0, 0, 0, 0, 0, np.mean(z)]

        # 使用 scipy.optimize.leastsq 进行参数优化
        params, success = leastsq(self._error_func, initial_guess, args=(x, y, z))

        self.ground_params = params
        print(f"地面拟合完成。参数: {params}")
        return params

    def _deformation_compensation(self, h):
        """
        对应论文 Section III-F (2): Compensation Calculation
        论文公式 (13): m_z' = m_z + log(m_z + 1) * 0.3
        """
        # 只有当高度大于0时才需要补偿
        compensation = np.log(h + 1) * 0.3
        return h + compensation

    def calculate_volume(self, coal_points):
        """
        对应论文 Section III-G (2): Triangular Mesh Construction and Volume Calculation
        """
        if self.ground_params is None:
            raise RuntimeError("请先调用 fit_ground_surface 拟合地面模型。")

        # 至少需要4个点才能构建有体积的形状
        if len(coal_points) < 4:
            return 0.0

        # 1. 提取 XY 平面坐标进行三角网构建
        points_2d = coal_points[:, :2]
        try:
            tri = Delaunay(points_2d)
        except Exception as e:
            print(f"三角剖分失败: {e}")
            return 0.0

        total_volume = 0.0

        # 遍历所有三角形
        for simplex in tri.simplices:
            idx0, idx1, idx2 = simplex
            p0 = coal_points[idx0]
            p1 = coal_points[idx1]
            p2 = coal_points[idx2]

            # --- 计算三角形投影面积 (S) ---
            area_xy = 0.5 * np.abs(
                (p1[0] - p0[0]) * (p2[1] - p0[1]) -
                (p2[0] - p0[0]) * (p1[1] - p0[1])
            )

            if area_xy < 1e-6:
                continue

            # --- 计算拟合平面高度 (h_fit) ---
            center_x = (p0[0] + p1[0] + p2[0]) / 3.0
            center_y = (p0[1] + p1[1] + p2[1]) / 3.0
            h_fit = self._quadratic_surface(self.ground_params, center_x, center_y)

            # --- 计算相对高度 (h) ---
            avg_z = (p0[2] + p1[2] + p2[2]) / 3.0
            h_relative = avg_z - h_fit

            # --- 地面变形补偿 ---
            if h_relative > 0:
                h_final = self._deformation_compensation(h_relative)
            else:
                h_final = 0

            # --- 累加体积 ---
            volume_i = area_xy * h_final
            total_volume += volume_i

        return total_volume


class PointCloudFilter:
    def __init__(self, file_path):
        self.pcd = o3d.io.read_point_cloud(file_path)
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.5)
        self.ground = None
        self.filtered_pcd = None
        self.re_filtered_pcd = None
        self.ground_points_np = None  # 新增：用于存储地面点数据的numpy数组
        self.csf_algorithm = CSF_Algorithm()
        print(f"点云加载完成：{file_path}")

    def ransac(self) -> None:
        '''
        计算好的平面方程
        '''
        a, b, c, d, inlier_cloud, outlier_cloud = extract_plane_ransac(self.pcd)
        #b = 5
        #while not (-0.025016494596247017 <= b <= 0.018158658795456674):
            #a, b, c, d, inlier_cloud, outlier_cloud = extract_plane_ransac(self.pcd)

        print(f"raw pcd: {self.pcd}")
        print(f"inlier_cloud is {inlier_cloud}")
        print(f"outlier_cloud is {outlier_cloud}")

        self.pcd = rotate_to_horizontal(self.pcd, a, b, c, d)
        # 注意：这里 original code save 了 ransac 后的点云
        # o3d.io.write_point_cloud(r"./map_900m/map_900m_ransac.pcd", self.pcd)

    def pass_through_filter(self, min_bound, max_bound) -> None:
        """
        :param min_bound: 最小边界 (x_min, y_min, z_min)
        :param max_bound: 最大边界 (x_max, y_max, z_max)
        :return None
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

        # [修改]：获取地面点索引 (ground_idx) 和 非地面点索引 (non_ground)
        # 假设 csf_seperation 返回 (ground_index, non_ground_index)
        ground_idx, non_ground = self.csf_algorithm.csf_seperation(points2)

        # [修改]：保存地面点数据，用于后续体积计算的基准面拟合
        self.ground_points_np = points2[ground_idx]
        print(f"提取到地面点数量: {len(self.ground_points_np)}")

        re_filtered_points = points2[non_ground]
        self.re_filtered_pcd = o3d.geometry.PointCloud()
        self.re_filtered_pcd.points = o3d.utility.Vector3dVector(re_filtered_points)

        print("点云过滤完成")

    def remove_outliers(self, pcd, nb_neighbors=20, std_ratio=2.0) -> o3d.geometry.PointCloud:
        """
        使用统计离群点移除算法过滤点云中没有足够邻居的点。
        """
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        filtered_pcd = pcd.select_by_index(ind)
        return filtered_pcd

    def filter_by_normal_z(self, pcd, z_threshold=0.2, radius=0.9, max_nn=30) -> o3d.geometry.PointCloud:
        """
        通过法线的z分量过滤点云。
        """
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)
        z_values = np.abs(normals[:, 2])
        valid_indices = np.where(z_values >= z_threshold)[0]
        coal_pile_points = pcd.select_by_index(valid_indices)
        # wall_points = pcd.select_by_index(np.setdiff1d(np.arange(len(points)), valid_indices))
        return coal_pile_points, None  # 这里不需要 wall_points

    def filter_by_normal_y(self, pcd, z_threshold=0.2, y_threshold=-0.9) -> o3d.geometry.PointCloud:
        """
        根据法线方向和与z轴的夹角过滤点云。
        """
        normals = np.asarray(pcd.normals)
        valid_indices = np.where((normals[:, 1] > y_threshold) & (np.abs(normals[:, 2]) > z_threshold))[0]
        filtered_pcd = pcd.select_by_index(valid_indices)
        return filtered_pcd

    def euclidean_clustering(self, pcd, eps=0.05, min_points=10):
        """
        使用DBSCAN进行欧几里得聚类。
        """
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        max_label = labels.max()
        print(f"点云包含 {max_label + 1} 个聚类。")

        clustered_points = []
        for i in range(max_label + 1):
            cluster = pcd.select_by_index(np.where(labels == i)[0])
            if not (len(np.asarray(cluster.points)) < 100):
                clustered_points.append(cluster)
        return clustered_points

    def visualize(self, min_bound=None, max_bound=None) -> None:
        if len(self.pcd.points) == 0:
            print("警告：原始点云为空，无法进行可视化。")
            return

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
        self.pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
        self.filtered_pcd.paint_uniform_color([1, 0, 0])  # 红色

        geometries = [self.pcd, axis]
        if self.re_filtered_pcd is not None:
            geometries.append(self.re_filtered_pcd)

        if min_bound is not None and max_bound is not None:
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            geometries.append(bbox)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geom in geometries:
            vis.add_geometry(geom)

        view_control = vis.get_view_control()
        view_control.set_front([0.5, 0.5, 0.5])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 0, 1])
        view_control.set_zoom(0.1)

        vis.run()
        vis.destroy_window()
        print("可视化完成")


# 使用示例
if __name__ == "__main__":
    # 创建点云过滤器对象
    #filter = PointCloudFilter(r"./map_900m/map_900m.pcd")
    #filter = PointCloudFilter(r"E:\SLAM\GuangXiSteel\Coal_Algorithm\verificares\sim_cone_perfect.pcd")
    filter = PointCloudFilter(r"E:\SLAM\GuangXiSteel\Coal_Algorithm\verificares\sim_cone_with_ground.pcd")
    #filter = PointCloudFilter(r"E:\SLAM\GuangXiSteel\Coal_Algorithm\verificares\sim_deformation_test.pcd")


    filter.ransac()

    # 定义过滤范围
    max_bound = np.array([10.0, 20.0, 14.0])
    min_bound = np.array([-1265.0, -50.0, -10.0])

    # 应用过滤器 (同时提取地面点)
    filter.pass_through_filter(min_bound, max_bound)

    # 进一步处理
    filter.re_filtered_pcd, _ = filter.filter_by_normal_z(filter.re_filtered_pcd, z_threshold=0.2, radius=2.114514,max_nn=70)
    #filter.re_filtered_pcd = filter.remove_outliers(filter.re_filtered_pcd, nb_neighbors=20, std_ratio=1.0)
    #filter.re_filtered_pcd = filter.filter_by_normal_y(filter.re_filtered_pcd, z_threshold=0.2, y_threshold=-0.7)

    # 聚类
    clusters = filter.euclidean_clustering(filter.re_filtered_pcd, eps=5, min_points=50)

    print("--------------------------------------------------")
    print("开始体积计算流程...")

    # 1. 实例化体积计算器
    vol_calc = CoalVolumeCalculator()

    # 2. 拟合地面基准面
    # 使用 pass_through_filter 中保存的地面点
    if filter.ground_points_np is not None and len(filter.ground_points_np) > 0:
        print("正在拟合地面模型...")
        ground_params = vol_calc.fit_ground_surface(filter.ground_points_np)

        if ground_params is not None:
            # 3. 遍历所有聚类（煤堆），计算体积
            total_inventory_volume = 0.0

            all_cluster = o3d.geometry.PointCloud()

            for i, cluster in enumerate(clusters):
                # 将 Open3D 点云转换为 numpy 数组
                cluster_points_np = np.asarray(cluster.points)

                # 计算该聚类的体积
                try:
                    vol = vol_calc.calculate_volume(cluster_points_np)
                    print(f"聚类 Cluster {i + 1} (点数: {len(cluster_points_np)}) -> 体积: {vol:.4f} m^3")
                    total_inventory_volume += vol
                except Exception as e:
                    print(f"聚类 Cluster {i + 1} 计算出错: {e}")

                # 用于可视化
                cluster.paint_uniform_color(np.random.rand(3))
                all_cluster += cluster

            print(f"\n>>> 总煤堆体积: {total_inventory_volume:.4f} m^3 <<<")

            # 可视化结果
            o3d.visualization.draw_geometries([all_cluster], window_name="Clusters with Volume Calculated")

        else:
            print("地面拟合失败，无法计算体积。")
    else:
        print("错误：未检测到地面点数据。请检查 CSF 算法是否正确分离了地面点。")