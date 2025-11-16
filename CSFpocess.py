import CSF
import numpy as np
import open3d as o3d


class CSF_Algorithm:
    def __init__(self):
        pass

    def csf_seperation(self, point_net):
        csf = CSF.CSF()
        csf.params.bSloopSmooth = True  # 粒子设置为不可移动
        csf.params.cloth_resolution = 1.014542 # 布料网格分辨率
        csf.params.rigidness = 3  # 布料刚性参数3,对应论文中的k
        csf.params.time_step = 0.65  # 步长
        csf.params.class_threshold = 0.3  # 点云与布料模拟点的距离阈值0.3,对应论文中的beta

        #csf.params.ransac_iterations = 100
        #csf.params.num_iterations = 100
        csf.params.interations = 500  # 最大迭代次数500
        csf.setPointCloud(point_net)

        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        csf.do_filtering(ground, non_ground)



        return ground, non_ground

    def csf_visualize(self, np1, np2):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np1)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(np2)
        # 数组转点云
        pcd.paint_uniform_color([0, 1, 0])  # 自定义颜色
        pcd1.paint_uniform_color([1, 0, 0])  # 自定义颜色
        o3d.visualization.draw_geometries([pcd], window_name='ground')
        o3d.visualization.draw_geometries([pcd1], window_name='tree')
