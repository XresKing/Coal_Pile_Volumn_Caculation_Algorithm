import open3d as o3d
import numpy as np

print("->正在加载点云... ")
#pcd = o3d.io.read_point_cloud(r"./map_900m/map_900m.pcd")
pcd = o3d.io.read_point_cloud(r"E:\SLAM\GuangXiSteel\Mine_Shed\Mine_Shed\2025-10-23-14-32-33_RAMY_full_map.pcd")
print(pcd)

print("->展示可视化中")
o3d.visualization.draw_geometries([pcd])
