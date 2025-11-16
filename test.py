import open3d as o3d
import numpy as np

print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud(r"./map_900m/map_900m.pcd")
print(pcd)

print("->展示可视化中")
o3d.visualization.draw_geometries([pcd])
