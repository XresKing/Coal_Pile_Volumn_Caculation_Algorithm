# **基于无人机 LiDAR SLAM 的室内煤堆盘点算法流程说明**

本系统实现了一套基于无人机激光雷达（LiDAR）点云数据的自动化煤堆体积测量流程。算法集成了点云预处理、姿态校正、地面分割、聚类分析以及基于地面拟合与形变补偿的体积计算模块。

## **1\. 算法核心流程 (Workflow Pipeline)**

整体处理流程主要包含以下六个阶段：

1. **数据加载与降采样 (Initialization)**  
   * 加载原始 PCD 点云数据。  
   * 应用 **体素下采样 (Voxel Downsampling)**，通过网格化减少点云密度，提高后续计算效率。  
   * *参数*: voxel\_size (默认 0.5m)。  
2. **姿态校正 (Attitude Correction)**  
   * 使用 **RANSAC (Random Sample Consensus)** 算法提取环境中的主平面（通常为地面或各向同性的基准面）。  
   * 计算该平面的法向量 $\\vec{n} = (a, b, c)$。  
   * 构建旋转矩阵 $R$，将点云刚体旋转，使得主平面法向量与世界坐标系 Z 轴 $(0, 0, 1)$ 对齐，确保后续高度计算的基准统一。  
3. **感兴趣区域提取与地面分割 (ROI & Ground Segmentation)**  
   * **直通滤波 (Pass-Through Filter)**: 根据预设的 min\_bound 和 max\_bound 裁剪点云，去除无关背景。  
   * **布料模拟滤波 (CSF, Cloth Simulation Filter)**:  
     * 将点云翻转，模拟布料在重力作用下覆盖在地形表面。  
     * 通过布料节点的最终位置区分**地面点 (Ground Points)** 和 **非地面点 (Non-Ground Points/Coal Points)**。  
     * *输出*: 纯净的地面点集 $P_g$ 和待处理的煤堆点集 $P_c$。  
4. **多级去噪 (Multi-stage Denoising)**  
   * **法线滤波 (Normal Filtering)**:  
     * 估计点云法线。  
     * **Z轴过滤**: 保留法线 Z 分量 $|n_z| > T_z$ 的点，去除垂直结构（如墙壁、立柱）。  
     * **Y轴过滤**: 根据场景朝向进一步剔除干扰。  
   * **统计离群点移除 (Statistical Outlier Removal)**:  
     * 计算每个点到临近 $k$ 个点的平均距离。  
     * 剔除距离超过 $\mu + \sigma \cdot \text{std\_ratio}$ 的噪点。  
5. **聚类分析 (Clustering)**  
   * 使用 **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** 算法对非地面点进行欧几里得聚类。  
   * 分离出独立的煤堆个体，便于单独计算每个煤堆的体积。  
6. **体积计算 (Volume Calculation)**  
   * **地面拟合**: 基于 CSF 提取的地面点，构建二次曲面基准模型。  
   * **三角网构建**: 对煤堆点进行 Delaunay 三角剖分。  
   * **体积积分**: 结合凸包修正与形变补偿，累加微分单元体积。

## **2\. 关键数学模型 (Mathematical Models)**

### **2.1 地面基准面拟合 (Ground Surface Fitting)**

由于实际地面并非绝对水平平面，算法采用**二次曲面模型**来拟合地面点 $P\_g$，以获取更精确的“零高度”基准。

$$ Z_{ground} (x, y) = ax^2 + by^2 + cxy + dx + ey + f$$

$$
\sin^2(\theta) + \cos^2(\theta) = 1
$$

* **求解方法**: 最小二乘法 (Least Squares Optimization)。  
* **目标函数**: $\min \sum (z_i - Z_{ground}(x_i, y_i))^2$。

### **2.2 Delaunay 三角网体积积分 (Volume Integration)**

将煤堆投影到 $XY$ 平面构建 Delaunay 三角网，将总体积 $V$ 分解为若干三角棱柱体积之和。

$$V = \sum_{i=1}^{N} S_i \times h'_{i}$$  
其中：

* $S_i$: 第 $i$ 个三角形在 $XY$ 平面的投影面积。  
  $$S_i = \frac{1}{2} |(x_1-x_0)(y_2-y_0) - (x_2-x_0)(y_1-y_0)|$$  
* $h'_{i}$: 第 $i$ 个棱柱的**补偿后相对高度**。

### **2.3 凸包效应修正 (Convex Hull Correction)**

Delaunay 三角剖分会自动连接最外层点形成凸包，可能导致在非凸形状（如 U 型煤堆）的边缘区域计算错误的“空气体积”。算法引入**最大边长阈值**进行过滤：

$$\text{If } \max(L_{edge\\_a}, L_{edge\\_b}, L_{edge\\_c}) > T_{max\\_edge}, \quad \text{Discard Triangle}$$

### **2.4 地面形变补偿 (Ground Deformation Compensation)**

基于论文理论，重载煤堆会导致地面发生非线性弹性形变（下沉），导致测量体积小于实际体积。引入对数补偿模型修正相对高度 $h$：

$$h' = h + \beta \log(h + 1)$$

* $h = \bar{z}_{tri} - Z_{ground}(\bar{x}_{tri}, \bar{y}_{tri})$: 测量相对高度。  
* $\beta$: 形变补偿系数 (volume_deformation_scale)。  
  * 对于硬化混凝土地面，建议设为 **0.0** (不补偿)。  
  * 对于软基泥地，论文建议值为 **0.3**。

## **3\. 配置参数说明 (Configuration)**

所有核心参数均通过 ProcessingConfig 数据类进行管理，便于调试：

| 参数类     | 参数名                        | 默认值   | 说明                             |
|:------- |:-------------------------- |:----- |:------------------------------ |
| **预处理** | voxel\_size                | 0.5   | 降采样网格大小 (m)                    |
| **裁剪**  | min\_bound/max\_bound      | (...) | 根据实际场景包围盒设定                    |
| **滤波**  | normal\_z\_threshold       | 0.2   | 法线 Z 分量阈值，越小越严格                |
| **聚类**  | cluster\_eps               | 5.0   | DBSCAN 聚类搜索半径 (m)              |
| **体积**  | max\_triangle\_edge        | 2.5   | 三角形最大边长，建议 voxel\_size 的 3-5 倍 |
| **体积**  | volume\_deformation\_scale | 0.2   | 形变补偿系数 (0.0 为关闭)               |

## **4\. 输出结果**

程序运行后将输出：

1. **RANSAC 平面参数**: $(a, b, c, d)$。  
2. **聚类信息**: 识别到的煤堆数量。  
3. **体积报告**: 每个独立煤堆的体积 ($m^3$) 及总库存体积。  
4. **可视化**: 包含 RANSAC 平面、分割后的煤堆以及体积计算结果的三维可视化。
