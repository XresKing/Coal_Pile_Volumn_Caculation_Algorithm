import streamlit as st
import os
import numpy as np
import open3d as o3d
import pandas as pd
import random
# 确保 Preprocess.py 在同一目录下，且包含 grid_section_clustering 方法
from Preprocess import PointCloudFilter, CoalVolumeCalculator, ProcessingConfig

# --- 页面配置 ---
st.set_page_config(
    page_title="煤堆体积智能盘点系统",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 样式优化 (CSS) ---
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    /* 进度条样式 */
    .volume-bar-container {
        display: flex;
        width: 100%;
        height: 35px;
        border-radius: 8px;
        overflow: hidden;
        margin-top: 10px;
        margin-bottom: 20px;
        background-color: #eee;
        border: 1px solid #ddd;
    }
    .volume-segment {
        height: 100%;
        display: flex;
        align-items: center;
        justify_content: center;
        color: white;
        font-size: 12px;
        font-weight: bold;
        transition: width 0.5s ease-in-out;
        white-space: nowrap;
        overflow: hidden;
        text-shadow: 0px 0px 2px rgba(0,0,0,0.5);
    }
    .volume-segment:hover {
        opacity: 0.9;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)


# --- 辅助函数：获取文件列表 ---
def get_pcd_files(directory="./map_900m"):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError:
            pass  # 忽略权限错误等
        return []
    files = [f for f in os.listdir(directory) if f.endswith('.pcd')]
    return files


# --- 辅助函数：生成随机颜色 ---
def generate_hex_colors(n):
    colors = []
    # 使用一组预设的高对比度颜色，如果不够再随机生成
    preset_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
        "#F7DC6F", "#BB8FCE", "#F1948A", "#82E0AA", "#85C1E9"
    ]
    for i in range(n):
        if i < len(preset_colors):
            colors.append(preset_colors[i])
        else:
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            colors.append(color)
    return colors


# --- 侧边栏：参数配置 ---
st.sidebar.title("🎛️ 参数配置控制台")
st.sidebar.info("⚠️ 注意：所有距离/坐标参数单位均为 米 (m)")

# 1. 文件选择
st.sidebar.subheader("1. 数据源选择")
pcd_dir = "./map_900m"
pcd_files = get_pcd_files(pcd_dir)
if not pcd_files:
    st.sidebar.warning(f"文件夹 {pcd_dir} 中没有找到 .pcd 文件")
selected_file = st.sidebar.selectbox("选择点云文件", pcd_files)

# 2. 预处理参数
with st.sidebar.expander("2. 预处理 & 裁剪 (Preprocessing)", expanded=False):
    voxel_size = st.slider("体素降采样 (m)", 0.01, 1.0, 0.5, 0.05)

    st.caption("裁剪范围 Min Bound (m)")
    col1, col2, col3 = st.columns(3)
    min_x = col1.number_input("Min X", value=-1265.0)
    min_y = col2.number_input("Min Y", value=-50.0)
    min_z = col3.number_input("Min Z", value=-10.0)

    st.caption("裁剪范围 Max Bound (m)")
    col4, col5, col6 = st.columns(3)
    max_x = col4.number_input("Max X", value=10.0)
    max_y = col5.number_input("Max Y", value=20.0)
    max_z = col6.number_input("Max Z", value=14.0)

# 3. 滤波参数
with st.sidebar.expander("3. 滤波与去噪 (Filtering)", expanded=False):
    normal_z_threshold = st.slider("法线 Z 阈值 (坡度)", 0.0, 1.0, 0.2, 0.05, help="保留 Z 分量大于此值的点，越小越严格")
    outlier_nb = st.number_input("离群点邻居数", value=20)
    outlier_std = st.number_input("离群点标准差倍数", value=1.0)

# 4. 聚类/分段参数 (更新)
with st.sidebar.expander("4. 聚类/分段分析 (Segmentation)", expanded=True):
    # 选择聚类模式
    clustering_method = st.radio(
        "选择分割方法",
        ("DBSCAN 欧几里得聚类 (自动)", "Grid Sectioning 线性切片 (固定步长)"),
        help="DBSCAN适合分离不连续的独立煤堆；线性切片适合计算连续长条形仓库的区间体积。"
    )

    if "DBSCAN" in clustering_method:
        cluster_eps = st.slider("聚类半径 Eps (m)", 0.5, 10.0, 5.0, 0.5)
        cluster_min_points = st.number_input("最小点数", value=50)
        section_step = 100.0  # 默认值，不使用
    else:
        # Grid Sectioning 模式
        section_step = st.number_input("切片步长 (m)", value=100.0, min_value=10.0, step=10.0,
                                       help="仓库长度方向每隔多少米计算一次体积")
        cluster_eps = 5.0  # 默认值
        cluster_min_points = 50  # 默认值
        st.info(f"将从 X={min_x}m 开始，每 {section_step}m 计算一次体积，直到 X={max_x}m")

# 5. 体积计算参数
with st.sidebar.expander("5. 体积计算 (Calculation)", expanded=True):
    volume_scale = st.slider("变形补偿系数", 0.0, 1.0, 0.2, 0.1, help="0.0 表示不补偿")
    max_edge = st.slider("最大三角形边长 (m)", 0.5, 10.0, 2.5, 0.1, help="防止边缘产生凸包效应")

# --- 主页面逻辑 ---
st.title("⛏️ 室内煤堆体积智能盘点系统")
st.markdown("基于 **UAV-LiDAR SLAM** 与 **Streamlit** 的实时计算平台")

if selected_file:
    file_path = os.path.join(pcd_dir, selected_file)

    # 组装 Config 对象
    config = ProcessingConfig(
        voxel_size=voxel_size,
        min_bound=(min_x, min_y, min_z),
        max_bound=(max_x, max_y, max_z),
        normal_z_threshold=normal_z_threshold,
        outlier_nb_neighbors=outlier_nb,
        outlier_std_ratio=outlier_std,
        cluster_eps=cluster_eps,
        cluster_min_points=cluster_min_points,
        section_step=section_step,  # 传入切片参数
        volume_deformation_scale=volume_scale,
        max_triangle_edge=max_edge
    )

    # 两个主要按钮
    col_btn1, col_btn2 = st.columns([1, 1])
    start_calc = col_btn1.button("🚀 开始计算 (Run Calculation)", type="primary")
    visualize_3d = col_btn2.button("👀 打开3D视图 (Open 3D Viewer)")

    if start_calc or visualize_3d:
        with st.spinner("正在加载点云并执行核心算法..."):
            try:
                # 1. 实例化处理流
                processor = PointCloudFilter(file_path, config)

                # 2. 执行流水线
                processor.ransac()
                processor.pass_through_filter()
                processor.process_pipeline()

                # [逻辑分支] 根据用户选择调用不同的聚类方法
                if "DBSCAN" in clustering_method:
                    clusters = processor.euclidean_clustering(processor.re_filtered_pcd)
                    prefix = "Cluster"
                else:
                    # 调用新加的切片方法
                    clusters = processor.grid_section_clustering(processor.re_filtered_pcd)
                    prefix = "Section"

                # 3. 体积计算
                vol_calc = CoalVolumeCalculator(config)

                results_data = []
                total_vol = 0.0

                # 颜色生成 (为每个簇分配固定颜色)
                cluster_colors_hex = generate_hex_colors(len(clusters))
                # Open3D 需要 0-1 的 RGB
                cluster_colors_rgb = [[int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255] for h in
                                      cluster_colors_hex]

                # 拟合地面
                ground_status = "❌ 未检测到地面"
                if processor.ground_points_np is not None and len(processor.ground_points_np) > 0:
                    vol_calc.fit_ground_surface(processor.ground_points_np)
                    ground_status = "✅ 地面拟合成功"

                    # 遍历簇计算
                    for i, cluster in enumerate(clusters):
                        pts = np.asarray(cluster.points)
                        try:
                            vol = vol_calc.calculate_volume(pts)
                        except Exception:
                            vol = 0.0

                        total_vol += vol

                        # 命名逻辑
                        name = f"{prefix} {i + 1}"
                        # 如果是 Grid 模式，显示具体的米数区间
                        if "Grid" in clustering_method:
                            start_dist = min_x + i * section_step
                            end_dist = start_dist + section_step
                            name = f"{start_dist:.0f}m - {end_dist:.0f}m"

                        results_data.append({
                            "ID": name,
                            "点云数量": len(pts),
                            "体积 (m³)": round(vol, 3),
                            "Color": cluster_colors_hex[i]  # 存储颜色用于显示
                        })

                        # 给点云上色
                        cluster.paint_uniform_color(cluster_colors_rgb[i])

                # --- 结果展示区 ---
                st.divider()
                st.subheader("📊 计算结果报告")

                # 1. 指标卡片
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("总库存体积", f"{total_vol:.2f} m³", delta=f"{len(clusters)} 个分区")
                m_col2.metric("地面拟合状态", ground_status)
                m_col3.metric("处理点云数", f"{len(processor.pcd.points)} -> {len(processor.re_filtered_pcd.points)}")

                # 2. [新增] 体积比例示意表 (Visual Bar)
                if total_vol > 0 and results_data:
                    st.write("#### 🧱 体积分布示意图 (Volume Distribution)")

                    # 构建 HTML 字符串
                    bar_html = '<div class="volume-bar-container">'
                    for idx, res in enumerate(results_data):
                        vol = res["体积 (m³)"]
                        if vol > 0:
                            percent = (vol / total_vol) * 100
                            # 只有宽度足够(>5%)才显示文字，避免拥挤
                            label = f"{res['ID']}" if percent > 5 else ""
                            color = res["Color"]
                            # title 属性用于鼠标悬停显示详细信息
                            bar_html += f'<div class="volume-segment" style="width: {percent}%; background-color: {color};" title="{res["ID"]}: {vol} m³ ({percent:.1f}%)">{label}</div>'
                    bar_html += '</div>'

                    st.markdown(bar_html, unsafe_allow_html=True)
                    # 添加图例说明
                    st.caption(
                        "🎨 不同颜色代表不同的煤堆或分段区间，长度代表其占总体积的比例。🖱️ 鼠标悬停在色块上可查看详细数值。")

                # 3. 详细表格
                if results_data:
                    # 为了表格美观，隐藏 Color 列
                    df_display = pd.DataFrame(results_data).drop(columns=["Color"])
                    st.dataframe(df_display, use_container_width=True)
                else:
                    st.warning("未检测到有效煤堆聚类，请调整聚类参数或裁剪范围。")

                # --- 3D 可视化逻辑 ---
                if visualize_3d:
                    st.toast("正在启动原生 Open3D 窗口...", icon="🖥️")
                    vis_list = []
                    # 地面
                    if processor.ground_points_np is not None and len(processor.ground_points_np) > 0:
                        ground_pcd = o3d.geometry.PointCloud()
                        ground_pcd.points = o3d.utility.Vector3dVector(processor.ground_points_np)
                        ground_pcd.paint_uniform_color([0.5, 0.5, 0.5])
                        vis_list.append(ground_pcd)
                    # 煤堆 (已上色)
                    if clusters:
                        vis_list.extend(clusters)
                    elif processor.re_filtered_pcd:
                        vis_list.append(processor.re_filtered_pcd)
                    # 坐标轴
                    vis_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0]))

                    o3d.visualization.draw_geometries(vis_list, window_name="Result Visualization (Native)")

            except Exception as e:
                st.error(f"算法执行出错: {str(e)}")
                st.exception(e)

else:
    st.info("请先在左侧侧边栏上传或选择一个 PCD 点云文件。")