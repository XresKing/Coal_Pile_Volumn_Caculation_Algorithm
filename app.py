import streamlit as st
import os
import numpy as np
import open3d as o3d
import pandas as pd
import random
import json
from datetime import datetime
from Preprocess import PointCloudFilter, CoalVolumeCalculator, ProcessingConfig

# --- 核心修改：锁定随机种子，消除算法波动 ---
np.random.seed(42)
random.seed(42)

# --- 全局配置 ---
HISTORY_DIR = "./volume"
HISTORY_FILE = os.path.join(HISTORY_DIR, "inventory_history.csv")

# --- 页面配置 ---
st.set_page_config(
    page_title="煤堆库存智能管理系统",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 样式优化 (CSS) ---
st.markdown("""
    <style>
    /* 卡片样式 */
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-title {
        color: #666;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #2E7D32;
        font-size: 26px;
        font-weight: bold;
    }
    .metric-time {
        font-size: 12px;
        color: #999;
        margin-top: 5px;
    }

    /* 进度条容器 */
    .volume-bar-container {
        display: flex;
        width: 100%;
        height: 25px;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 8px;
        background-color: #f0f0f0;
    }
    .volume-segment {
        height: 100%;
        display: flex;
        align-items: center;
        justify_content: center;
        color: white;
        font-size: 10px;
        transition: width 0.3s ease;
        text-shadow: 0 0 2px rgba(0,0,0,0.5);
        cursor: help;
    }

    /* 自定义按钮样式 */
    div.stButton > button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# --- 数据管理函数 ---
def init_storage():
    """初始化存储目录和文件"""
    if not os.path.exists(HISTORY_DIR):
        try:
            os.makedirs(HISTORY_DIR)
        except OSError:
            pass
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=[
            "timestamp", "warehouse_name", "total_volume",
            "pcd_file", "segment_data", "config_json"
        ])
        df.to_csv(HISTORY_FILE, index=False)


def load_history():
    """加载历史数据"""
    init_storage()
    try:
        df = pd.read_csv(HISTORY_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        # 如果文件损坏或为空，返回空DataFrame结构
        return pd.DataFrame(
            columns=["timestamp", "warehouse_name", "total_volume", "pcd_file", "segment_data", "config_json"])


def save_record(warehouse_name, total_vol, pcd_file, segments, config):
    """保存计算记录"""
    init_storage()
    config_dict = config.__dict__
    new_record = {
        "timestamp": datetime.now(),
        "warehouse_name": warehouse_name,
        "total_volume": float(total_vol),
        "pcd_file": pcd_file,
        "segment_data": json.dumps(segments),
        "config_json": json.dumps(config_dict)
    }
    df = pd.DataFrame([new_record])
    df.to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)
    return new_record


def delete_warehouse_data(wh_name):
    """删除指定仓库的所有数据"""
    df = load_history()
    if not df.empty:
        df_new = df[df['warehouse_name'] != wh_name]
        df_new.to_csv(HISTORY_FILE, index=False)
        return True
    return False


def update_volume_record(wh_name, timestamp_input, new_vol):
    """
    更新特定记录的体积
    修复：使用严格的字符串格式化进行比对，避免时间戳精度问题
    """
    df = load_history()
    if not df.empty:
        # 将输入的时间转换为统一的字符串格式
        target_ts_str = pd.to_datetime(timestamp_input).strftime('%Y-%m-%d %H:%M:%S')

        # 将 DataFrame 中的时间列也转换为统一格式进行比对
        df_ts_strs = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        mask = (df['warehouse_name'] == wh_name) & (df_ts_strs == target_ts_str)

        if mask.any():
            df.loc[mask, 'total_volume'] = float(new_vol)
            df.to_csv(HISTORY_FILE, index=False)
            return True
    return False


def get_pcd_files(directory="./map_900m"):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except:
            pass
        return []
    return [f for f in os.listdir(directory) if f.endswith('.pcd')]


def generate_hex_colors(n):
    """生成十六进制颜色列表 (无#前缀问题)"""
    preset_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F", "#BB8FCE"]
    colors = []
    for i in range(n):
        if i < len(preset_colors):
            colors.append(preset_colors[i])
        else:
            # 修复：使用 {:06x} 而不是 {:#06x}，避免产生 0x 前缀
            colors.append("#{:06x}".format(random.randint(0, 0xFFFFFF)))
    return colors


# ==========================================
# 页面逻辑
# ==========================================

# 初始化 Session State 用于总览页面的交互
if 'selected_warehouse_overview' not in st.session_state:
    st.session_state['selected_warehouse_overview'] = None

# 侧边栏导航
st.sidebar.title("🏭 煤堆库存管理")
app_mode = st.sidebar.radio("功能菜单", ["📊 仓库总览 (Overview)", "🧮 新盘点计算 (Calculator)"])

# 加载历史数据
df_history = load_history()

# -----------------------------------------------------------------------------
# 页面 1: 仓库总览 (Overview)
# -----------------------------------------------------------------------------
if app_mode == "📊 仓库总览 (Overview)":
    st.title("📊 仓库库存总览")
    st.markdown("查看所有仓库的最新状态，点击卡片下方按钮查看历史详情。")
    st.divider()

    if df_history.empty:
        st.info("👋 暂无数据。请切换到 **“新盘点计算”** 菜单，上传点云并计算第一个仓库的体积。")
    else:
        # 获取每个仓库的最新一条记录
        latest_df = df_history.sort_values('timestamp').groupby('warehouse_name').tail(1).sort_values('total_volume',
                                                                                                      ascending=False)

        # --- 仓库卡片网格 ---
        global_max_vol = df_history['total_volume'].max() * 1.1 if not df_history.empty else 10000

        cols = st.columns(3)
        for idx, (_, row) in enumerate(latest_df.iterrows()):
            wh_name = row['warehouse_name']
            current_vol = row['total_volume']
            update_time = row['timestamp'].strftime('%Y-%m-%d %H:%M')
            pct = min((current_vol / global_max_vol) * 100, 100) if global_max_vol > 0 else 0

            with cols[idx % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🏢 {wh_name}</div>
                    <div class="metric-value">{current_vol:,.2f} <span style="font-size:14px;color:#666;">m³</span></div>
                    <div style="background-color:#eee;height:6px;border-radius:3px;margin:8px 0;overflow:hidden;">
                        <div style="background-color:#4CAF50;width:{pct}%;height:100%;"></div>
                    </div>
                    <div class="metric-time">🕒 {update_time}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"查看详情 🔍", key=f"btn_{wh_name}"):
                    st.session_state['selected_warehouse_overview'] = wh_name

        # --- 详情展示区域 ---
        selected_wh = st.session_state['selected_warehouse_overview']

        # 检查选中仓库是否依然存在
        if selected_wh and selected_wh in df_history['warehouse_name'].values:
            st.divider()

            col_head1, col_head2 = st.columns([3, 1])
            with col_head1:
                st.subheader(f"📈 {selected_wh} - 历史与趋势")

            # 过滤该仓库数据
            wh_data = df_history[df_history['warehouse_name'] == selected_wh].sort_values('timestamp')

            # 核心指标
            if len(wh_data) >= 2:
                last_vol = wh_data.iloc[-1]['total_volume']
                prev_vol = wh_data.iloc[-2]['total_volume']
                delta = last_vol - prev_vol
                delta_str = f"{delta:+.2f} m³"
            else:
                delta_str = "首次记录"

            m1, m2 = st.columns(2)
            m1.metric("当前体积", f"{wh_data.iloc[-1]['total_volume']:.2f} m³", delta=delta_str)
            m2.metric("记录次数", f"{len(wh_data)} 次")

            # 图表
            chart_data = wh_data.set_index('timestamp')[['total_volume']]
            st.line_chart(chart_data, height=300)

            # --- 🛠️ 管理与操作区 ---
            st.markdown("### 🛠️ 数据管理与修正")

            tab1, tab2, tab3 = st.tabs(["📄 详细记录表", "📝 手动修正体积", "🗑️ 危险操作"])

            # Tab 1: 历史表格
            with tab1:
                display_cols = ['timestamp', 'total_volume', 'pcd_file']
                st.dataframe(
                    wh_data[display_cols].style.format({'total_volume': '{:.2f}'}),
                    use_container_width=True
                )

                st.caption("🔍 选择记录查看当时的计算参数")
                selected_record_idx = st.selectbox(
                    "选择一条记录:",
                    wh_data.index,
                    format_func=lambda x: wh_data.loc[x, 'timestamp'].strftime('%Y-%m-%d %H:%M'),
                    key="config_select"
                )
                if selected_record_idx is not None:
                    config_str = wh_data.loc[selected_record_idx, 'config_json']
                    with st.expander("查看参数详情"):
                        try:
                            st.json(json.loads(config_str))
                        except:
                            st.text("配置解析失败")

            # Tab 2: 修改体积
            with tab2:
                st.info("如发现计算误差，可在此手动修正历史记录中的体积数值。")
                col_edit1, col_edit2 = st.columns(2)

                with col_edit1:
                    # 获取格式化后的时间字符串列表
                    time_options = wh_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                    edit_timestamp = st.selectbox(
                        "选择要修正的时间点:",
                        time_options,
                        key="edit_ts_select"
                    )

                if edit_timestamp:
                    # --- 修复核心：安全地筛选记录 ---
                    # 1. 构造一个临时的字符串列进行精确比对
                    wh_data['ts_str'] = wh_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    # 2. 筛选
                    matched_records = wh_data[wh_data['ts_str'] == edit_timestamp]

                    if not matched_records.empty:
                        current_record = matched_records.iloc[0]
                        old_vol = current_record['total_volume']

                        with col_edit2:
                            new_vol_input = st.number_input(
                                f"修正体积 (原值: {old_vol:.2f})",
                                value=float(old_vol),
                                step=10.0,
                                format="%.2f"
                            )

                        if st.button("💾 保存修正", type="primary"):
                            if update_volume_record(selected_wh, current_record['timestamp'], new_vol_input):
                                st.success(f"已将 {edit_timestamp} 的体积修正为 {new_vol_input} m³")
                                st.rerun()
                            else:
                                st.error("修正失败，未找到记录。")
                    else:
                        st.error("未找到对应时间点的记录，请刷新页面重试。")

            # Tab 3: 删除仓库
            with tab3:
                st.warning(f"⚠️ 警告：此操作将永久删除 **{selected_wh}** 的所有历史数据，且不可恢复！")

                col_del1, col_del2 = st.columns([3, 1])
                with col_del1:
                    confirm_check = st.checkbox(f"我已知晓后果，确认删除 {selected_wh}")

                with col_del2:
                    if st.button("🔴 彻底删除", disabled=not confirm_check, type="primary"):
                        if delete_warehouse_data(selected_wh):
                            st.toast(f"仓库 {selected_wh} 已删除", icon="🗑️")
                            st.session_state['selected_warehouse_overview'] = None
                            st.rerun()
                        else:
                            st.error("删除失败，请重试。")

        elif selected_wh:
            st.warning("该仓库数据已不存在。")


# -----------------------------------------------------------------------------
# 页面 2: 新盘点计算 (Calculator)
# -----------------------------------------------------------------------------
elif app_mode == "🧮 新盘点计算 (Calculator)":
    st.title("🧮 新库存盘点")

    # 侧边栏配置区
    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ 盘点设置")

    # 1. 仓库选择 (混合输入)
    existing_warehouses = []
    if not df_history.empty:
        existing_warehouses = df_history['warehouse_name'].unique().tolist()

    warehouse_source = st.sidebar.radio(
        "仓库选择模式",
        ["📂 选择现有仓库", "➕ 新建仓库"],
        index=0 if existing_warehouses else 1
    )

    warehouse_name = ""
    if warehouse_source == "📂 选择现有仓库":
        if existing_warehouses:
            warehouse_name = st.sidebar.selectbox("选择目标仓库", existing_warehouses)
        else:
            st.sidebar.warning("暂无历史仓库，请切换到新建模式。")
    else:
        warehouse_name = st.sidebar.text_input("输入新仓库名称", placeholder="例如: 三号煤棚")

    # 2. 文件选择
    pcd_dir = "./map_900m"
    pcd_files = get_pcd_files(pcd_dir)

    if not pcd_files:
        selected_file = None
        st.error(f"目录 {pcd_dir} 中未找到 PCD 文件。")
    else:
        selected_file = st.sidebar.selectbox("📂 选择点云文件", pcd_files)

    # 3. 参数配置
    with st.sidebar.expander("预处理 & 裁剪", expanded=False):
        voxel_size = st.slider("体素降采样 (m)", 0.01, 1.0, 0.5, 0.05)
        st.caption("裁剪范围 (m)")
        c1, c2 = st.columns(2)
        min_x = c1.number_input("Min X", -1265.0)
        max_x = c2.number_input("Max X", 10.0)
        min_y = c1.number_input("Min Y", -50.0)
        max_y = c2.number_input("Max Y", 20.0)
        min_z = c1.number_input("Min Z", -10.0)
        max_z = c2.number_input("Max Z", 14.0)

    with st.sidebar.expander("分割/聚类方法", expanded=True):
        clustering_method = st.radio("选择方法", ("Grid Sectioning 线性切片", "DBSCAN 欧几里得聚类"), index=0)
        if "Grid" in clustering_method:
            section_step = st.number_input("📏 切片步长 (m)", value=100.0, step=10.0)
            cluster_eps, cluster_min_points = 5.0, 50
        else:
            cluster_eps = st.slider("聚类半径 (m)", 0.5, 10.0, 5.0)
            cluster_min_points = st.number_input("最小点数", 50)
            section_step = 100.0

    with st.sidebar.expander("算法微调", expanded=False):
        normal_z_threshold = st.slider("坡度阈值", 0.0, 1.0, 0.2)
        volume_scale = st.slider("变形补偿系数", 0.0, 0.5, 0.2)
        max_edge = st.slider("最大三角边长", 1.0, 5.0, 2.5)

    # --- 主操作区 ---
    if not warehouse_name:
        st.info("👈 请在侧边栏输入或选择 **仓库名称**。")
    elif selected_file:
        st.info(f"准备就绪: 将对 **{warehouse_name}** 使用文件 **{selected_file}** 进行计算。")

        c1, c2 = st.columns([1, 1])
        start_btn = c1.button("🚀 开始计算并存档", type="primary", use_container_width=True)
        view_btn = c2.button("👀 仅 3D 预览 (不保存)", use_container_width=True)

        if start_btn or view_btn:
            # 构建配置
            config = ProcessingConfig(
                voxel_size=voxel_size,
                min_bound=(min_x, min_y, min_z),
                max_bound=(max_x, max_y, max_z),
                normal_z_threshold=normal_z_threshold,
                outlier_nb_neighbors=20,
                outlier_std_ratio=1.0,
                cluster_eps=cluster_eps,
                cluster_min_points=cluster_min_points,
                section_step=section_step,
                volume_deformation_scale=volume_scale,
                max_triangle_edge=max_edge
            )
            file_path = os.path.join(pcd_dir, selected_file)

            with st.spinner("正在执行算法 (加载 -> 滤波 -> 聚类 -> 积分)..."):
                try:
                    # 1. 预处理
                    processor = PointCloudFilter(file_path, config)
                    processor.ransac()
                    processor.pass_through_filter()
                    processor.process_pipeline()

                    # 2. 聚类
                    if "Grid" in clustering_method:
                        clusters = processor.grid_section_clustering(processor.re_filtered_pcd)
                        prefix = "Section"
                    else:
                        clusters = processor.euclidean_clustering(processor.re_filtered_pcd)
                        prefix = "Cluster"

                    # 3. 体积计算
                    vol_calc = CoalVolumeCalculator(config)
                    total_vol = 0.0
                    segments_info = []

                    has_ground = False
                    if processor.ground_points_np is not None and len(processor.ground_points_np) > 0:
                        vol_calc.fit_ground_surface(processor.ground_points_np)
                        has_ground = True

                        cluster_colors_hex = generate_hex_colors(len(clusters))
                        cluster_colors_rgb = [[int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255] for
                                              h in cluster_colors_hex]

                        for i, cluster in enumerate(clusters):
                            pts = np.asarray(cluster.points)
                            try:
                                vol = vol_calc.calculate_volume(pts)
                            except:
                                vol = 0.0
                            total_vol += vol

                            seg_id = f"{prefix} {i + 1}"
                            if "Grid" in clustering_method:
                                start_d = min_x + i * section_step
                                end_d = start_d + section_step
                                seg_id = f"{start_d:.0f}-{end_d:.0f}m"

                            segments_info.append({"id": seg_id, "volume": round(vol, 3), "points": len(pts),
                                                  "color": cluster_colors_hex[i]})
                            cluster.paint_uniform_color(cluster_colors_rgb[i])

                    # 4. 结果展示
                    if has_ground:
                        st.success(f"计算完成！ **{warehouse_name}** 总库存: **{total_vol:.2f} m³**")

                        if start_btn:
                            save_record(warehouse_name, total_vol, selected_file, segments_info, config)
                            st.toast(f"✅ 已保存至历史记录", icon="💾")

                        if total_vol > 0:
                            bar_html = '<div class="volume-bar-container">'
                            for seg in segments_info:
                                vol = seg['volume']
                                if vol > 0:
                                    pct = (vol / total_vol) * 100
                                    label = seg['id'] if pct > 8 else ""
                                    bar_html += f'<div class="volume-segment" style="width:{pct}%;background-color:{seg["color"]};" title="{seg["id"]}: {vol}m³">{label}</div>'
                            bar_html += '</div>'
                            st.markdown(bar_html, unsafe_allow_html=True)
                            st.caption("各分段/煤堆体积占比示意图")

                        if segments_info:
                            df_seg = pd.DataFrame(segments_info)[['id', 'volume', 'points']]
                            st.dataframe(df_seg, use_container_width=True)

                        if view_btn:
                            vis_list = []
                            if processor.ground_points_np is not None:
                                g_pcd = o3d.geometry.PointCloud()
                                g_pcd.points = o3d.utility.Vector3dVector(processor.ground_points_np)
                                g_pcd.paint_uniform_color([0.5, 0.5, 0.5])
                                vis_list.append(g_pcd)
                            vis_list.extend(clusters)
                            vis_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0))
                            st.toast("正在打开 Open3D 窗口...", icon="🖥️")
                            o3d.visualization.draw_geometries(vis_list, window_name=f"Inventory: {warehouse_name}")
                    else:
                        st.error("无法拟合地面，请检查数据或调整裁剪范围。")
                except Exception as e:
                    st.error(f"处理出错: {e}")