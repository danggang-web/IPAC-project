import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy import signal
from scipy.signal import tf2zpk, zpk2tf, step, bode
import control as ctrl
import math

# --- 页面基础设置 ---
st.set_page_config(
    page_title="水箱液位控制系统仿真平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS美化
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
    }
    .sidebar-header {
        font-size: 18px;
        font-weight: bold;
        color: #2e86ab;
        margin-bottom: 10px;
    }
    .expander-header {
        font-size: 16px;
        font-weight: bold;
    }
    .toggle-button {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 标题与说明 ---
st.title("�水箱液位控制系统仿真平台")
st.markdown("太原理工大学 IPAC 实验室 | 过程控制实验平台")
st.divider()

# --- 侧边栏：核心配置区 ---
with st.sidebar:
    # 1. 水箱阶数切换
    st.markdown('<div class="sidebar-header">📦 水箱系统配置</div>', unsafe_allow_html=True)
    tank_type = st.selectbox(
        "水箱类型",
        options=["单水箱（一阶）", "双水箱（二阶）"],
        index=0,
        help="单水箱为一阶惯性系统，双水箱为二阶惯性系统"
    )

    # 2. 控制算法切换
    st.markdown('<div class="sidebar-header">🔧 控制算法选择</div>', unsafe_allow_html=True)
    control_alg = st.selectbox(
        "控制算法",
        options=["经典PID", "增量式PID", "模糊PID"],
        index=0,
        help="不同PID变种算法的控制效果对比"
    )

    # 3. 整定模块
    st.markdown('<div class="sidebar-header">🎯 PID参数整定</div>', unsafe_allow_html=True)
    tuning_method = st.selectbox(
        "整定方法",
        options=["手动整定", "Ziegler-Nichols整定", "临界比例度法"],
        index=0
    )
    auto_tune = st.button("📊 自动整定参数", disabled=(tuning_method == "手动整定"))

    # 4. 系统参数（根据水箱阶数动态显示）
    st.markdown('<div class="sidebar-header">⚙️ 系统参数</div>', unsafe_allow_html=True)
    K = st.slider("系统总增益 K", 0.1, 10.0, 2.0, 0.1, help="液位对输入流量的增益系数")
    L = st.slider("纯滞后时间 τ (s)", 0.0, 20.0, 0.0, 0.5, help="管道传输滞后时间")

    if tank_type == "单水箱（一阶）":
        T1 = st.slider("时间常数 T1 (s)", 1.0, 50.0, 10.0, 1.0, help="单水箱液位响应时间常数")
        T2 = 0  # 二阶参数置0
    else:
        T1 = st.slider("第一水箱时间常数 T1 (s)", 1.0, 50.0, 10.0, 1.0)
        T2 = st.slider("第二水箱时间常数 T2 (s)", 1.0, 50.0, 15.0, 1.0)

    # 5. PID参数（整定后自动更新）
    st.markdown('<div class="sidebar-header">📐 PID控制器参数</div>', unsafe_allow_html=True)
    if 'kp_auto' not in st.session_state:
        st.session_state.kp_auto = 5.0
        st.session_state.ti_auto = 15.0
        st.session_state.td_auto = 2.0

    Kp = st.slider("比例系数 Kp", 0.1, 20.0, st.session_state.kp_auto, 0.1)
    Ti = st.slider("积分时间 Ti (s)", 0.1, 60.0, st.session_state.ti_auto, 1.0)
    Td = st.slider("微分时间 Td (s)", 0.0, 20.0, st.session_state.td_auto, 0.1)

    # 滞后环节处理
    use_pade = st.checkbox("使用Pade近似处理纯滞后", value=True, help="τ>0时启用Pade近似分析零极点")


# --- 核心函数定义 ---
## 1. PID参数自整定函数
def tune_pid_params(tank_type, K, T1, T2, L, method):
    """
    根据Ziegler-Nichols或临界比例度法整定PID参数
    :param tank_type: 水箱类型（一阶/二阶）
    :param K: 系统增益
    :param T1/T2: 时间常数
    :param L: 滞后时间
    :param method: 整定方法
    :return: Kp, Ti, Td
    """
    if tank_type == "单水箱（一阶）":
        # 一阶系统 K/(Ts+1)
        T = T1
        if method == "Ziegler-Nichols整定":
            Kp = 1.2 * T / (K * L) if L > 0 else 0.6 / K
            Ti = 2 * L if L > 0 else T / 2
            Td = L / 2 if L > 0 else T / 8
        else:  # 临界比例度法
            Kp = 0.6 * (T / (K * L)) if L > 0 else 0.5 / K
            Ti = T if L > 0 else T / 1.5
            Td = 0.125 * T
    else:
        # 二阶系统 K/((T1s+1)(T2s+1))
        T_avg = (T1 + T2) / 2
        if method == "Ziegler-Nichols整定":
            Kp = 1.4 * T_avg / (K * L) if L > 0 else 0.7 / K
            Ti = 1.5 * L if L > 0 else T_avg
            Td = 0.375 * L if L > 0 else T_avg / 6
        else:  # 临界比例度法
            Kp = 0.7 * (T_avg / (K * L)) if L > 0 else 0.6 / K
            Ti = 1.2 * T_avg if L > 0 else T_avg
            Td = 0.25 * T_avg

    # 参数限幅
    Kp = np.clip(Kp, 0.1, 20.0)
    Ti = np.clip(Ti, 0.1, 60.0)
    Td = np.clip(Td, 0.0, 20.0)

    return round(Kp, 1), round(Ti, 1), round(Td, 1)


## 2. 构建系统传递函数
def build_system_model(tank_type, K, T1, T2, L, Kp, Ti, Td, control_alg, use_pade=True):
    """
    构建不同水箱+不同控制算法的传递函数
    """
    # 1. 构建水箱对象传递函数
    if tank_type == "单水箱（一阶）":
        # 一阶：G(s) = K/(T1s + 1)
        num_G = [K]
        den_G = [T1, 1]
    else:
        # 二阶：G(s) = K/((T1s+1)(T2s+1))
        num_G = [K]
        den_G = np.convolve([T1, 1], [T2, 1])

    # 2. 构建控制器传递函数
    if control_alg == "经典PID":
        # 经典PID：Gc(s) = Kp*(1 + 1/(Ti*s) + Td*s) = Kp*(Td*s² + s + 1/Ti)/s
        num_PID = [Kp * Td, Kp, Kp / Ti]
        den_PID = [1, 0]
    elif control_alg == "增量式PID":
        # 增量式PID离散化等效连续域近似
        Ts = 1.0  # 采样时间
        num_PID = [Kp * Td / Ts, Kp * (1 + Td / Ts), Kp * (1 / Ti * Ts - 1)]
        den_PID = [1, -1]
    else:  # 模糊PID（简化为带参数修正的PID）
        # 模糊PID：在经典PID基础上增加修正系数
        kp_fuzzy = 1.2  # 模糊修正系数
        ti_fuzzy = 0.8
        td_fuzzy = 1.1
        num_PID = [Kp * Td * td_fuzzy, Kp * kp_fuzzy, (Kp / Ti) * ti_fuzzy]
        den_PID = [1, 0]

    # 3. 纯滞后环节Pade近似
    if L > 0 and use_pade:
        num_delay = [-L / 2, 1]
        den_delay = [L / 2, 1]
        num_G_delay = np.convolve(num_G, num_delay)
        den_G_delay = np.convolve(den_G, den_delay)
    else:
        num_G_delay = num_G
        den_G_delay = den_G

    # 4. 开环/闭环传递函数
    num_open = np.convolve(num_PID, num_G_delay)
    den_open = np.convolve(den_PID, den_G_delay)

    num_closed = num_open
    den_closed = np.polyadd(den_open, num_open)

    # 转换为control对象
    sys_open = ctrl.TransferFunction(num_open, den_open)
    sys_closed = ctrl.TransferFunction(num_closed, den_closed)

    # 计算零极点
    z_open, p_open, k_open = tf2zpk(num_open, den_open)
    z_closed, p_closed, k_closed = tf2zpk(num_closed, den_closed)

    return {
        "open_loop": sys_open,
        "closed_loop": sys_closed,
        "z_open": z_open,
        "p_open": p_open,
        "z_closed": z_closed,
        "p_closed": p_closed,
        "k_open": k_open,
        "k_closed": k_closed,
        "control_alg": control_alg,
        "tank_type": tank_type
    }


## 3. 性能指标计算
def calculate_performance(t, y, setpoint=1.0):
    """计算阶跃响应性能指标"""
    y_norm = y / setpoint if setpoint != 0 else y

    # 上升时间（0.1→0.9）
    idx_10 = np.where(y_norm >= 0.1)[0][0] if np.any(y_norm >= 0.1) else 0
    idx_90 = np.where(y_norm >= 0.9)[0][0] if np.any(y_norm >= 0.9) else len(t) - 1
    rise_time = t[idx_90] - t[idx_10]

    # 超调量
    max_y = np.max(y)
    overshoot = ((max_y - setpoint) / setpoint * 100) if setpoint != 0 else 0
    overshoot = max(0, overshoot)

    # 稳态误差
    steady_state = y[-10:] if len(y) >= 10 else y
    steady_error = abs(np.mean(steady_state) - setpoint)

    return {
        "上升时间(s)": round(rise_time, 2),
        "超调量(%)": round(overshoot, 2),
        "稳态误差": round(steady_error, 4)
    }


# --- 自动整定逻辑 ---
if auto_tune:
    Kp_tuned, Ti_tuned, Td_tuned = tune_pid_params(tank_type, K, T1, T2, L, tuning_method)
    st.session_state.kp_auto = Kp_tuned
    st.session_state.ti_auto = Ti_tuned
    st.session_state.td_auto = Td_tuned
    st.success(f"✅ 参数整定完成！Kp={Kp_tuned}, Ti={Ti_tuned}, Td={Td_tuned}")
    # 刷新页面应用新参数
    st.rerun()

# --- 构建系统模型 ---
system_data = build_system_model(tank_type, K, T1, T2, L, Kp, Ti, Td, control_alg, use_pade)

# --- 1. 系统配置信息展示 ---
st.subheader("📋 当前系统配置")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"**水箱类型**: {tank_type}")
with col2:
    st.info(f"**控制算法**: {control_alg}")
with col3:
    st.info(f"**整定方法**: {tuning_method}")

# --- 2. 零极点公式显示模块 ---
st.subheader("📐 零极点分析 (公式形式)")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 开环系统 (控制器+水箱)")
    z_open_str = ", ".join([f"{z:.2f}" for z in system_data["z_open"]]) if len(system_data["z_open"]) > 0 else "无"
    p_open_str = ", ".join([f"{p:.2f}" for p in system_data["p_open"]]) if len(system_data["p_open"]) > 0 else "无"

    st.latex(f"零点(Z): \\{{{z_open_str}\\}}")
    st.latex(f"极点(P): \\{{{p_open_str}\\}}")
    st.latex(
        f"开环传递函数: G_o(s) = {system_data['k_open']:.2f} \\cdot \\frac{{\\prod (s - z_i)}}{{\\prod (s - p_i)}}")

with col2:
    st.markdown("#### 闭环系统")
    z_closed_str = ", ".join([f"{z:.2f}" for z in system_data["z_closed"]]) if len(system_data["z_closed"]) > 0 else "无"
    p_closed_str = ", ".join([f"{p:.2f}" for p in system_data["p_closed"]]) if len(system_data["p_closed"]) > 0 else "无"

    st.latex(f"零点(Z): \\{{{z_closed_str}\\}}")
    st.latex(f"极点(P): \\{{{p_closed_str}\\}}")
    st.latex(f"闭环传递函数: G_{{cl}}(s) = \\frac{{G_o(s)}}{{1 + G_o(s)}}")

st.divider()

# --- 3. 零极点图模块 ---
st.subheader("📈 零极点分布图")
fig_zp = go.Figure()

# 开环零极点
fig_zp.add_trace(go.Scatter(
    x=np.real(system_data["z_open"]),
    y=np.imag(system_data["z_open"]),
    mode='markers',
    name='开环零点 (○)',
    marker=dict(symbol='circle', size=10, color='blue', line=dict(width=2))
))
fig_zp.add_trace(go.Scatter(
    x=np.real(system_data["p_open"]),
    y=np.imag(system_data["p_open"]),
    mode='markers',
    name='开环极点 (×)',
    marker=dict(symbol='x', size=10, color='red', line=dict(width=2))
))

# 闭环零极点
fig_zp.add_trace(go.Scatter(
    x=np.real(system_data["z_closed"]),
    y=np.imag(system_data["z_closed"]),
    mode='markers',
    name='闭环零点 (○)',
    marker=dict(symbol='circle', size=12, color='green', line=dict(width=2), opacity=0.7)
))
fig_zp.add_trace(go.Scatter(
    x=np.real(system_data["p_closed"]),
    y=np.imag(system_data["p_closed"]),
    mode='markers',
    name='闭环极点 (×)',
    marker=dict(symbol='x', size=12, color='orange', line=dict(width=2), opacity=0.7)
))

# 虚轴
fig_zp.add_vline(x=0, line=dict(color='gray', dash='dash'), annotation_text="虚轴 (Re=0)")

fig_zp.update_layout(
    title=f"{tank_type} - {control_alg} 零极点分布图",
    xaxis_title="实部 (Re)",
    yaxis_title="虚部 (Im)",
    height=400,
    showlegend=True,
    xaxis=dict(zeroline=True, zerolinewidth=2),
    yaxis=dict(zeroline=True, zerolinewidth=2)
)
st.plotly_chart(fig_zp, use_container_width=True)

st.divider()

# --- 4. 阶跃响应模块 ---
st.subheader("📊 阶跃响应曲线")

# 计算阶跃响应
t_step, y_step = ctrl.step_response(system_data["closed_loop"], T=np.linspace(0, 100, 1000))
perf_metrics = calculate_performance(t_step, y_step)

# 绘制阶跃响应图
fig_step = go.Figure()
fig_step.add_trace(go.Scatter(
    x=t_step, y=y_step,
    mode='lines', name='液位响应',
    line=dict(color='#1f77b4', width=2)
))
fig_step.add_hline(y=1.0, line=dict(color='red', dash='dash'), annotation_text="设定值")

# 标注关键指标
fig_step.add_annotation(
    x=perf_metrics["上升时间(s)"], y=0.9,
    text=f"上升时间: {perf_metrics['上升时间(s)']}s",
    showarrow=True, arrowhead=2
)
if perf_metrics["超调量(%)"] > 0:
    max_idx = np.argmax(y_step)
    fig_step.add_annotation(
        x=t_step[max_idx], y=y_step[max_idx],
        text=f"超调量: {perf_metrics['超调量(%)']}%",
        showarrow=True, arrowhead=2
    )

fig_step.update_layout(
    title=f"{tank_type} - {control_alg} 闭环阶跃响应",
    xaxis_title="时间 (s)",
    yaxis_title="液位 (归一化)",
    height=400
)
st.plotly_chart(fig_step, use_container_width=True)

# 性能指标展示
st.subheader("🎯 性能指标")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("上升时间", f"{perf_metrics['上升时间(s)']} s")
with col2:
    st.metric("超调量", f"{perf_metrics['超调量(%)']} %")
with col3:
    st.metric("稳态误差", perf_metrics["稳态误差"])

st.divider()

# --- 5. 根轨迹模块 ---
st.subheader("🌐 根轨迹图")
fig_rl = go.Figure()

# 计算根轨迹数据
sys_open = system_data["open_loop"]
rl_x, rl_y, k_vals = ctrl.root_locus(sys_open, plot=False)

# 绘制根轨迹
for i in range(rl_x.shape[0]):
    fig_rl.add_trace(go.Scatter(
        x=rl_x[i], y=rl_y[i],
        mode='lines', name=f'根轨迹{i + 1}',
        line=dict(color='#2ca02c', width=1)
    ))

# 绘制极点起点
fig_rl.add_trace(go.Scatter(
    x=np.real(system_data["p_open"]),
    y=np.imag(system_data["p_open"]),
    mode='markers',
    name='开环极点 (起点)',
    marker=dict(symbol='x', size=10, color='red')
))

# 绘制零点终点
fig_rl.add_trace(go.Scatter(
    x=np.real(system_data["z_open"]),
    y=np.imag(system_data["z_open"]),
    mode='markers',
    name='开环零点 (终点)',
    marker=dict(symbol='circle', size=10, color='blue')
))

# 虚轴
fig_rl.add_vline(x=0, line=dict(color='gray', dash='dash'), annotation_text="虚轴 (Re=0)")

fig_rl.update_layout(
    title=f"{tank_type} - {control_alg} 根轨迹图 (Kp从0→∞)",
    xaxis_title="实部 (Re)",
    yaxis_title="虚部 (Im)",
    height=400,
    showlegend=True
)
st.plotly_chart(fig_rl, use_container_width=True)

st.divider()

# --- 6. 波特图模块 ---
st.subheader("📉 波特图 (频率响应)")
fig_bode = sp.make_subplots(
    rows=2, cols=1,
    subplot_titles=('幅频特性', '相频特性'),
    vertical_spacing=0.1
)

# 计算波特图数据
omega, mag, phase = ctrl.bode(system_data["open_loop"], plot=False)

# 幅频特性
fig_bode.add_trace(go.Scatter(
    x=np.log10(omega), y=20 * np.log10(mag),
    mode='lines', name='幅频',
    line=dict(color='#ff7f0e')
), row=1, col=1)

# 相频特性
fig_bode.add_trace(go.Scatter(
    x=np.log10(omega), y=phase,
    mode='lines', name='相频',
    line=dict(color='#d62728')
), row=2, col=1)

# 标注截止频率
mag_dB = 20 * np.log10(mag)
cutoff_idx = np.where(mag_dB <= 0)[0][0] if np.any(mag_dB <= 0) else -1
if cutoff_idx != -1:
    fig_bode.add_vline(
        x=np.log10(omega[cutoff_idx]),
        row=1, col=1,
        line=dict(color='gray', dash='dot'),
        annotation_text=f"截止频率: {omega[cutoff_idx]:.2f} rad/s"
    )

fig_bode.update_layout(
    title=f"{tank_type} - {control_alg} 波特图",
    height=600,
    showlegend=False
)
fig_bode.update_xaxes(title_text='频率 (log10(rad/s))', row=1, col=1)
fig_bode.update_xaxes(title_text='频率 (log10(rad/s))', row=2, col=1)
fig_bode.update_yaxes(title_text='幅值 (dB)', row=1, col=1)
fig_bode.update_yaxes(title_text='相位 (°)', row=2, col=1)

st.plotly_chart(fig_bode, use_container_width=True)

st.divider()

# --- 稳定性判读说明模块 ---
st.subheader("🔍 系统稳定性判读说明（零极点图与根轨迹）")
st.markdown("""
1. 系统稳定性由 **极点（×）** 决定，零点（○）仅用于分析零极点关系。

2. 所有极点实部 < 0 → 系统稳定；有极点实部 > 0 → 不稳定。

3. 阶跃响应若持续增大或振荡，说明系统进入不稳定区。

### 水箱系统特殊说明
- 单水箱（一阶）系统本身稳定，增加PID后需关注极点位置
- 双水箱（二阶）系统更容易出现振荡，需合理选择PID参数
- 纯滞后环节（τ>0）会降低系统稳定性，需通过Pade近似分析等效零极点
- 不同PID算法对比：
  - 经典PID：控制精度高，但易超调
  - 增量式PID：无积分饱和，适合执行器增量控制
  - 模糊PID：鲁棒性强，适合非线性/大滞后系统
""")

# --- 版权信息 ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "太原理工大学 IPAC 实验室 © 2025"
    "</div>",
    unsafe_allow_html=True
)