import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 页面设置 ---
st.set_page_config(page_title="温度控制系统智慧仿真", layout="wide")

st.title("🌡️ 过程控制：温度系统 PID 仿真实验")
st.markdown("本实验模拟一个具有**大滞后**特性的加热炉温度控制系统。请调整左侧参数，观察控制效果。")

# --- 侧边栏：参数设置 ---
with st.sidebar:
    st.header("🎮 控制参数 (PID)")
    kp = st.slider("比例增益 (Kp)", 0.1, 10.0, 2.0, 0.1)
    ti = st.slider("积分时间 (Ti)", 0.1, 50.0, 10.0, 0.5)
    td = st.slider("微分时间 (Td)", 0.0, 10.0, 0.5, 0.1)

    st.divider()
    st.header("🏭 对象参数 (FOPDT)")
    K_process = st.slider("对象增益 (K)", 1.0, 10.0, 5.0)
    T_process = st.slider("时间常数 (T)", 10.0, 100.0, 50.0)
    L_delay = st.slider("滞后时间 (τ)", 0, 20, 10)  # 纯滞后

    st.divider()
    setpoint = st.number_input("目标温度设定值", value=100.0)


# --- 核心仿真逻辑 (离散化模拟) ---
# 为了处理纯滞后，使用离散迭代比传递函数库更容易在Web端实现
def run_simulation(kp, ti, td, K, T, L, sp, total_time=300, dt=0.5):
    n_steps = int(total_time / dt)
    time = np.linspace(0, total_time, n_steps)

    # 初始化数组
    y = np.zeros(n_steps)  # 输出温度
    u = np.zeros(n_steps)  # 控制量(阀门开度)
    error = np.zeros(n_steps)  # 误差

    # 滞后缓冲区 (Delay Buffer)
    delay_steps = int(L / dt)

    # PID 积分项和微分项初始化
    integral = 0
    prev_error = 0

    for i in range(1, n_steps):
        # 1. 计算当前误差
        # 注意：实际系统中，控制器看到的是当前的y(i-1)，因为y(i)还没算出来
        error[i] = sp - y[i - 1]

        # 2. PID 算法
        integral += error[i] * dt
        derivative = (error[i] - prev_error) / dt

        # 防止积分饱和(可选简单限幅)
        if integral > 100: integral = 100
        if integral < -100: integral = -100

        # 计算控制量 u
        # 理想PID: u = Kp * (e + 1/Ti * ∫e + Td * de/dt)
        # 简单处理：若Ti太小防除零
        term_i = (1 / ti * integral) if ti > 0.01 else 0

        u_val = kp * (error[i] + term_i + td * derivative)

        # 执行器限幅 (0-100%开度)
        u[i] = np.clip(u_val, 0, 100)

        prev_error = error[i]

        # 3. 对象模型解算 (一阶惯性 + 滞后)
        # 离散化公式: y[k] = (dt/T)*K*u_delayed + (1 - dt/T)*y[k-1]

        # 获取滞后后的控制量
        idx_delayed = i - delay_steps
        if idx_delayed < 0:
            u_delayed = 0
        else:
            u_delayed = u[idx_delayed]

        # 一阶惯性环节迭代
        y[i] = (dt / T) * K * u_delayed + (1 - (dt / T)) * y[i - 1]

    return time, y, u, error


# --- 运行仿真 ---
time, y, u, error = run_simulation(kp, ti, td, K_process, T_process, L_delay, setpoint)

# --- 绘图展示 (使用Plotly实现交互式图表) ---
# 图1：温度响应
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(x=time, y=y, mode='lines', name='实际温度 PV'))
fig_temp.add_trace(go.Scatter(x=time, y=[setpoint] * len(time), mode='lines', name='设定值 SP', line=dict(dash='dash')))
fig_temp.update_layout(title='温度响应曲线', xaxis_title='时间 (s)', yaxis_title='温度 (℃)', height=400)
st.plotly_chart(fig_temp, use_container_width=True)

# 图2：控制量输出
fig_u = go.Figure()
fig_u.add_trace(go.Scatter(x=time, y=u, mode='lines', name='阀门开度 OP', line=dict(color='orange')))
fig_u.update_layout(title='控制量(阀门开度)变化', xaxis_title='时间 (s)', yaxis_title='开度 (%)', height=300)
st.plotly_chart(fig_u, use_container_width=True)

# --- 智慧教学区：AI 分析 ---
st.info(
    "💡 **AI 助教提示：** 试着将滞后时间 $\\tau$ 增加到 15s，你会发现系统开始震荡。此时尝试减小 Kp 或增加 Ti 来重新稳定系统。")
