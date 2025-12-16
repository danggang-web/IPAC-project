# ===============================================
# 太原理工大学 IPAC 实验室
# 水箱系统控制与分析教学平台 © 2025
# ===============================================

import numpy as np
import streamlit as st
import control
import matplotlib
import matplotlib.pyplot as plt

# ========== 页面设置 ==========
st.set_page_config(layout="wide")

# ========== 中文显示 ==========
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'PingFang SC',
    'Heiti SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS'
]
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 标题 ==========
st.markdown("""
<div style="background-color:#1976D2;padding:15px;border-radius:8px">
<h2 style="color:white;text-align:center">
太原理工大学 IPAC 实验室<br>
水箱系统建模与控制综合实验平台
</h2>
</div>
""", unsafe_allow_html=True)

# ========== 淡蓝色模块 ==========
def blue_block(title):
    st.markdown(f"""
    <div style="background-color:#E3F2FD;
                padding:12px;border-radius:8px;margin-bottom:10px;">
    <h4>{title}</h4>
    """, unsafe_allow_html=True)

def end_block():
    st.markdown("</div>", unsafe_allow_html=True)

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("⚙️ 参数设置")

    model_type = st.selectbox(
        "水箱模型选择",
        ["单水箱（一阶）", "双水箱（二阶）"]
    )

    controller_type = st.selectbox(
        "控制算法",
        ["经典 PID", "增量 PID", "模糊 PID"]
    )

    # ===== 自动整定模块 =====
    st.subheader("🤖 自动整定模块")

    tune_method = st.selectbox(
        "整定方法",
        ["经验整定（教学版）", "Ziegler–Nichols（近似）"]
    )

    if "auto_params" not in st.session_state:
        st.session_state.auto_params = (2.0, 1.0, 0.5)

    if st.button("🚀 一键自动整定"):
        if model_type == "单水箱（一阶）":
            tau = 5.0
            if tune_method == "经验整定（教学版）":
                Kp = 1.5
                Ki = 0.8
                Kd = 0.3
            else:
                Kp = 1.2 * tau
                Ki = Kp / (2 * tau)
                Kd = 0.5 * tau
        else:
            if tune_method == "经验整定（教学版）":
                Kp, Ki, Kd = 2.5, 1.2, 0.4
            else:
                Kp, Ki, Kd = 3.0, 1.5, 0.6

        st.session_state.auto_params = (Kp, Ki, Kd)
        st.success("自动整定完成，可继续手动微调")

    st.subheader("🎯 PID 参数（可手动微调）")

    Kp, Ki, Kd = st.session_state.auto_params

    Kp = st.slider("Kp", 0.0, 10.0, Kp)
    Ki = st.slider("Ki", 0.0, 5.0, Ki)
    Kd = st.slider("Kd", 0.0, 5.0, Kd)

    st.session_state.auto_params = (Kp, Ki, Kd)

# ========== 系统模型 ==========
if model_type == "单水箱（一阶）":
    G = control.tf([1], [5, 1])
else:
    G = control.tf([1], [10, 6, 1])

# ========== 控制器 ==========
C = control.tf([Kd, Kp, Ki], [1, 0])
sys = control.feedback(C * G, 1)

# ========== 响应与性能 ==========
t, y = control.step_response(sys)
y_final = y[-1]

rise_time = (
    t[np.where(y >= 0.9 * y_final)[0][0]]
    if y_final != 0 and np.any(y >= 0.9 * y_final)
    else None
)

overshoot = (
    (np.max(y) - y_final) / y_final * 100
    if y_final != 0 else None
)

steady_error = abs(1 - y_final)

def show(x):
    return "--" if x is None else round(float(x), 4)

# ========== 第一排 ==========
c1, c2 = st.columns(2)

with c1:
    blue_block("零极点公式显示")
    st.latex(r"G(s)=\frac{\prod (s-z_i)}{\prod (s-p_i)}")
    st.write("零点：", control.zeros(sys))
    st.write("极点：", control.poles(sys))
    end_block()

with c2:
    blue_block("性能指标")
    st.metric("上升时间 (s)", show(rise_time))
    st.metric("超调量 (%)", show(overshoot))
    st.metric("稳态误差", show(steady_error))
    end_block()

# ========== 第二排 ==========
c3, c4 = st.columns(2)

with c3:
    blue_block("零极点图")
    poles = control.poles(sys)
    zeros = control.zeros(sys)
    fig, ax = plt.subplots()
    ax.scatter(poles.real, poles.imag, color='red', marker='x', s=80, label='极点')
    ax.scatter(zeros.real, zeros.imag,
               facecolors='none', edgecolors='blue',
               s=80, label='零点')
    ax.axhline(0); ax.axvline(0)
    ax.set_xlabel("实轴")
    ax.set_ylabel("虚轴")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    end_block()

with c4:
    blue_block("阶跃响应")
    fig, ax = plt.subplots()
    ax.plot(t, y, label="阶跃响应")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("输出")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    end_block()

# ========== 第三排 ==========
c5, c6 = st.columns(2)

with c5:
    blue_block("根轨迹")
    fig, ax = plt.subplots()
    control.root_locus(C * G, ax=ax, grid=True)
    ax.set_xlabel("实轴")
    ax.set_ylabel("虚轴")
    st.pyplot(fig)
    end_block()

with c6:
    blue_block("波特图")
    fig, ax = plt.subplots(2, 1)
    control.bode(sys, ax=ax)
    st.pyplot(fig)
    end_block()

# ========== 稳定性说明 ==========
blue_block("🔍 系统稳定性判读说明（零极点图与根轨迹）")
st.markdown("""
1. 系统稳定性由 **极点（×）** 决定，零点（○）仅用于结构分析  
2. 所有极点实部 < 0 → **系统稳定**  
3. 存在极点实部 > 0 → **系统不稳定**  
4. 阶跃响应持续振荡或发散 → 系统进入不稳定区  
""")
end_block()

# ========== 版权 ==========
st.markdown("""
<hr>
<div style="text-align:center;color:gray">
© 2025 太原理工大学 IPAC 实验室
</div>
""", unsafe_allow_html=True)
