# =========================================================
# 水箱系统建模与控制分析平台（中文显示修复完整版）
# 太原理工大学 IPAC 实验室 © 2025
# =========================================================

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import control as ctl
from matplotlib.font_manager import FontProperties

# =========================================================
# 中文字体（关键）
# =========================================================
ch_font = FontProperties(family='SimHei')

plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 页面配置 & 样式
# =========================================================
st.set_page_config(layout="wide")

st.markdown("""
<style>
.header {
    background-color:#007acc;
    padding:18px;
    border-radius:10px;
    margin-bottom:20px;
}
.header h1 {
    color:white;
    text-align:center;
}
.card {
    background:#eaf4ff;
    padding:16px;
    border-radius:12px;
    margin-bottom:16px;
}
.card h3 {
    color:#005b99;
}
.footer {
    text-align:center;
    color:#666;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
<h1>太原理工大学 IPAC 实验室 —— 水箱系统建模与控制分析平台</h1>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 工具函数
# =========================================================
def safe(x):
    return "--" if x is None else f"{x:.3f}"

def performance_metrics(t, y):
    try:
        y_final = y[-1]
        y_peak = np.max(y)
        overshoot = (y_peak - y_final) / y_final * 100 if y_final != 0 else 0
        idx = np.where(y >= 0.9 * y_final)[0]
        rise = t[idx[0]] if len(idx) else None
        ess = abs(1 - y_final)
        return rise, overshoot, ess
    except:
        return None, None, None

# =========================================================
# 侧边栏
# =========================================================
with st.sidebar:
    st.header("⚙️ 系统配置")

    tank_type = st.radio("水箱模型", ["单水箱（一阶）", "双水箱（二阶）"])
    ctrl_type = st.selectbox("控制器类型", ["经典 PID", "增量 PID", "模糊 PID"])
    tune_method = st.radio("整定方式", ["手动整定", "ZN 临界比例法"])

    Kp = st.slider("Kp", 0.0, 10.0, 2.0)
    Ki = st.slider("Ki", 0.0, 5.0, 1.0)
    Kd = st.slider("Kd", 0.0, 5.0, 0.5)

    if tune_method == "ZN 临界比例法":
        Ku = st.slider("临界比例 Ku", 0.1, 20.0, 5.0)
        Tu = st.slider("临界周期 Tu", 0.1, 20.0, 2.0)
        if st.button("一键 ZN 整定"):
            Kp = 0.6 * Ku
            Ki = 1.2 * Ku / Tu
            Kd = 0.075 * Ku * Tu

# =========================================================
# 系统模型
# =========================================================
if tank_type == "单水箱（一阶）":
    G = ctl.tf([1], [10, 1])
else:
    G = ctl.tf([1], [50, 15, 1])

# 控制器
if ctrl_type == "经典 PID":
    C = ctl.tf([Kd, Kp, Ki], [1, 0])
elif ctrl_type == "增量 PID":
    C = ctl.tf([Kd, Kp, Ki], [1, -1])
else:
    C = ctl.tf([Kd, 0.8*Kp, 0.5*Ki], [1, 0])

sys_cl = ctl.feedback(C * G, 1)

# =========================================================
# 仿真
# =========================================================
t, y = ctl.step_response(sys_cl)
rise, over, err = performance_metrics(t, y)

zeros = ctl.zeros(sys_cl)
poles = ctl.poles(sys_cl)

# =========================================================
# 第一行
# =========================================================
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📐 在线零极点显示（公式）")
    st.latex(r"G(s)=\frac{\prod (s-z_i)}{\prod (s-p_i)}")
    st.write("零点 z：", np.round(zeros, 3))
    st.write("极点 p：", np.round(poles, 3))
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 性能指标")
    st.metric("上升时间 (s)", safe(rise))
    st.metric("超调量 (%)", safe(over))
    st.metric("稳态误差", safe(err))
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 第二行
# =========================================================
c3, c4 = st.columns(2)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📍 零极点图（左右半平面区分）")

    fig, ax = plt.subplots()
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.plot(np.real(poles), np.imag(poles),
            'rx', markersize=10, label="极点（×）")
    ax.plot(np.real(zeros), np.imag(zeros),
            'bo', markersize=8, label="零点（○）")

    ax.set_xlabel("实部 Re", fontproperties=ch_font)
    ax.set_ylabel("虚部 Im", fontproperties=ch_font)
    ax.legend(prop=ch_font)
    ax.grid(True)

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 阶跃响应")

    fig, ax = plt.subplots()
    ax.plot(t, y, label="系统响应")
    ax.plot(t, np.ones_like(t), "--", label="参考输入")

    ax.set_xlabel("时间 (s)", fontproperties=ch_font)
    ax.set_ylabel("液位", fontproperties=ch_font)
    ax.legend(prop=ch_font)
    ax.grid(True)

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 第三行
# =========================================================
c5, c6 = st.columns(2)

with c5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧭 根轨迹")

    fig, ax = plt.subplots()
    ctl.root_locus(G, ax=ax, grid=True)

    ax.set_xlabel("实部 Re", fontproperties=ch_font)
    ax.set_ylabel("虚部 Im", fontproperties=ch_font)

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with c6:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📐 波特图")

    fig, ax = plt.subplots(2, 1)
    ctl.bode_plot(sys_cl, ax=ax)

    ax[0].set_ylabel("幅值 (dB)", fontproperties=ch_font)
    ax[1].set_ylabel("相位 (deg)", fontproperties=ch_font)
    ax[1].set_xlabel("频率 (rad/s)", fontproperties=ch_font)

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 稳定性说明
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("""
### 🔍 系统稳定性判读说明（零极点图与根轨迹）

1. 系统稳定性由 **极点（×）** 决定  
2. 零点（○）仅用于零极点关系分析  
3. 所有极点实部 < 0 → **系统稳定**  
4. 存在极点实部 > 0 → **系统不稳定**  
5. 阶跃响应发散或持续振荡 → 进入不稳定区
""")
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 页脚
# =========================================================
st.markdown(
    '<div class="footer">太原理工大学 IPAC 实验室 © 2025</div>',
    unsafe_allow_html=True
)
