# =========================================================
# 水箱液位控制系统综合仿真平台
# 太原理工大学 IPAC 实验室 © 2025
# =========================================================

import streamlit as st
import numpy as np
import control as ctrl
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.signal import tf2zpk

# =========================================================
# 页面设置
# =========================================================
def tf_to_latex(sys, var='s'):
    """
    将 control.TransferFunction 转为 LaTeX 字符串
    兼容常数 / 一阶 / 二阶系统
    """
    def ensure_1d_array(x):
        x = np.array(x).flatten()
        if x.size == 0:
            return np.array([0.0])
        return x

    num = ensure_1d_array(sys.num[0][0])
    den = ensure_1d_array(sys.den[0][0])

    def poly_to_latex(p):
        p = ensure_1d_array(p)
        deg = len(p) - 1
        terms = []

        for i, coef in enumerate(p):
            if abs(coef) < 1e-10:
                continue
            power = deg - i
            coef_str = f"{coef:.3g}"

            if power == 0:
                terms.append(coef_str)
            elif power == 1:
                terms.append(f"{coef_str}{var}")
            else:
                terms.append(f"{coef_str}{var}^{{{power}}}")

        return " + ".join(terms) if terms else "0"

    num_latex = poly_to_latex(num)
    den_latex = poly_to_latex(den)

    return rf"\frac{{{num_latex}}}{{{den_latex}}}"



st.set_page_config(
    page_title="水箱液位控制系统仿真平台",
    layout="wide"
)

st.title("💧 水箱液位控制系统仿真平台")
st.markdown("**太原理工大学 IPAC 实验室 | 过程控制教学平台**")
st.divider()

# =========================================================
# 工具函数
# =========================================================

def build_plant(model_type, K, T1, T2):
    if model_type == "单水箱（一阶）":
        return ctrl.tf([K], [T1, 1])
    else:
        return ctrl.tf([K], np.convolve([T1, 1], [T2, 1]))


def build_controller(ctrl_type, Kp, Ti, Td):
    if ctrl_type == "经典 PID":
        return ctrl.tf([Kp * Td, Kp, Kp / Ti], [1, 0])

    elif ctrl_type == "增量 PID":
        # 教学等效形式
        return ctrl.tf([Kp * Td, Kp], [1, 0])

    else:  # 模糊 PID（简化教学模型）
        Kp_f = 0.8 * Kp
        Ti_f = 1.2 * Ti
        Td_f = 0.5 * Td
        return ctrl.tf([Kp_f * Td_f, Kp_f, Kp_f / Ti_f], [1, 0])


def performance_metrics(t, y):
    try:
        t10 = t[np.where(y >= 0.1)[0][0]]
        t90 = t[np.where(y >= 0.9)[0][0]]
        rise_time = t90 - t10
    except:
        rise_time = 0.0

    overshoot = max(0, (np.max(y) - 1) * 100)
    steady_error = abs(y[-1] - 1)

    return round(rise_time, 2), round(overshoot, 2), round(steady_error, 4)

# =========================================================
# 侧边栏：参数与算法选择
# =========================================================

with st.sidebar:
    st.header("⚙️ 系统建模")

    model_type = st.selectbox(
        "水箱模型",
        ["单水箱（一阶）", "双水箱（二阶）"]
    )

    K = st.slider("系统增益 K", 0.1, 5.0, 1.0, 0.1)
    T1 = st.slider("时间常数 T1 (s)", 1.0, 30.0, 5.0, 1.0)

    if model_type == "双水箱（二阶）":
        T2 = st.slider("时间常数 T2 (s)", 1.0, 30.0, 8.0, 1.0)
    else:
        T2 = 0.0

    st.header("🎯 控制算法")

    ctrl_type = st.selectbox(
        "控制策略",
        ["经典 PID", "增量 PID", "模糊 PID"]
    )

    Kp = st.slider("Kp", 0.1, 20.0, 5.0, 0.1)
    Ti = st.slider("Ti", 0.1, 30.0, 10.0, 0.5)
    Td = st.slider("Td", 0.0, 10.0, 1.0, 0.1)

# =========================================================
# 系统构建
# =========================================================

G = build_plant(model_type, K, T1, T2)
Gc = build_controller(ctrl_type, Kp, Ti, Td)

G_open = ctrl.series(Gc, G)
G_cl = ctrl.feedback(G_open, 1)

# =========================================================
# 传递函数公式显示
# =========================================================

st.subheader("📐 传递函数（公式显示）")

st.latex(r"G(s) = " + tf_to_latex(G))
st.latex(r"G_c(s) = " + tf_to_latex(Gc))
st.latex(r"T(s) = \frac{G_c(s)G(s)}{1+G_c(s)G(s)}")

# =========================================================
# 零极点图
# =========================================================

st.subheader("📍 零极点分布图")

z_o, p_o, _ = tf2zpk(G_open.num[0][0], G_open.den[0][0])
_, p_c, _ = tf2zpk(G_cl.num[0][0], G_cl.den[0][0])

fig_zp = go.Figure()

fig_zp.add_trace(go.Scatter(
    x=np.real(z_o), y=np.imag(z_o),
    mode="markers", name="零点 ○",
    marker=dict(symbol="circle", size=10)
))

fig_zp.add_trace(go.Scatter(
    x=np.real(p_o), y=np.imag(p_o),
    mode="markers", name="开环极点 ×",
    marker=dict(symbol="x", size=10)
))

fig_zp.add_trace(go.Scatter(
    x=np.real(p_c), y=np.imag(p_c),
    mode="markers", name="闭环极点 ×",
    marker=dict(symbol="x", size=12, color="red")
))

fig_zp.add_vline(x=0, line=dict(dash="dash"))
fig_zp.update_layout(xaxis_title="Re", yaxis_title="Im", height=400)

st.plotly_chart(fig_zp, use_container_width=True)

# =========================================================
# 阶跃响应 & 性能指标
# =========================================================

st.subheader("📊 阶跃响应与性能指标")

t, y = ctrl.step_response(G_cl, T=np.linspace(0, 100, 1000))
rise, over, err = performance_metrics(t, y)

fig_step = go.Figure()
fig_step.add_trace(go.Scatter(x=t, y=y, mode="lines", name="阶跃响应"))
fig_step.add_hline(y=1, line=dict(dash="dash"))

st.plotly_chart(fig_step, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("上升时间 (s)", rise)
c2.metric("超调量 (%)", over)
c3.metric("稳态误差", err)

# =========================================================
# 根轨迹
# =========================================================

st.subheader("🌐 根轨迹图")

rlist, klist = ctrl.root_locus(G_open, plot=False)

fig_rl = go.Figure()
for i in range(rlist.shape[0]):
    fig_rl.add_trace(go.Scatter(
        x=np.real(rlist[i]),
        y=np.imag(rlist[i]),
        mode="lines"
    ))

fig_rl.add_vline(x=0, line=dict(dash="dash"))
fig_rl.update_layout(xaxis_title="Re", yaxis_title="Im", height=400)

st.plotly_chart(fig_rl, use_container_width=True)

# =========================================================
# 波特图
# =========================================================

st.subheader("📉 波特图")

omega, mag, phase = ctrl.bode(G_open, plot=False)

fig_bode = sp.make_subplots(rows=2, cols=1,
    subplot_titles=("幅频特性 (dB)", "相频特性 (deg)")
)

fig_bode.add_trace(
    go.Scatter(x=np.log10(omega), y=20*np.log10(mag)),
    row=1, col=1
)
fig_bode.add_trace(
    go.Scatter(x=np.log10(omega), y=phase),
    row=2, col=1
)

fig_bode.update_layout(height=600)
st.plotly_chart(fig_bode, use_container_width=True)

# =========================================================
# 稳定性说明模块
# =========================================================

st.subheader("🔍 系统稳定性判读说明（零极点图与根轨迹）")

st.markdown("""
1. **系统稳定性由极点（×）决定**，零点（○）仅用于分析零极点关系。  
2. **所有极点实部 < 0 → 系统稳定**；若存在极点实部 > 0 → 系统不稳定。  
3. **阶跃响应持续增大或振荡**，说明系统进入不稳定区。
""")

# =========================================================
# 页脚
# =========================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;'>"
    "太原理工大学 IPAC 实验室 © 2025"
    "</div>",
    unsafe_allow_html=True
)
