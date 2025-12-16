import streamlit as st
import numpy as np
import control
import matplotlib.pyplot as plt

# ======================================================
# 页面配置
# ======================================================
st.set_page_config(
    page_title="太原理工大学 IPAC — 水箱在线仿真平台",
    layout="wide"
)

# ======================================================
# 主题 CSS（对标 HTML）
# ======================================================
st.markdown("""
<style>
:root { --blue:#0077cc; --card-bg:#ffffff; --page-bg:#f5f7fa; }

html, body, [class*="css"] {
  background-color: var(--page-bg);
  font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
}

.header {
  height:60px;
  background:#007acc;
  color:white;
  display:flex;
  align-items:center;
  padding:0 20px;
  font-size:22px;
  font-weight:bold;
  border-radius:6px;
  margin-bottom:18px;
}

.card {
  background:var(--card-bg);
  border-radius:10px;
  box-shadow:0 2px 10px rgba(0,0,0,0.08);
  padding:14px;
  margin-bottom:18px;
}

.card h3 {
  margin-top:0;
  color:var(--blue);
  text-align:center;
  font-size:16px;
}

.metrics table {
  width:100%;
  border-collapse:collapse;
  text-align:center;
}
.metrics th, .metrics td {
  border:1px solid #e6eef6;
  padding:6px;
}

.stability {
  padding:18px;
  background:#eef7ff;
  border-left:6px solid var(--blue);
  border-radius:8px;
  line-height:1.6;
  color:#073b6b;
}

.footer {
  text-align:center;
  padding:12px;
  background:#f0f0f0;
  font-size:14px;
  color:#555;
  margin-top:18px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# Header
# ======================================================
st.markdown("""
<div class="header">
太原理工大学 IPAC — 水箱在线仿真平台
</div>
""", unsafe_allow_html=True)

# ======================================================
# 左右布局
# ======================================================
left, right = st.columns([1.1, 2.2])

# ======================================================
# 左侧：参数与控制
# ======================================================
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>PID 与水箱参数</h3>", unsafe_allow_html=True)

    Kp = st.slider("控制 Kp", 0.0, 20.0, 5.0, 0.1)
    Ki = st.slider("控制 Ki", 0.0, 10.0, 2.7, 0.01)
    Kd = st.slider("控制 Kd", 0.0, 5.0, 4.7, 0.01)

    dt = st.slider("采样时间 dt (s)", 0.005, 0.5, 0.05, 0.005)

    tank_type = st.selectbox("水箱类型", ["单水箱（一阶）", "双水箱（二阶）"])
    ctrl_type = st.selectbox("控制算法", ["经典 PID", "增量 PID", "模糊 PID"])

    K = st.number_input("被控对象增益 K", value=1.0)

    if tank_type == "单水箱（一阶）":
        T1 = st.number_input("时间常数 τ (s)", value=5.0)
    else:
        T1 = st.number_input("时间常数 T1 (s)", value=2.0)
        T2 = st.number_input("时间常数 T2 (s)", value=5.0)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# 系统建模
# ======================================================
if tank_type == "单水箱（一阶）":
    G = control.tf([K], [T1, 1])
else:
    G = control.tf([K], [T1*T2, T1+T2, 1])

C = control.tf([Kd, Kp, Ki], [1, 0])
sys_cl = control.feedback(C * G)

# ======================================================
# 右侧：图形与分析
# ======================================================
with right:

    # ---------- 传递函数 & 零极点 ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>闭环传递函数（零极点形式）</h3>", unsafe_allow_html=True)

    zeros = control.zeros(sys_cl)
    poles = control.poles(sys_cl)

    z_latex = " ".join([f"(s-({z.real:.2f}))" for z in zeros]) or "1"
    p_latex = " ".join([f"(s-({p.real:.2f}))" for p in poles])

    st.latex(rf"G_{{cl}}(s)=\frac{{{z_latex}}}{{{p_latex}}}")

    fig_pz, ax = plt.subplots()
    control.pzmap(sys_cl, ax=ax, grid=True)
    st.pyplot(fig_pz)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 阶跃响应 ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>阶跃响应</h3>", unsafe_allow_html=True)

    t, y = control.step_response(sys_cl)
    fig, ax = plt.subplots()
    ax.plot(t, y, label="闭环输出")
    ax.plot(t, np.ones_like(t), "--", label="参考输入")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("水位")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 根轨迹 ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>根轨迹</h3>", unsafe_allow_html=True)

    fig_rl, ax = plt.subplots()
    control.root_locus(G, ax=ax, grid=True)
    st.pyplot(fig_rl)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 波特图 ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>波特图</h3>", unsafe_allow_html=True)

    fig_bode, ax = plt.subplots(2, 1, figsize=(6, 6))
    control.bode_plot(sys_cl, ax=ax, grid=True)
    st.pyplot(fig_bode)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# 性能指标
# ======================================================
y_final = y[-1]
rise_time = t[np.where(y >= 0.9*y_final)[0][0]]
overshoot = (np.max(y)-y_final)/y_final*100
steady_error = abs(1-y_final)

st.markdown(f"""
<div class="card metrics">
<h3>性能指标</h3>
<table>
<tr><th>上升时间 t<sub>r</sub></th><th>超调量 M<sub>p</sub> (%)</th><th>稳态误差 e<sub>ss</sub></th></tr>
<tr>
<td>{rise_time:.3f}</td>
<td>{overshoot:.2f}</td>
<td>{steady_error:.4f}</td>
</tr>
</table>
</div>
""", unsafe_allow_html=True)

# ======================================================
# 稳定性说明
# ======================================================
st.markdown("""
<div class="stability">
<h3>🔍 系统稳定性判读说明（零极点图与根轨迹）</h3>
<p><strong>1.</strong> 系统稳定性由 <strong>极点（×）</strong> 决定，零点（○）仅用于分析零极点关系。</p>
<p><strong>2.</strong> 所有极点实部 &lt; 0 → <span style="color:green;font-weight:700;">系统稳定</span>；
存在极点实部 &gt; 0 → <span style="color:red;font-weight:700;">不稳定</span>。</p>
<p><strong>3.</strong> 阶跃响应若持续增大或振荡，说明系统进入不稳定区。</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# 页脚
# ======================================================
st.markdown("""
<div class="footer">
太原理工大学 IPAC 实验室 © 2025
</div>
""", unsafe_allow_html=True)
