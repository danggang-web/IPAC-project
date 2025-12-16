import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import control

# =======================
# 中文字体 & Matplotlib
# =======================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'PingFang SC',
    'Noto Sans CJK SC', 'WenQuanYi Zen Hei'
]
matplotlib.rcParams['axes.unicode_minus'] = False

# =======================
# 页面设置
# =======================
st.set_page_config(page_title="水箱控制系统教学平台", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.card {
  background: #ffffff;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  padding: 10px 12px;
  margin-bottom: 12px;
}
.card h3 {
  margin: 4px 0 8px 0;
  text-align: center;
  font-size: 15px;
  color: #2563eb;
}
</style>
""", unsafe_allow_html=True)

st.title("💧 水箱系统控制与分析教学平台")

# =======================
# 左侧参数区
# =======================
left, right = st.columns([1.0, 2.0])

with left:
    st.markdown('<div class="card"><h3>模型与控制器选择</h3>', unsafe_allow_html=True)

    tank_type = st.selectbox("水箱模型", ["单水箱（一阶）", "双水箱（二阶）"])
    ctrl_type = st.selectbox("控制算法", ["经典 PID", "增量 PID", "模糊 PID"])

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>PID 参数整定</h3>', unsafe_allow_html=True)
    Kp = st.slider("Kp", 0.0, 10.0, 2.0)
    Ki = st.slider("Ki", 0.0, 5.0, 1.0)
    Kd = st.slider("Kd", 0.0, 5.0, 0.2)
    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# 系统模型
# =======================
if tank_type == "单水箱（一阶）":
    Gp = control.tf([1], [5, 1])
else:
    Gp = control.tf([1], [10, 6, 1])

# =======================
# 控制器
# =======================
if ctrl_type == "经典 PID":
    Gc = control.tf([Kd, Kp, Ki], [1, 0])

elif ctrl_type == "增量 PID":
    Gc = control.tf([Kd, Kp, Ki], [1, 0])  # 教学等效表示

else:  # 简化模糊 PID
    Gc = control.tf([0.8*Kd, 0.8*Kp, 0.8*Ki], [1, 0])

# =======================
# 闭环系统
# =======================
sys_cl = control.feedback(Gc * Gp, 1)

# =======================
# 性能指标
# =======================
def performance_metrics(t, y):
    y_final = y[-1]
    rise = None
    over = None
    err = None

    try:
        rise = t[np.where(y >= 0.9 * y_final)[0][0]]
        over = (np.max(y) - y_final) / y_final * 100
        err = abs(1 - y_final)
    except:
        pass
    return rise, over, err

# =======================
# 右侧显示区
# =======================
with right:

    # ---------- 零极点公式 ----------
    st.markdown('<div class="card"><h3>闭环系统零极点（公式）</h3>', unsafe_allow_html=True)
    zeros = control.zeros(sys_cl)
    poles = control.poles(sys_cl)

    def zp_latex(zp):
        if len(zp) == 0:
            return "1"
        s = []
        for z in zp:
            if abs(z.imag) < 1e-6:
                s.append(f"(s - {z.real:.2f})")
            else:
                s.append(f"(s - ({z.real:.2f} {'+' if z.imag>0 else '-'} {abs(z.imag):.2f}i))")
        return " ".join(s)

    st.latex(rf"G(s)=\frac{{{zp_latex(zeros)}}}{{{zp_latex(poles)}}}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 零极点图 ----------
    st.markdown('<div class="card"><h3>零极点图</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    control.pzmap(sys_cl, ax=ax, grid=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 阶跃响应 ----------
    st.markdown('<div class="card"><h3>阶跃响应</h3>', unsafe_allow_html=True)
    t, y = control.step_response(sys_cl)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(t, y, label="系统输出")
    ax.plot(t, np.ones_like(t), "--", label="参考输入")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("水位")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 性能指标 ----------
    st.markdown('<div class="card"><h3>性能指标</h3>', unsafe_allow_html=True)
    rise, over, err = performance_metrics(t, y)

    def show(x):
        return "--" if x is None else f"{x:.3f}"

    c1, c2, c3 = st.columns(3)
    c1.metric("上升时间 (s)", show(rise))
    c2.metric("超调量 (%)", show(over))
    c3.metric("稳态误差", show(err))
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 根轨迹 ----------
    st.markdown('<div class="card"><h3>根轨迹</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    control.root_locus(Gc * Gp, ax=ax, grid=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 波特图 ----------
    st.markdown('<div class="card"><h3>波特图</h3>', unsafe_allow_html=True)
    fig = plt.figure(figsize=(5.2, 4.0))
    control.bode_plot(sys_cl, dB=True, grid=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# 底部说明
# =======================
st.markdown("""
<div class="card">
<h3>🔍 系统稳定性判读说明（零极点图与根轨迹）</h3>
<ol>
<li>系统稳定性由 <b>极点（×）</b> 决定，零点（○）仅用于分析零极点关系。</li>
<li>所有极点实部 &lt; 0 → 系统稳定；存在极点实部 &gt; 0 → 系统不稳定。</li>
<li>阶跃响应若持续增大或振荡，说明系统进入不稳定区。</li>
</ol>
<p style="text-align:center;margin-top:8px;">
© 2025 太原理工大学 IPAC 实验室
</p>
</div>
""", unsafe_allow_html=True)
