# ===============================================
# 太原理工大学 IPAC 实验室
# 水箱系统建模与控制综合实验平台 © 2025
# ===============================================

import numpy as np
import streamlit as st
import control
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ======== 强制指定中文字体（100%生效）========
font_cn = FontProperties(fname="fonts/msyh.ttc", size=10)

# ========= 页面设置 =========
st.set_page_config(layout="wide")

st.markdown("""
<div style="background-color:#1976D2;padding:15px;border-radius:8px">
<h2 style="color:white;text-align:center">
太原理工大学 IPAC 实验室<br>
水箱系统建模与控制综合实验平台
</h2>
</div>
""", unsafe_allow_html=True)

# ========= 模块样式 =========
def blue_block(title):
    st.markdown(f"""
    <div style="
        background-color:#E3F2FD;
        padding:12px;
        border-radius:8px;
        margin-bottom:10px;">
    <h4>太原理工大学 IPAC 实验室 —— {title}</h4>
    """, unsafe_allow_html=True)

def end_block():
    st.markdown("</div>", unsafe_allow_html=True)

# ========= 左侧参数 =========
with st.sidebar:
    st.header("参数设置")

    model_type = st.selectbox("水箱模型", ["单水箱（一阶）", "双水箱（二阶）"])
    controller_type = st.selectbox("控制算法", ["经典PID", "增量PID", "模糊PID"])

# ========= 参数整定模块（单独） =========
blue_block("PID 参数整定模块")
Kp = st.slider("比例增益 Kp", 0.0, 10.0, 2.0)
Ki = st.slider("积分增益 Ki", 0.0, 5.0, 1.0)
Kd = st.slider("微分增益 Kd", 0.0, 5.0, 0.5)
end_block()

# ========= 系统模型 =========
if model_type == "单水箱（一阶）":
    G = control.tf([1], [5, 1])
else:
    G = control.tf([1], [10, 6, 1])

# ========= 控制器 =========
C = control.tf([Kd, Kp, Ki], [1, 0])
sys = control.feedback(C * G, 1)

# ========= 性能指标 =========
t, y = control.step_response(sys)
y_final = y[-1]

def calc_rise_time(t, y):
    try:
        return t[np.where(y >= 0.9 * y[-1])[0][0]]
    except:
        return None

rise_time = calc_rise_time(t, y)
overshoot = (np.max(y) - y_final) / y_final * 100 if y_final != 0 else None
steady_error = abs(1 - y_final)

def show(x):
    return "--" if x is None else round(float(x), 4)

# ========= 第一排 =========
col1, col2 = st.columns(2)

with col1:
    blue_block("零极点公式显示")
    st.latex(r"G(s)=\frac{\prod (s-z_i)}{\prod (s-p_i)}")
    st.write("零点：", control.zeros(sys))
    st.write("极点：", control.poles(sys))
    end_block()

with col2:
    blue_block("性能指标")
    st.metric("上升时间 (s)", show(rise_time))
    st.metric("超调量 (%)", show(overshoot))
    st.metric("稳态误差", show(steady_error))
    end_block()

# ========= 第二排 =========
col3, col4 = st.columns(2)

with col3:
    blue_block("零极点图")
    poles = control.poles(sys)
    zeros = control.zeros(sys)

    fig, ax = plt.subplots()
    ax.scatter(poles.real, poles.imag, marker='x',
               color='red', s=80, label='极点')
    ax.scatter(zeros.real, zeros.imag, marker='o',
               facecolors='none', edgecolors='blue',
               s=80, label='零点')

    ax.axhline(0, color='black')
    ax.axvline(0, color='black')

    ax.set_xlabel("实轴", fontproperties=font_cn)
    ax.set_ylabel("虚轴", fontproperties=font_cn)
    ax.legend(prop=font_cn)
    ax.grid(True)

    st.pyplot(fig)
    end_block()

with col4:
    blue_block("阶跃响应")
    fig, ax = plt.subplots()
    ax.plot(t, y, label="阶跃响应")
    ax.set_xlabel("时间 (s)", fontproperties=font_cn)
    ax.set_ylabel("输出", fontproperties=font_cn)
    ax.legend(prop=font_cn)
    ax.grid(True)
    st.pyplot(fig)
    end_block()

# ========= 第三排 =========
col5, col6 = st.columns(2)

with col5:
    blue_block("根轨迹")
    fig, ax = plt.subplots()
    control.root_locus(C * G, ax=ax, grid=True)
    ax.set_xlabel("实轴", fontproperties=font_cn)
    ax.set_ylabel("虚轴", fontproperties=font_cn)
    st.pyplot(fig)
    end_block()

with col6:
    blue_block("波特图")
    fig, ax = plt.subplots(2, 1)
    control.bode(sys, ax=ax)
    st.pyplot(fig)
    end_block()

# ========= 稳定性说明 =========
blue_block("系统稳定性判读说明（零极点图与根轨迹）")
st.markdown("""
🔍 **系统稳定性判读说明**

1. 系统稳定性由 **极点（×）** 决定  
2. 所有极点实部 < 0 → **系统稳定**  
3. 存在极点实部 > 0 → **系统不稳定**  
4. 阶跃响应持续振荡或发散 → 系统不稳定  
""")
end_block()

# ========= 版权 =========
st.markdown("""
<hr>
<div style="text-align:center;color:gray">
© 2025 太原理工大学 IPAC 实验室
</div>
""", unsafe_allow_html=True)
