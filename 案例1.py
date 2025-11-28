import numpy as np
import matplotlib.pyplot as plt
import control as ctrl  # 安装：pip install control

plt.rcParams['font.sans-serif'] = ['SimHei']        # 显示中文
plt.rcParams['axes.unicode_minus'] = False          # 显示负号
plt.rcParams['mathtext.fontset'] = 'cm'             # 数学公式用 LaTeX 风格

# ===========================
# 1. 定义系统参数
# ===========================
Kp = 2.0     # 比例系数
Ki = 0.5     # 积分系数
Kd = 0.1     # 微分系数

# 一阶惯性对象 Gp(s) = 1 / (5s + 1)
num_p = [1]
den_p = [5, 1]
Gp = ctrl.TransferFunction(num_p, den_p)

# PID 控制器 Gc(s) = Kp + Ki/s + Kd*s
Gc = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])

# 闭环系统 G(s) = (Gc*Gp) / (1 + Gc*Gp)
G_closed = ctrl.feedback(Gc * Gp, 1)

# ===========================
# 2. 阶跃响应仿真
# ===========================
t = np.linspace(0, 50, 500)
t, y = ctrl.step_response(G_closed, T=t)

# ===========================
# 3. 绘图展示
# ===========================
plt.figure(figsize=(8, 5))
plt.plot(t, y, linewidth=2)
plt.title("PID 控制系统阶跃响应", fontsize=14)
plt.xlabel("时间 (s)", fontsize=12)
plt.ylabel("系统输出", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.text(30, 0.8, f"Kp={Kp}, Ki={Ki}, Kd={Kd}", fontsize=10)
plt.show()
