# %%
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from numba import jit, njit, prange

# 全局变量定义
length = 500  # 人阵规模，不得超出10000*10000
width = 500

# 功能函数A，计算某种基于输入参数的复合数学函数
@jit(nopython=True)
def A(a, b):
    y = math.log1p(a) / math.log(2) * 0.5 * (1 + math.tanh(5 * (b + 1.5)))
    return y

# 功能函数g，生成基于时间t的随机函数值
@jit(nopython=True)
def g(t):
    y = 1 + math.tanh(7 * (math.sin((math.sqrt(3)) * 0.7 * t) + math.sin((math.sqrt(5)) * 0.7 * t) +
                           0.332 * math.sin((math.sqrt(16)) * 0.7 * t) + math.sin((math.sqrt(14)) * 0.7 * t) +
                           1.02 * math.sin((math.sqrt(2.5803) * 0.7 * t))))
    return y

# 函数g2，用于错开各个f的随机初始值
@jit(nopython=True)
def g2(t):
    y = (math.sin((math.sqrt(3)) * 0.7 * t) + math.sin((math.sqrt(5)) * 0.7 * t) +
         0.332 * math.sin((math.sqrt(16)) * 0.7 * t) + math.sin((math.sqrt(14)) * 0.7 * t) +
         1.02 * math.sin((math.sqrt(2.5803) * 0.7 * t)))
    return y

# 递推步进函数，用于计算下一时间步的值
@jit(nopython=True, parallel=True)
def Nextf(step, ramvec, deltat):
    t = 0.05 * step
    # 使用显式循环而非生成器表达式来计算fl
    fl_sum = 0.0
    for i in prange(6):
        fl_sum += ramvec[(step + 19 - i) % 20]
    fl = fl_sum / 6

    # 使用显式循环计算fv
    fv_sum1 = 0.0
    fv_sum2 = 0.0
    for i in prange(5):
        fv_sum1 += ramvec[(step + 19 - i) % 20]
        fv_sum2 += ramvec[(step + 9 - i) % 20]
    fv = ((fv_sum1 / 5) - (fv_sum2 / 5)) * 2
    
    # 确保g和A函数与numba兼容
    ft = g(t - deltat) * A(fl, fv) + 0.001 * g(t - deltat)
    return ft

# 卷积核函数，决定声音如何传播，衰减
@jit(nopython=True)
def core(i, j, fdata):
    width, length = fdata.shape
    loud = 0.0
    fsum = 0.0
    f1sum = 0.0
    for m in range(width):
        for n in range(length):
            r = np.sqrt((m - i) ** 2 + (n - j) ** 2)  # 计算距离
            daor = 0 if r == 0 else (0.1 * r + 1) ** -3
            fsum += fdata[m, n] * daor
            f1sum += daor
    loud = fsum / f1sum if f1sum != 0 else 0
    return loud
# @njit
# def vectorized_core(i, j, fdata):
#     m, n = np.meshgrid(np.arange(fdata.shape[0]), np.arange(fdata.shape[1]), indexing='ij')
#     r = np.sqrt((m - i) ** 2 + (n - j) ** 2)
#     daor = np.where(r == 0, 0, (0.1 * r + 1) ** -3)
#     fsum = np.sum(fdata * daor)
#     f1sum = np.sum(daor)
#     loud = fsum / f1sum if f1sum != 0 else 0
#     return loud
# 初始化传出响度定义
# fdata = [[1 for _ in range(length)] for _ in range(width)]
fdata = np.ones((width, length))

# 历史接受响度初始化
fsave = np.ones((width, length, 20))

simulength = 20 * 600
result = np.zeros((simulength, width, length))

# %% 主循环
@jit(nopython=True, parallel=True)
def update_fsave_and_fdata(fsave, fdata, result, simulength):
    for step in range(1, simulength + 1):
        # print('step:', step, '/', simulength)
        width, length = fdata.shape
        # 卷积核计算
        for i in prange(width):
            for j in prange(length):
                fsave[i, j, step % 20] = core(i, j, fdata)
        # 反馈计算
        for i in prange(width):
            for j in prange(length):
                fdata[i, j] = Nextf(step, fsave[i, j], (114 * g2(i) + 1919 * g2(j)))

        # 保存step的矩阵
        result[step-1] = fdata

s_time = time.time()
update_fsave_and_fdata(fsave, fdata, result, simulength)
time_duration = time.time() - s_time
print('time_duration:', time_duration)


# 保存result
np.save('result.npy', result)
print('\nsave.')


# # %% 将result 绘制动画
# fig, ax = plt.subplots()

# ims = []
# for i in range(len(result)):
#     im = plt.imshow(result[i], cmap = 'magma', animated=True)
#     plt.colorbar()
#     ims.append([im])

# ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# ani.save('/Users/richard/Downloads/1test.mp4', writer='ffmpeg', dpi=200)
# %%
