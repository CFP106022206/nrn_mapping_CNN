# %%
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
from numba import jit, njit, prange
from tqdm import trange, tqdm

# @jit
def galvo_y_pos(t, ttl_freq):

    initial_y = 30               # μm, galvo 初始位置偏移
    delta_y = 1745/1024 * 128    # μm, 三角波峰峰值
    
    galvo_freq = ttl_freq/(128*2)    # 1/2T = 128 pixels

    start_shift = 1/2 * 1/galvo_freq  # 三角波的起始相位
    y_pos = delta_y * np.abs(((t+start_shift) * 2*galvo_freq) % 2 - 1) + initial_y  # 创建三角波

    return y_pos

# @jit
def tag_z_pos(t, ttl_freq):

    initial_z = -120  # μm, TAG Lens 初始位置偏移(最高点)
    delta_z = 200   # μm

    z_pos = delta_z/2 * np.sin(2*np.pi*ttl_freq * t + np.pi/2) + initial_z

    duty_cycle = 1/11   # TTL信号占空比

    # if t/ttl周期 的余数在占空比范围(duty_cycle * 频率倒数)内，输出高电平，否则输出低电平
    ttl_signal = int((t % (1/ttl_freq)) < (duty_cycle/ttl_freq))
    return z_pos, ttl_signal

# %% 创建记录每个光焦点位置的数组

adj_ratio = 1   # 快速调整一些画图的参数倍率（1 for 2 MHz, 2 for 4 MHz, 4 for 8 MHz）

simulation_steps = 500000

focus_num = 32      # x轴
galvo_steps = 128   # y轴
repitition_rate = adj_ratio*2.034 * 1e6   # Hz

# flip_z = False

ttl_freq = 69e3 # Hz

data_table = np.zeros((simulation_steps, 3))    # y, z, ttl

# 计算循环开始
for i in tqdm(range(simulation_steps)):
    timeline = i / repitition_rate  # 打光时间

    y_pos = galvo_y_pos(timeline, ttl_freq)
    z_pos, ttl_signal = tag_z_pos(timeline, ttl_freq)

    data_table[i] = [y_pos, z_pos, ttl_signal]



def plot_z_axis_sample(data_table, plot_range, simulation_steps, repitition_rate):
    x = np.arange(0, simulation_steps/repitition_rate, 1/repitition_rate)[plot_range[0]:plot_range[1]]
    y = data_table[plot_range[0]:plot_range[1], 1]
    
    plt.figure(figsize=(8,6))
    grid = plt.GridSpec(3, 1, wspace=0.5, hspace=0.5)
    plt.subplot(grid[:2,0])
    plt.minorticks_on()
    plt.plot(x, y, '.', label='z-axis sampling point')
    plt.xlabel('Time(s)')
    plt.ylabel('z-axis position(μm)')
    plt.legend()
    plt.title("Laser Repetition Rate: {:.2e} Hz".format(repitition_rate))
    
    plt.subplot(grid[2,0])
    ttl_sig = data_table[plot_range[0]:plot_range[1], 2]
    plt.plot(x,ttl_sig, label='TTL Signal',c='#BBE600')
    plt.xlabel('Time(s)')
    plt.legend()
    plt.savefig('./Figure/Tag_z_axis_Sample_at_{:.2e}Hz.png'.format(repitition_rate), dpi=100, bbox_inches='tight')
    plt.show()

plot_z_axis_sample(data_table, [0, adj_ratio*100], simulation_steps, repitition_rate)




def plot_galvo_move(data_table, plot_range, simulation_steps, repitition_rate):
    x = np.arange(0, simulation_steps/repitition_rate, 1/repitition_rate)[plot_range[0]:plot_range[1]]
    y = data_table[plot_range[0]:plot_range[1], 0]
    plt.minorticks_on()
    plt.plot(x, y)
    plt.xlabel('Time(s)')
    plt.ylabel('y-axis(galvo) position(μm)')
    plt.title("Laser Repetition Rate: {:.2e} Hz".format(repitition_rate))
    plt.savefig('./Figure/Galvo_Move_at_{:.2e}Hz.png'.format(repitition_rate), dpi=100, bbox_inches='tight')
    plt.show()

plot_galvo_move(data_table, [0, adj_ratio*7000], simulation_steps, repitition_rate)



def plot_yz_plane_sample(data_table, plot_range, dot_size=5):
    x = data_table[plot_range[0]:plot_range[1], 0]  # 图上的x 轴是 galvo 轴
    y = data_table[plot_range[0]:plot_range[1], 1]  # 图上的y 轴是 tag 轴
    plt.minorticks_on()
    plt.scatter(x, y, s=dot_size, linewidth=0, label='y-z plane sampling point')
    plt.xlabel('y-axis(galvo) position(μm)')
    plt.ylabel('z-axis(TAG Lens) position(μm)') 
    # plt.legend()
    plt.title("Laser Repetition Rate: {:.2e} Hz".format(repitition_rate))
    plt.savefig('./Figure/YZ_plane_Sample_at_{:.2e}Hz.png'.format(repitition_rate), dpi=150, bbox_inches='tight')
    plt.show()

plot_yz_plane_sample(data_table, [0, adj_ratio*3500], dot_size=1)


# %% 创建荧光样本

# 正方形螢光球樣本陣列生成
def gen_sample_array(origin, edge_num, radius, space):
    # 建構以origin為原點的正方形螢光球樣本陣列
    center_lst = [[origin[0] + i * space, origin[1] + j * space] 
                   for i in range(edge_num) for j in range(edge_num)]
    radius_lst = [radius] * edge_num**2 
    return center_lst, radius_lst

# 自定義螢光球陣列擺放角度
def rotate_points(center_lst, angle):
    theta = np.radians(angle)   #換成弧度
    # 旋轉矩陣
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta), np.cos(theta)]])
    
    rotation_center = np.array(center_lst[0])  # 旋轉中心
    
    # 平移陣列以應用旋轉
    translated_points = [point - rotation_center for point in np.array(center_lst)]
    
    # 旋轉陣列
    rotated_points = [np.dot(rotation_matrix, point) for point in translated_points]
    
    # 將旋轉後的陣列位置重新移動回到原位
    rotated_translated_points = [point + rotation_center for point in rotated_points]

    return rotated_translated_points

center_lst, radius_lst = gen_sample_array([125,95], 6, 3, 10)
center_lst = rotate_points(center_lst, 45)  # 旋轉陣列角度30度

# 四角荧光球的位置和半径
center_lst += [[40, 50], [200 ,50],[40, 210], [200, 210]]  # 圆心的位置
radius_lst += [
    10,10,10,10]  # 半径

accuracy = 0.1  # 模擬樣本的像素精度(μm)

size = 250  # 样本总大小(μm)

fl_beads = np.column_stack((center_lst, radius_lst))    # y, z, radius
fl_sample = np.zeros((int(size/accuracy), int(size/accuracy)))

for bead in fl_beads:
    img_size = fl_sample.shape[0]
    bead = (bead/accuracy).astype(np.int32)

    # 计算每个元素距离圆心的距离
    x, y = np.ogrid[:img_size, :img_size]
    dist_from_center = np.sqrt((x - bead[0])**2 + (y - bead[1])**2)

    # 如果距离小于或等于半径，那么就把元素设为1
    # fl_sample[dist_from_center <= bead[2]] = 1
    # 更花俏的荧光球, 大于0.8半径边缘渐变
    fl_sample[dist_from_center <= bead[2]] = 1 - np.maximum((dist_from_center[dist_from_center <= bead[2]]/bead[2])-0.8, 0)


# %% 激光采样点和荧光球样本
def plot_yz_plane_fl_sample(data_table, plot_range, fl_sample, dot_size=3):
    x = data_table[plot_range[0]:plot_range[1], 0]  # 图上的x 轴是 galvo 轴
    y = data_table[plot_range[0]:plot_range[1], 1]  # 图上的y 轴是 tag 轴
    
    plt.figure(figsize=(6,6))
    plt.minorticks_on()
    plt.imshow(fl_sample, cmap='magma', extent=[0,size,-size,0])

    plt.scatter(x, y, s=dot_size, linewidth=0, label='y-z plane sampling point')
    plt.xlabel('y-axis(galvo) position(μm)')
    plt.ylabel('z-axis(TAG Lens) position(μm)') 
    # plt.legend()
    plt.title("Laser Repetition Rate: {:.2e} Hz".format(repitition_rate))
    plt.savefig('./Figure/YZ_plane_FL_Sample_at_{:.2e}Hz.png'.format(repitition_rate), dpi=200, bbox_inches='tight')
    plt.show()


plot_yz_plane_fl_sample(data_table, [0, adj_ratio*3500], fl_sample, dot_size=1)

# %%
# 四舍五入光点位置
sample_positions = np.round(data_table[:, :2]/accuracy).astype(np.int32)

# 记录光点位置矩阵中y轴向下为负，这里要改成 numpy的向下为正
sample_positions[:,1] = np.abs(sample_positions[:,1])

# 获取采样点的像素值
sampled_pixels = fl_sample[sample_positions[:, 1], sample_positions[:, 0]]

# 找到ttl信号中上升边缘的位置，即差异数组中值为1的位置
diff = np.diff(data_table[:, 2])
rising_edges = np.where(diff == 1)[0] + 1
rising_edges = np.insert(rising_edges, 0, 0)    # 第一个位置也是rising edge

# 重建图像
z_axis_length = np.max(np.diff(rising_edges))  # z轴最长长度
reconstructed_img = np.zeros((z_axis_length, galvo_steps))

# （不翻z）
for i in range(galvo_steps):
    start, end = rising_edges[i], rising_edges[i+1]
    reconstructed_z = sampled_pixels[start:end]
    reconstructed_img[:len(reconstructed_z), i] = reconstructed_z

# 显示完整图像
plt.figure(figsize=(6,6))   #正方形图片
plt.imshow(reconstructed_img, cmap='magma', aspect='auto')
plt.title("Reconstruct. Repetition Rate: {:.2e} Hz".format(repitition_rate))
plt.savefig('./Figure/Reconstruct_YZ_plane_FL_Sample_at_{:.2e}Hz.png'.format(repitition_rate), dpi=100, bbox_inches='tight')
plt.show()

# 重建图像（翻z）
for i in range(galvo_steps):
    start, end = rising_edges[i], rising_edges[i+1]
    mid = np.ceil(np.mean([start, end])).astype(np.int32)   # 遇到0.5无条件向上进位

    z_down = sampled_pixels[start:mid]
    z_up = sampled_pixels[mid:end]
    
    # 上下颠倒z_up
    z_up = z_up[::-1]

    # 重建z轴: 偶数行是z_down, 奇数行是z_up
    reconstructed_z = np.zeros((z_axis_length)) # 用z軸最長長度作為重組圖像的z軸長度
    
    reconstructed_z[::2][:len(z_down)] = z_down
    reconstructed_z[1::2][:len(z_up)] = z_up
    reconstructed_img[:, i] = reconstructed_z

# 显示完整图像
plt.figure(figsize=(6,6))   #正方形图片
plt.imshow(reconstructed_img, cmap='magma', aspect='auto')
plt.title("Reconstruct_Z-Flipped. Repetition Rate: {:.2e} Hz".format(repitition_rate))
plt.savefig('./Figure/Reconstruct_Z-Flipped_YZ_plane_FL_Sample_at_{:.2e}Hz.png'.format(repitition_rate), dpi=100, bbox_inches='tight')
plt.show()
# %% 激光采样点和荧光球样本 动画
def animation_yz_plane(data_table, plot_range, fl_sample, dot_size=3):
    x = data_table[plot_range[0]:plot_range[1], 0]  # 图上的x 轴是 galvo 轴
    y = data_table[plot_range[0]:plot_range[1], 1]  # 图上的y 轴是 tag 轴
    
    fig, ax = plt.subplots(figsize=(6,6))
    # 画出样本图
    plt.minorticks_on()
    plt.imshow(fl_sample, cmap='magma', extent=[0,size,-size,0])
    plt.xlabel('y-axis(galvo) position(μm)')
    plt.ylabel('z-axis(TAG Lens) position(μm)')
    plt.title("Laser Repetition Rate: {:.2e} Hz".format(repitition_rate))

    # 初始化一个空的点集合
    points = ax.scatter([],[], s=dot_size, linewidth=0)

    # 更新函数，用于动画(四个点一起画，加速动画形成)
    def update(i):
        add = 10 if i*10 + 10 < len(x) else len(x) - i*10
        ax.scatter(x[i*4:i*4+add], y[i*4:i*4+add], s=dot_size, linewidth=0, color='#2CFFEC')

    # 创建一个动画
    ani = animation.FuncAnimation(fig, update, frames=range(np.ceil(len(x)/10).astype(np.int32)), interval=10)
    ani.save('./Figure/YZ_plane_FL_Sample_at_{:.2e}Hz.mp4'.format(repitition_rate), dpi=100, writer='ffmpeg', fps=60)

# animation_yz_plane(data_table, [0, adj_ratio*4000], fl_sample, dot_size=2)

# # %%
# # 分析真實Galvo電壓位置對應情況
# galvo_pos = pd.read_csv('/Users/richard/Desktop/Galvo_IO_test.csv')

# plt.figure(figsize=(8,6))
# scatter1 = plt.scatter(galvo_pos.iloc[3150:3300,0], galvo_pos.iloc[3150:3300, 1], s=4, linewidths=0, label='Input Voltage')
# scatter2 = plt.scatter(galvo_pos.iloc[3150:3300,0], galvo_pos.iloc[3150:3300,3],s=4,linewidths=0, label='Output (True Location)')

# # create legend
# legend1 = mlines.Line2D([], [], color=scatter1.get_facecolor(), marker='.', linestyle='None',
#                           markersize=10, label='Input Voltage')
# legend2 = mlines.Line2D([], [], color=scatter2.get_facecolor(), marker='.', linestyle='None',
#                           markersize=10, label='Output (True Location)')

# plt.legend(handles=[legend1, legend2])
# plt.xlabel('Time(ms)')
# plt.minorticks_on()
# plt.grid(True, linestyle='--', linewidth=0.4)
# plt.savefig('./Figure/Galvo_Position.png', dpi=150, bbox_inches='tight')
# %% TAG TTL 信號頻率變化對採樣點造成的影響，以動畫呈現

ttl_freq_lst = np.linspace(69e3,70e3,500)

def plot_yz_plane_fl_sample_animation(ttl_freq, data_table, plot_range, fl_sample, dot_size=3):
    x = data_table[plot_range[0]:plot_range[1], 0]  # 图上的x 轴是 galvo 轴
    y = data_table[plot_range[0]:plot_range[1], 1]  # 图上的y 轴是 tag 轴
    
    plt.figure(figsize=(6,6))
    plt.minorticks_on()
    plt.imshow(fl_sample, cmap='magma', extent=[0,size,-size,0])

    plt.scatter(x, y, s=dot_size, linewidth=0, label='y-z plane sampling point')
    plt.xlabel('y-axis(galvo) position(μm)')
    plt.ylabel('z-axis(TAG Lens) position(μm)') 
    # plt.legend()
    plt.title("TTL Frequency: {:.2e} Hz".format(ttl_freq))
    # plt.savefig('./Figure/YZ_plane_FL_Sample_at_{:.2e}Hz.png'.format(repitition_rate), dpi=200, bbox_inches='tight')
    # plt.show()

# 創建一個用於畫動畫的畫布
fig, ax = plt.subplots(figsize=(6,6))

ims = []

plot_range = [0, adj_ratio*3500]
dot_size = 1

for ttl_freq in ttl_freq_lst:

    data_table = np.zeros((simulation_steps, 3))    # y, z, ttl

    # 計算循環開始
    for i in range(simulation_steps):
        timeline = i / repitition_rate  # 打光時間

        y_pos = galvo_y_pos(timeline, ttl_freq)
        z_pos, ttl_signal = tag_z_pos(timeline, ttl_freq)

        data_table[i] = [y_pos, z_pos, ttl_signal]

    x = data_table[plot_range[0]:plot_range[1], 0]
    y = data_table[plot_range[0]:plot_range[1], 1]
    
    im = ax.scatter(x, y, s=dot_size, color='cyan', linewidth=0)
    im2 = ax.imshow(fl_sample, cmap='magma', extent=[0,size,-size,0]) # 將imshow的輸出儲存為im2
    plt.minorticks_on()

    ttl = plt.title("TTL Frequency: {:.6e} Hz".format(ttl_freq))  # ttl_freq 不是列表，所以不能用索引
    ims.append([im, im2, ttl])  # 每一幀動畫包含的藝術家元素，這裡需要將所有的藝術家元素包括到這個列表中

print('Create Animation...')

# 創建動畫
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
ani.save('./Figure/Diff_TTL.mp4', dpi=100, writer='ffmpeg', fps=10)

print('Complete.')

# %%
