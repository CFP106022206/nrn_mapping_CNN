# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import time

fc_id = 'TH-F-000091' #'Cha-F-800086'
em_id = '331662710'#'5812981264'
aug = '_aug6'    

em_path = './data/standard_views/EM/' + em_id + '_views' + aug + '.npz'
fc_path = './data/standard_views/FC/' + fc_id + '_views' + aug + '.npz'
output_path = './Figure/predict_3view/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
# %%
em_views = np.load(em_path)['views']
fc_views = np.load(fc_path)['views']


def plot_pairs(em_views, fc_views, em_id, fc_id, outputname=''):
    em_size = em_views.shape[1]
    fc_size = fc_views.shape[1]

    fig = plt.figure(figsize=(10,6))
    fig.suptitle(f'FC: {fc_id}_EM: {em_id}', fontsize=16)
    for i in range(3):
        plt.subplot(2,3,i+1)
        # 画FC
        plt.imshow(fc_views[i], cmap='magma')
        plt.xticks([0,fc_size])
        plt.yticks([0,fc_size])
        plt.gca().invert_yaxis()  # 反转y轴

        # 畫EM
        plt.subplot(2,3,i+4)
        plt.imshow(em_views[i], cmap='magma')
        plt.xticks([0,em_size])
        plt.yticks([0,em_size])
        plt.gca().invert_yaxis()  # 反转y轴

    plt.savefig(output_path+f'{fc_id}_{em_id}'+outputname+'.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_pairs(em_views, fc_views, em_id, fc_id)

# 嘗試統一pair圖片的尺寸

def _to_uint8_views(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    if v.dtype == np.uint8:
        return v
    vf = v.astype(np.float32, copy=False)
    vmax = float(np.nanmax(vf)) if vf.size else 0.0
    # 常见：如果是 0~1 浮点，就转 0~255
    if vmax <= 1.0:
        vf = np.round(vf * 255.0)
    vf = np.clip(vf, 0.0, 255.0)
    return vf.astype(np.uint8)


def _pad_to_same_size(fc_views: np.ndarray, em_views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把较小的一侧用黑边补齐到较大的一侧尺寸（居中）。输入/输出都是 (3,H,W)。"""
    fc = np.asarray(fc_views)
    em = np.asarray(em_views)

    target = max(int(fc.shape[1]), int(em.shape[1]))

    def _pad(v: np.ndarray) -> np.ndarray:
        _, h, w = v.shape   # h=w, 因为输入的fc和em都是正方形的
        pad = target - h
        if pad < 0:
            raise ValueError(f"target smaller than current: current=({h},{w}) target=({target},{target})")
        top = pad // 2
        bottom = pad - top
        left = pad // 2
        right = pad - left
        return np.pad(v, ((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)

    return _pad(fc), _pad(em)

def _resize_to_50(views: np.ndarray, out_hw: tuple[int, int] = (50, 50)) -> np.ndarray:
    """把 (3,H,W) 下采样到 (3,out_h,out_w)。

    使用“自适应 max pooling”（每个输出像素取对应输入块的最大值），
    对稀疏线条/骨架图更不容易产生断裂。
    """

    v = _to_uint8_views(views)

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    h, w = int(v.shape[1]), int(v.shape[2])

    if h == out_h and w == out_w:
        return v

    # 如果出现比目标还小的情况：直接 padding 到目标尺寸（居中补黑边），不做上采样，避免结构被复制/变粗。
    if h < out_h or w < out_w:
        pad_h = max(out_h - h, 0)
        pad_w = max(out_w - w, 0)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        vv = np.pad(v, ((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)
        # 安全裁切：避免 padding 后仍超出目标
        return vv[:, :out_h, :out_w].astype(np.uint8, copy=False)

    # 自适应 max pooling（向量化）：
    # 仍然是用 y0=(oy*h)//out_h, y1=((oy+1)*h)//out_h 的分箱方式，
    y_starts = (np.arange(out_h, dtype=np.int64) * h) // out_h  # (out_h,)
    x_starts = (np.arange(out_w, dtype=np.int64) * w) // out_w  # (out_w,)

    # 先沿 y 方向做分段 maxpool： (3,H,W) -> (3,out_h,W)
    tmp = np.maximum.reduceat(v, y_starts, axis=1)
    # 再沿 x 方向做分段 maxpool： (3,out_h,W) -> (3,out_h,out_w)
    out = np.maximum.reduceat(tmp, x_starts, axis=2)
    return out.astype(np.uint8, copy=False)

# def _resize_to_50(views: np.ndarray, out_hw: tuple[int, int] = (50, 50)) -> np.ndarray:
#     """把 (3,H,W) 下采样到 (3,out_h,out_w)。

#     使用“自适应 max pooling”（每个输出像素取对应输入块的最大值），
#     对稀疏线条/骨架图更不容易产生断裂。
#     """

#     v = _to_uint8_views(views)
#     if v.ndim != 3 or v.shape[0] != 3:
#         raise ValueError(f"Expect (3,H,W). got {v.shape}")

#     out_h, out_w = int(out_hw[0]), int(out_hw[1])
#     h, w = int(v.shape[1]), int(v.shape[2])

#     if h == out_h and w == out_w:
#         return v

#     # 如果出现比目标还小的情况：直接 padding 到目标尺寸（居中补黑边），不做上采样，避免结构被复制/变粗。
#     if h < out_h or w < out_w:
#         pad_h = max(out_h - h, 0)
#         pad_w = max(out_w - w, 0)
#         top = pad_h // 2
#         bottom = pad_h - top
#         left = pad_w // 2
#         right = pad_w - left
#         vv = np.pad(v, ((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)
#         # 如果只在一个维度 padding，另一维可能仍然 > 目标（理论上不会），这里做安全裁切
#         return vv[:, :out_h, :out_w].astype(np.uint8, copy=False)

#     out = np.zeros((3, out_h, out_w), dtype=np.uint8)

#     # 自适应 max pooling
#     for oy in range(out_h):
#         y0 = (oy * h) // out_h
#         y1 = ((oy + 1) * h) // out_h
#         if y1 <= y0:
#             y1 = min(y0 + 1, h)
#         for ox in range(out_w):
#             x0 = (ox * w) // out_w
#             x1 = ((ox + 1) * w) // out_w
#             if x1 <= x0:
#                 x1 = min(x0 + 1, w)
#             # 对三张 view 同时取 max
#             out[:, oy, ox] = v[:, y0:y1, x0:x1].max(axis=(1, 2))

#     return out


# ---- Step 1) padding 到相同尺寸（黑边补齐） ----
print('[before] fc_views shape:', np.asarray(fc_views).shape, 'em_views shape:', np.asarray(em_views).shape)
fc_pad, em_pad = _pad_to_same_size(fc_views, em_views)
print('[pad]    fc_pad shape :', fc_pad.shape, 'em_pad shape :', em_pad.shape)
plot_pairs(em_pad, fc_pad, em_id, fc_id, outputname='_pad')

# ---- Step 2) 一起 resize 到 50x50 ----
st = time.time()
fc_50 = _resize_to_50(fc_pad, (50, 50))
em_50 = _resize_to_50(em_pad, (50, 50))
print('[50x50]  fc_50 shape  :', fc_50.shape, 'em_50 shape  :', em_50.shape)
print('Time cost:', time.time() - st, 'seconds')
# 可视化检查（输出一张统一尺寸后的图）
plot_pairs(em_50, fc_50, em_id, fc_id, outputname='_pad_then_resize50')

# %% 找出舊的圖
old_path = './data/mapping_data/'
# 讀取裡面所有的npz檔案
fc_id_lst, em_id_lst, fc_views_lst, em_views_lst = [], [], [], []

for file in os.listdir(old_path):
    if file.endswith('.npz'):
        data = np.load(os.path.join(old_path, file))
        fc_id_lst.append(data['id_a'])
        em_id_lst.append(data['id_b'])
        fc_views_lst.append(data['views_a'])
        em_views_lst.append(data['views_b'])
# 合併array
fc_id_lst = np.concatenate(fc_id_lst,axis=0)
em_id_lst = np.concatenate(em_id_lst,axis=0)
fc_views_all = np.concatenate(fc_views_lst, axis=0)
em_views_all = np.concatenate(em_views_lst, axis=0)

# 找到指定的fc_id和em_id
for i in range(len(fc_id_lst)):
    if str(fc_id_lst[i]) == fc_id and str(em_id_lst[i]) == em_id:
        fc_views = fc_views_all[i]
        em_views = em_views_all[i]
        break

# 画图
plot_pairs(em_views, fc_views, em_id, fc_id, outputname='_target_aligned')
# %%


# %%
