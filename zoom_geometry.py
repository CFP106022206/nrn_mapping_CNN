# -*- coding: utf-8 -*-
"""

用於result analysis fig 3 拼接圖片

zoom_geometry.py — 由 main / zoom 兩張渲染圖的 metadata 算出:
    1. 放大區在主圖上的來源框 (像素座標)
    2. 兩者各自的 px_per_um

前提: 兩張圖都用平行投影, 且視角相同 (view_name 一致)。
"""

import numpy as np

# view_name -> (影像水平軸對應的世界軸, 影像垂直軸對應的世界軸)
VIEW_AXES = {
    'xy': (0, 1),   # 由 +z 往下看: 影像 x=世界x, 影像 y=世界y
    'xz': (0, 2),   # 由 +y 看過去: 影像 x=世界x, 影像 y=世界z
    'yz': (1, 2),   # 由 +x 看過去: 影像 x=世界y, 影像 y=世界z
}


def compute_zoom_rect(main, zoom):
    """
    回傳放大區在「主圖像素座標」中的矩形 (x0, y0, w, h),
    原點在左上角 (符合 imshow 的預設)。

    main, zoom : read_scale_info() 的回傳 dict
    無法計算時回傳 None。
    """
    need = ('view_name', 'parallel_scale', 'image_w', 'image_h',
            'focal_x', 'focal_y', 'focal_z')
    for d in (main, zoom):
        if d is None or any(k not in d for k in need):
            return None

    vn = main.get('view_name', '')
    if vn != zoom.get('view_name', '') or vn not in VIEW_AXES:
        return None

    h_ax, v_ax = VIEW_AXES[vn]
    fp = lambda d, ax: d[('focal_x', 'focal_y', 'focal_z')[ax]]

    Wm, Hm = main['image_w'], main['image_h']
    Sm = main['parallel_scale']
    ppu_main = Hm / (2.0 * Sm)          # 主圖: 每世界單位幾像素

    Wz, Hz = zoom['image_w'], zoom['image_h']
    Sz = zoom['parallel_scale']

    # 放大區中心相對主圖中心的世界位移
    dh = fp(zoom, h_ax) - fp(main, h_ax)
    dv = fp(zoom, v_ax) - fp(main, v_ax)

    cx = Wm / 2.0 + dh * ppu_main
    cy = Hm / 2.0 - dv * ppu_main       # 影像 y 軸向下, 故取負

    # 放大區的世界範圍: 垂直半高 = Sz, 水平半寬 = Sz * (Wz/Hz)
    half_h = Sz * ppu_main
    half_w = Sz * (Wz / Hz) * ppu_main

    return (cx - half_w, cy - half_h, 2 * half_w, 2 * half_h)


def zoom_factor(main, zoom):
    """放大倍率 = zoom 的 px_per_um / main 的 px_per_um"""
    if main is None or zoom is None:
        return None
    if not main.get('px_per_um') or not zoom.get('px_per_um'):
        return None
    return zoom['px_per_um'] / main['px_per_um']
