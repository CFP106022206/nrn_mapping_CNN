import numpy as np
import os
import trimesh
from mayavi import mlab
from tvtk.api import tvtk
import pandas as pd
from pathlib import Path 
import copy


def _interpolate_swc_df(df: pd.DataFrame, max_dist: float = 0.5) -> pd.DataFrame:
    """
    將 SWC 的骨架在 DataFrame 內進行內插，使所有 parent-child 的距離 <= max_dist。
    回傳：包含新增節點、且已更新 parent 的新 df（欄位同原 SWC）。
    """
    if df.empty:
        return df.copy()

    swc_columns = ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']
    # 確保欄位齊全（有些 swc 的 type 欄位名可能不同）
    df = df.copy()
    df.columns = swc_columns

    nodes_dict = df.set_index('id').to_dict('index')
    new_nodes = []
    parent_updates = {}  # child_id -> new_parent_id
    next_node_id = int(df['id'].max()) + 1

    for _, child_node in df.iterrows():
        parent_id = int(child_node['parent'])
        if parent_id == -1:
            continue
        if parent_id not in nodes_dict:
            # 父節點不存在就跳過這段 edge
            continue

        parent_node = nodes_dict[parent_id]
        child_pos = np.array([child_node['x'], child_node['y'], child_node['z']], dtype=float)
        parent_pos = np.array([parent_node['x'], parent_node['y'], parent_node['z']], dtype=float)
        child_radius = float(child_node['radius'])
        parent_radius = float(parent_node['radius'])

        distance = np.linalg.norm(child_pos - parent_pos)
        if distance > max_dist:
            num_segments = int(np.ceil(distance / max_dist))
            num_new_points = num_segments - 1
            step_vec = (child_pos - parent_pos) / num_segments
            step_rad = (child_radius - parent_radius) / num_segments

            current_parent_id = parent_id
            for i in range(1, num_new_points + 1):
                new_pos = parent_pos + i * step_vec
                new_radius = parent_radius + i * step_rad
                new_nodes.append({
                    'id': next_node_id,
                    'type': int(child_node['type']),
                    'x': float(new_pos[0]),
                    'y': float(new_pos[1]),
                    'z': float(new_pos[2]),
                    'radius': float(new_radius),
                    'parent': int(current_parent_id),
                })
                current_parent_id = next_node_id
                next_node_id += 1

            # 原本 child 的 parent 改指向最後一個新節點
            parent_updates[int(child_node['id'])] = int(current_parent_id)

    if parent_updates:
        df['parent'] = df['id'].map(parent_updates).fillna(df['parent']).astype(int)

    if new_nodes:
        new_df = pd.DataFrame(new_nodes, columns=swc_columns)
        out = pd.concat([df, new_df], ignore_index=True)
    else:
        out = df

    # 依 id 排序讓檔案/資料更整潔
    out = out.sort_values(by='id').reset_index(drop=True)
    return out

# ——（可選）如果你還需要把內插後結果寫成檔案，保留這個方便的函式 ——
def interpolate_swc(input_path: str, output_path: str, max_dist: float = 0.5):
    swc_columns = ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']
    df = pd.read_csv(
        input_path, sep=r'\s+', comment='#', header=None, names=swc_columns,
        dtype={'id': int, 'parent': int}
    )
    out = _interpolate_swc_df(df, max_dist=max_dist)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out[swc_columns].to_csv(output_path, sep=' ', header=False, index=False, float_format='%.6f')

class Anatomical_analysis():
    """
    一個專注於使用 Mayavi 進行 3D 大腦結構與神經元視覺化的類別。
    """
    def __init__(self, root='Result/', neuropil_path='', template='FlyEM'):
        if template == 'FAFB':
            self.neuropil_path = 'FAFB_neuropil/'
        elif template == "FlyCircuit":
            self.neuropil_path = 'FlyCircuit_neuropil/'
            self.swc_path = 'FlyCircuit_skeleton/'
        elif template == 'FlyEM':
            self.neuropil_path = 'FlyEM_neuropil/'
            self.swc_path = 'FlyEM_skeleton/'

        self.neuropil_space_dict = {}
        self.neuropil_coord_dict = {}
        self.bounding_box_dict = {}
        self.xyzi = []

    def load_neuropil(self, neuropils):
        for neuropil in neuropils:
            if neuropil == 'whole brain':
                neuropil = 'ebo_ns_instbs_20081209.surf'
            if neuropil not in self.neuropil_space_dict:
                print(f"Loading neuropil: {neuropil}")
                neuropil_space = trimesh.load(f'{self.neuropil_path}{neuropil}.obj')
                self.neuropil_space_dict[neuropil] = neuropil_space

    def read_swc(self, file_name: str, omit_ratio: float = 0.0,
                 interpolate: bool = False, max_dist: float = 0.5,
                 return_df: bool = False):
        """
        從 .swc 檔案讀取神經元骨架的 3D 座標，可選擇在讀取時執行「內插」。

        Args:
            file_name (str): .swc 完整路徑
            omit_ratio (float): 忽略點的比例 (0~1)，用於降採樣
            interpolate (bool): 是否對骨架做內插 (使所有相鄰節點距離 <= max_dist)
            max_dist (float): 內插時的最大節點間距
            return_df (bool): True 則同時回傳處理後的 DataFrame

        Returns:
            xyz (list[list[float]]): [[x,y,z], ...]
            （若 return_df=True，則回傳 (xyz, df)）
        """
        swc_columns = ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']
        neu = pd.read_csv(file_name, delim_whitespace=True, header=None, comment='#')
        neu.columns = swc_columns

        if interpolate:
            neu = _interpolate_swc_df(neu, max_dist=max_dist)

        if omit_ratio > 0:
            keep = max(0.0, min(1.0, 1.0 - omit_ratio))
            neu = neu.sample(frac=keep, random_state=0).reset_index(drop=True)

        xyz = neu[['x', 'y', 'z']].to_numpy().tolist()
        if return_df:
            return xyz, neu
        return xyz

    # 其餘 methods（get_bounding_box_of_neuropil、visualize_neuropil、visualize_neuron、...）維持不變


    def get_bounding_box_of_neuropil(self, neuropil_space):
        """
        計算 3D 模型的邊界框。

        Args:
            neuropil_space (trimesh.Trimesh): Trimesh 物件。

        Returns:
            tuple: (xmin, xmax, ymin, ymax, zmin, zmax)。
        """
        x, y, z = neuropil_space.vertices.T
        bounding_box = (np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z))
        return bounding_box

    def visualize_neuropil(self, obj=['CA(R)', 'MB(R)'], create_new=False, color=(0.9, 0.9, 0.9), opacity=0.1):
        """
        使用 Mayavi 視覺化神經氈的 3D 網格模型。

        Args:
            obj (list): 要顯示的神經氈名稱列表。
            create_new (bool): 是否創建一個新的 Mayavi 場景。
            color (tuple): 模型的顏色 (R, G, B)。
            opacity (float): 模型的透明度。
        """
        if create_new:
            mlab.clf()
            fig = mlab.figure(bgcolor=(1, 1, 1))
        
        if obj:
            for neuropil_file in obj:
                if neuropil_file not in self.neuropil_space_dict:
                    self.load_neuropil([neuropil_file])
                
                neuropil = self.neuropil_space_dict[neuropil_file]
                x, y, z = neuropil.vertices.T
                mlab.triangular_mesh(x, y, z, neuropil.faces, color=color, opacity=opacity)

    def visualize_neuron(self, xyz, color=(1.0, 0.0, 0.0), size=100.0):
        """
        使用 Mayavi 將神經元座標視覺化為 3D 散點。

        Args:
            xyz (np.array): 神經元座標陣列 (N x 3)。
            color (tuple): 點的顏色 (R, G, B)。
            size (float): 點的大小。
        """
        if not isinstance(xyz, np.ndarray):
            xyz = np.array(xyz)
        
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
            print("Warning: Invalid or empty coordinate array provided to visualize_neuron.")
            return

        mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, scale_factor=size, mode='sphere')

    def visualize_density_by_density(self, density, obj='LH(R)', cmap='Reds', contour_num=4, template='FlyEM'):
        """
        根據已計算的密度數據進行視覺化。

        Args:
            density (np.array): 3D 密度陣列。
            obj (str): 用於定義邊界框的神經氈名稱。
            cmap (str): 等值面的顏色映射。
            contour_num (int): 等值面的數量。
            template (str): 使用的模板 ('FlyEM' 或 'FAFB')。
        """
        if obj not in self.neuropil_space_dict:
            self.load_neuropil([obj])
        
        neuropil = self.neuropil_space_dict[obj]
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_bounding_box_of_neuropil(neuropil)
        x_num, y_num, z_num = density.shape
        
        xi, yi, zi = np.mgrid[xmin:xmax:complex(0, x_num),
                              ymin:ymax:complex(0, y_num),
                              zmin:zmax:complex(0, z_num)]
        
        mlab.contour3d(xi, yi, zi, density, opacity=0.5, colormap=cmap, contours=contour_num)

    def visualize_mlab(self):
        """
        顯示 Mayavi 視窗。
        """
        mlab.show()


# --- 主程式執行區塊 ---
def get_EM_file_name_dict():
    """ 
    用於 plot_neuron_neuropil_for_EM_FC_comparison 的輔助函數，從檔名中解析 ID。
    """
    path = 'C:/Users/cockr/Project/eFlyPlotv2p1/eFlyPlotv2p1/Data/skeleton_via_ChaoChung/'
    files = [i for i in os.listdir(path) if 'Fix_from_FlyEM_um' in i]
    to_FlyCircuit_dict = {copy.deepcopy(i).split("-")[2].split("_")[0]: i for i in files if "FlyCircuit" in i}
    to_FlyEM_dict = {copy.deepcopy(i).split("-")[2].split("_")[0]: i for i in files if "FlyEM" in i}
    return to_FlyCircuit_dict, to_FlyEM_dict

def plot_neuron_neuropil_for_EM_FC_comparison(fc_neuron='', em_neuron='', template='FlyCircuit'):
    """
    比較並視覺化 FlyCircuit 和 FlyEM 中的神經元。
    """
    em_to_fc_swc_path = 'C:/Users/cockr/Project/eFlyPlotv2p1/eFlyPlotv2p1/Data/skeleton_via_ChaoChung/'
    to_FC_dict, _ = get_EM_file_name_dict()

    if template == "FlyCircuit":
        a = Anatomical_analysis(template='FlyCircuit')
        neuropils = ['whole brain', 'al_3_instd_r', 'mb_4_instd_r', 'lh_25_instd_r']
        a.visualize_neuropil(neuropils, create_new=True, opacity=0.1)
        
        # 視覺化 FlyCircuit 神經元 (紅色)
        xyz_fc = a.read_swc(file_name=f'{a.swc_path}{fc_neuron}')
        a.visualize_neuron(np.array(xyz_fc, dtype=int), color=(1, 0, 0), size=1500)
        
        # 視覺化對應的 FlyEM 神經元 (藍色)
        em_file_name = to_FC_dict.get(str(em_neuron))
        if em_file_name:
            xyz_em = a.read_swc(file_name=f'{em_to_fc_swc_path}{em_file_name}')
            a.visualize_neuron(np.array(xyz_em, dtype=int), color=(0, 0, 1), size=1500)
        
        a.visualize_mlab()

if __name__ == '__main__':
    # 建立一個 Anatomical_analysis 物件
    vis_tool = Anatomical_analysis(template='FlyEM')

    # 定義要視覺化的神經氈
    neuropils_to_show = ["AL_ALL(R)", 'MB(R)', 'LH(R)']
    
    # 創建一個新的 3D 場景並繪製神經氈
    vis_tool.visualize_neuropil(obj=neuropils_to_show, create_new=True, opacity=0.15)
    
    # 讀取一個神經元的 SWC 檔案 (請確認檔案路徑和名稱正確)
    # 這裡使用一個假想的檔案名稱 '5813012892.swc' 作為範例
    try:
        neuron_coordinates = vis_tool.read_swc(f'{vis_tool.swc_path}301318641.swc', interpolate=True, max_dist=0.5)
        
        # 在場景中繪製神經元
        vis_tool.visualize_neuron(xyz=neuron_coordinates, color=(0.8, 0.2, 0.2), size=80)
    except FileNotFoundError:
        print("範例 SWC 檔案 '5813012892.swc' 不存在，將只顯示神經氈。")

    # 顯示 Mayavi 視窗
    vis_tool.visualize_mlab()

    # 執行另一個比較範例
    # 注意：此函數需要特定的檔案路徑和數據，可能無法直接運行
    # plot_neuron_neuropil_for_EM_FC_comparison(
    #     fc_neuron='G0239-F-000001.swc', 
    #     em_neuron='5813068729', 
    #     template='FlyCircuit'
    # )