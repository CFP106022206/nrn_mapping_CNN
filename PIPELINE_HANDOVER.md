# 全流程使用说明

这份文档说明 `nrn_mapping_CNN` 里从 `SWC` 原始数据到 `descriptor`、`pair matching`、`standard view` 的完整流水线，目标是让第一次接手的人可以直接按步骤运行，而不是先读完全部代码。

## 1. 流水线总览

主入口脚本是 [swc_pair_and_draw.py](swc_pair_and_draw.py)。它按顺序做四件事：

1. 读取 FC / EM 的原始 `SWC` 文件，生成 descriptor。
2. 用 descriptor 做 candidate matching，输出 pair CSV。
3. 根据 pair CSV，渲染 FC 的标准三视图。
4. 根据同一份 pair CSV，渲染 EM 的标准三视图。

整个过程不会覆盖原始 `SWC` 数据；只会在你指定的输出目录里生成中间文件和结果文件。

## 2. 关键目录

默认约定如下：

- 原始 `SWC`：`./data/SWC/FC`、`./data/SWC/EM`
- FC descriptor 输出：`./data/descriptors_FC`
- EM descriptor 输出：`./data/descriptors_EM`
- pair 输出：`./data/pairs_label`
- 标准视图输出：`./data/standard_views/FC`、`./data/standard_views/EM`

如果你的目录结构不同，可以在命令行参数里改掉。

## 3. 每一步分别做什么

### 3.1 Descriptor 提取

这一步把每个 `SWC` 文件转换成 descriptor 文件，典型输出包括：

- `descriptors_FC.parquet` / `descriptors_EM.parquet`
- `centroids_FC.npy` / `centroids_EM.npy`
- `eigvecs_FC.npy` / `eigvecs_EM.npy`
- `eigvals_ratio_FC.npy` / `eigvals_ratio_EM.npy`
- `neuron_ids_FC.npy` / `neuron_ids_EM.npy`

这一步由 [swc_descriptor_batch.py](swc_descriptor_batch.py) 里的 `batch_run()` 完成，主脚本里是第 1、2 步。

### 3.2 Candidate Matching

这一步由 [candidate_matching.py](candidate_matching.py) 完成。

它会读取 FC / EM descriptor，先按 centroid 距离过滤，再按 inertia ratio 距离过滤，最后再做 orientation 过滤，输出 pair CSV：

- `pairs_FC_EM.csv`

注意：这份 pair CSV 是这个 pipeline 的标准中间产物，不是 `EMxFC_all.csv` 那一类下游 pseudo-label 文件。

### 3.3 Standard View 渲染

这一步由 [standard_draw.py](standard_draw.py) 完成。

它读取 pair CSV，然后分别从 FC / EM 的原始 `SWC` 里把 pair 对应的神经元渲染成标准三视图，输出为 `npz` 文件：

- `data/standard_views/FC/*_views.npz`
- `data/standard_views/EM/*_views.npz`

每个 `npz` 里通常包含：

- `nid`
- `views`
- `grid_size`

## 4. 一键运行全流程

### 4.1 命令行入口

这个全流程是通过 [swc_pair_and_draw.py](swc_pair_and_draw.py) 直接用命令行启动的。最基本的调用形式是：

```bash
python swc_pair_and_draw.py \
    --fc_swc_dir ./data/SWC/FC \
    --em_swc_dir ./data/SWC/EM
```

建议在仓库根目录下执行，因为默认的输入和输出路径都是按当前目录解析的。

从仓库根目录运行：

```bash
cd /cluster/home/ming/Project/nrn_mapping_CNN
conda run -n ming python swc_pair_and_draw.py \
    --fc_swc_dir ./data/SWC/FC \
    --em_swc_dir ./data/SWC/EM
```

这条命令会使用默认输出目录，依次生成 descriptor、pair、standard view。

如果你的数据目录不在默认位置，可以显式指定：

```bash
conda run -n ming python swc_pair_and_draw.py \
    --fc_swc_dir /path/to/FC \
    --em_swc_dir /path/to/EM \
    --descriptor_root /path/to/output_root \
    --pairs_out_dir /path/to/output_root/pairs_label \
    --views_root /path/to/output_root/standard_views
```

## 5. 常用参数

`swc_pair_and_draw.py` 的主要参数：

- `--fc_swc_dir`：FC 原始 `SWC` 目录
- `--em_swc_dir`：EM 原始 `SWC` 目录
- `--descriptor_root`：descriptor 输出根目录，默认 `./data`
- `--pairs_out_dir`：pair CSV 输出目录，默认 `./data/pairs_label/`
- `--views_root`：标准视图输出根目录，默认 `./data/standard_views/`
- `--centroid_th`：candidate matching 的 centroid 距离阈值，默认 `100.0`
- `--ratio_th`：candidate matching 的 ratio 距离阈值，默认 `0.4`
- `--no-recursive`：只扫描顶层，不递归子目录
- `--fail-fast`：遇到单个 `SWC` 错误时立即停止
- `--scale_um_per_px`：标准视图缩放参数，默认 `5.0`
- `--normalize`：归一化方式，`max` 或 `p99`
- `--no-skip-existing`：即使文件已存在也重新渲染

## 6. 输出结果怎么检查

如果全流程成功，通常你会看到以下类型的文件：

- `data/descriptors_FC/descriptors_FC.parquet`
- `data/descriptors_EM/descriptors_EM.parquet`
- `data/pairs_label/pairs_FC_EM.csv`
- `data/standard_views/FC/*.npz`
- `data/standard_views/EM/*.npz`

你可以用下面的方式快速确认：

```bash
ls data/descriptors_FC
ls data/descriptors_EM
ls data/pairs_label
ls data/standard_views/FC | head
ls data/standard_views/EM | head
```

## 7. 哪些文件不属于这个 pipeline

下面这些 `EMxFC_*` 文件属于别的后处理或 pseudo-label 流程，不是这条主 pipeline 的标准输出：

- `EMxFC_all.csv`
- `EMxFC_all_filtered.csv`
- `EMxFC_all_high_confidence.csv`
- `EMxFC_1000K.csv`
- `EMxFC_5000K.csv`
- `EMxFC_10KK.csv`
- `EMxFC_1000-5000K.csv`
- `EMxFC_6KK_last.csv`
- `EMxFC_all_0_rk20.csv`
- `EMxFC_shuffle.csv`

如果你在这条 pipeline 里看到这些名字，通常是接错了文件来源。

## 8. 只跑某一段时怎么做

### 8.1 只跑 candidate matching

前提是 descriptor 已经存在：

```bash
conda run -n ming python candidate_matching.py \
    --fc_dir ./data/descriptors_FC \
    --em_dir ./data/descriptors_EM \
    --out_dir ./data/pairs_label
```

### 8.2 只跑 standard view 渲染

前提是 pair CSV 已经存在：

```bash
conda run -n ming python standard_draw.py \
    --swc_dir ./data/SWC/FC \
    --neuron_list ./data/pairs_label/pairs_FC_EM.csv \
    --csv_id_col fc_id \
    --output_dir ./data/standard_views/FC \
    --format npz
```

EM 侧同理，把 `--swc_dir` 和 `--csv_id_col` 改成 `./data/SWC/EM` 和 `em_id`。

## 9. 回归测试脚本

仓库里还有 [swc_pair_and_draw_test.py](swc_pair_and_draw_test.py)，它的目标不是覆盖原始数据，而是在临时目录里复跑完整流程，然后和现有归档结果做比较。

如果你想确认这条 pipeline 有没有被改坏，可以用：

```bash
conda run -n ming python swc_pair_and_draw_test.py \
    --fc_swc_dir ./data/SWC/FC \
    --em_swc_dir ./data/SWC/EM
```

这个测试脚本会：

- 在临时目录里生成 descriptor、pair、standard views
- 不覆盖原始归档
- 默认只检查 pair CSV 的 schema，如果你另外提供真正的 pair 归档路径，也可以做严格对比

## 10. 常见问题

### Q: 为什么会报找不到 pair CSV？

先确认有没有跑过 [swc_pair_and_draw.py](swc_pair_and_draw.py) 的 matching 阶段。pair 文件是 `candidate_matching.py` 生成的，不是 `standard_draw.py` 生成的。

### Q: 为什么只有 descriptor 有文件，但 pair 没有？

通常是 matching 阶段没跑，或者 `--pairs_out_dir` 指到了别的目录。

### Q: 为什么 standard view 没有生成？

通常是 pair CSV 没有正确生成，或者 `--swc_dir` / `--neuron_list` 传错。

### Q: 运行时为什么不要直接覆盖原始目录？

因为这条 pipeline 的中间产物很多，建议全部写到输出目录或临时目录里，便于排查和回滚。

## 11. 建议的交接顺序

如果你是第一次接手，建议按这个顺序理解代码：

1. 先看 [swc_pair_and_draw.py](swc_pair_and_draw.py)
2. 再看 [candidate_matching.py](candidate_matching.py)
3. 再看 [standard_draw.py](standard_draw.py)
4. 最后看 [swc_descriptor_batch.py](swc_descriptor_batch.py) 和 [swc_descriptor.py](swc_descriptor.py)

这样最容易把“输入是什么、输出到哪里、哪一步负责什么”串起来。