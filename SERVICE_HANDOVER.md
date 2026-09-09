# 網頁比對服務交接說明

使用者上傳一個 SWC → 在另一個資料庫中找出最相似的 n 個神經元 → 輸出 CSV。

前置的兩段流程請看 [DRAW_PIPELINE_HANDOVER.md](DRAW_PIPELINE_HANDOVER.md)（SWC → descriptor → 三視圖）
和 [MODEL_PIPELINE_HANDOVER.md](MODEL_PIPELINE_HANDOVER.md)（模型訓練與離線 prediction）。
這份文件講的是把那兩段包成線上服務的部分。

---

## 1. 一句話總結

`nrn_service/` 是一個常駐服務物件，啟動時把模型、descriptor 索引、三視圖全部載入記憶體，
之後每次查詢 0.2～0.8 秒。curated 資料庫（`data/`）在服務執行期間**只讀不寫**；
使用者上傳的東西一律進 `user_data/`，經人工確認才會並入 `data/`。

---

## 2. 快速開始

有兩個入口，做的事完全一樣，差別只在誰來呼叫（詳見 §3）：

```bash
# 入口 A：命令列。你自己測試用，不需要安裝任何額外套件
python3 match_cli.py --swc /path/to/neuron.swc --query_side FC --out result.csv
python3 match_cli.py --swc /path/to/body.swc  --query_side EM --out result.csv
python3 match_cli.py --neuron_id Trh-F-000040 --query_side FC --out result.csv

# 入口 B：Web 服務。給前端網頁呼叫用
pip install fastapi uvicorn python-multipart
uvicorn app:app --host 0.0.0.0 --port 8000
```

輸出 CSV 固定四欄：

```
source_id,target_id,similarity_score,rank
Trh-F-000040,1135311365,0.69388247,1
Trh-F-000040,820856726,0.5896449,2
...
```

`rank` 從 1 開始，依 `similarity_score` 由大到小；同分時用 `target_id` 決定順序，
所以同樣的輸入永遠得到同樣的排序。

---

## 3. 這些檔案分別在做什麼

整個系統分成兩層：

```
                 瀏覽器（前端網頁）
                        │  使用者選了一個 neuron.swc
                        ▼  HTTP POST /match
   ┌──────────────────────────────────────────────────┐
   │  app.py            ← 「櫃台」                     │
   │  只做四件事：                                      │
   │    1. 接收 HTTP 上傳的檔案                         │
   │    2. 呼叫 nrn_service                            │
   │    3. 把回傳的表格轉成 CSV 文字                     │
   │    4. 用 HTTP 回傳出去                             │
   │  不含任何比對邏輯。                                 │
   └──────────────────────────────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────────┐
   │  nrn_service/      ← 「工廠」，真正在做事的部分      │
   │  驗證 → descriptor → 三視圖 → 候選初篩 → 打分 → 排序 │
   │  完全不知道 HTTP 是什麼，只認「檔案內容 → 結果表格」  │
   └──────────────────────────────────────────────────┘
                        ▲
                        │  同一個工廠的另一個入口
   ┌──────────────────────────────────────────────────┐
   │  match_cli.py      ← 終端機測試用           │
   │  不經過網路，所以不需要安裝任何 web 套件            │
   └──────────────────────────────────────────────────┘
```

**為什麼需要 `app.py`？** 前端的網頁是在使用者的瀏覽器裡跑的，你的程式是在伺服器上跑的，
兩者之間必須有一個「會講 HTTP 的窗口」才能溝通。`app.py` 就是那個窗口。
把它拿掉，`nrn_service/` 仍然完整可用（用 `match_cli.py` 跑），只是前端叫不到。

### 檔案清單

| 檔案 | 角色 | 負責什麼 |
|---|---|---|
| `nrn_service/config.py` | 後端 | ★ 所有參數的唯一真實來源（畫圖 / 模型 / 初篩 / 驗證 / 路徑）|
| `nrn_service/validation.py` | 後端 | 上傳檔驗證、檔名清理、空圖 guard |
| `nrn_service/view_store.py` | 後端 | 三視圖的 memmap 常駐儲存 |
| `nrn_service/db_index.py` | 後端 | descriptor + KDTree + sha256 索引 |
| `nrn_service/matching.py` | 後端 | 單顆 vs 一整個資料庫的候選初篩 |
| `nrn_service/scoring.py` | 後端 | Siamese CNN 打分 |
| `nrn_service/upload_store.py` | 後端 | `user_data/` 的落地與 sqlite registry |
| `nrn_service/service.py` | 後端 | 主流程 `NeuronMatchService.query()` |
| `match_cli.py` | 入口 | 命令列（不需要 web 套件）|
| `app.py` | 入口 | HTTP 窗口（需要 fastapi）|
| `tools/pack_views.py` | 維運 | 把 standard_views 打包成 memmap |
| `tools/build_curated_index.py` | 維運 | 建 curated 資料庫的 sha256 索引 |
| `tools/promote_upload.py` | 維運 | 人工確認後把上傳檔並入 curated 資料庫 |

---

## 4. 和前端的介面約定

**約定**：前端傳一個 `.swc` 檔案，服務回傳一份 CSV。

### 4.1 端點

| 方法 | 路徑 | 用途 |
|---|---|---|
| `POST` | `/match` | 上傳 SWC，回傳 CSV ← **前端只需要這一個** |
| `POST` | `/match/by_id` | 用資料庫既有的 neuron id 查詢（測試用）|
| `GET` | `/health` | 服務是否正常、資料庫大小、模型名稱 |
| `GET` | `/sides` | 可選的資料庫（目前只有 FC / EM）|

### 4.2 `/match` 的參數

用 `multipart/form-data` 送出：

| 欄位 | 必填 | 說明 |
|---|---|---|
| `file` | ✅ | 使用者上傳的 `.swc` |
| `query_side` | ✅ **目前必填** | `"FC"` 或 `"EM"`，這顆神經元屬於哪一側 |
| `target_side` | ✗ | 要搜尋哪個資料庫，不給就是另一側 |
| `top_n` | ✗ | 回傳幾對，預設 5 |
| `mirror` | ✗ | 左右腦鏡像，目前傳 `true` 會回 501 |

### 4.3 前端要寫的程式

```javascript
const form = new FormData();
form.append("file", swcFile);        // 使用者選的 .swc
form.append("query_side", "FC");     // 目前必填

const res = await fetch("http://<伺服器位址>:8000/match", {
  method: "POST",
  body: form,
});

if (res.ok) {
  const csv = await res.text();      // 就是最終的 CSV 內容
  // 想看警告訊息（中文，做過百分比編碼）：
  const w = res.headers.get("X-Warnings");
  if (w) console.warn(decodeURIComponent(w));
} else {
  const err = await res.json();      // { detail: "錯誤原因（中文）" }
  alert(err.detail);
}
```

用 `curl` 測試等價於：

```bash
curl -F "file=@neuron.swc" -F "query_side=FC" \
     http://localhost:8000/match -o result.csv
```

### 4.4 回傳內容

**成功（HTTP 200）**，`Content-Type: text/csv`，內容就是純 CSV 文字：

```
source_id,target_id,similarity_score,rank
Trh-F-000040,1135311365,0.69388247,1
Trh-F-000040,820856726,0.5896449,2
...
```

另外附幾個 header 供前端顯示狀態：

| Header | 內容 |
|---|---|
| `X-Source-Id` | 這次查詢的神經元 id |
| `X-Query-Side` / `X-Target-Side` | 查詢側 / 被搜尋的資料庫 |
| `X-Candidates` / `X-Scored` | 初篩出的候選數 / 實際打分數 |
| `X-Resolution` | 這顆是怎麼被解析出來的（是否命中資料庫）|
| `X-Elapsed-Seconds` | 耗時 |
| `X-Upload-Id` | 上傳暫存區的編號，回報問題時給這個 |
| `X-Warnings` | 警告訊息，**百分比編碼**，前端要 `decodeURIComponent` |

> 為什麼警告要編碼：HTTP header 只允許 latin-1，中文直接塞會讓整個回應
> 拋 `UnicodeEncodeError`。所以一律 `quote()` 後再放。

**失敗**，回傳 JSON `{"detail": "中文說明"}`，前端直接顯示給使用者即可：

| 狀態碼 | 什麼情況 | 訊息範例 |
|---|---|---|
| `400` | 上傳檔有問題 | 「座標數量級不對（max\|coord\| = 277120 > 5000）。這通常表示 SWC 還是原始的 nm 或 voxel 單位，必須先 warp 到 standard brain」 |
| `400` | 檔名不合法 | 「檔名含有不允許的字元」 |
| `404` | 找不到任何候選 | 「在目標資料庫中找不到任何幾何特徵相近的候選」（常見原因見 §12.1 右腦問題）|
| `501` | 用了尚未實作的功能 | 「mirror（左右腦鏡像）尚未實作」 |
| `503` | 服務還在啟動 | 「服務尚未完成啟動」（模型載入約 3 秒）|

### 4.5 部署

```bash
pip install fastapi uvicorn python-multipart
uvicorn app:app --host 0.0.0.0 --port 8000
```

啟動後就一直掛著，模型只載入一次。前端同事需要知道的只有**伺服器位址 + port + `/match`**。

**跨網域（CORS）**：如果前端網頁不是掛在同一個網域，瀏覽器會擋掉請求。
這需要在 `app.py` 加上 `CORSMiddleware` 並列出允許的來源。
**目前尚未加**，因為還不知道前端網頁最後會放在哪個網址。確定之後再加。

**不要用 `uvicorn --workers N`**：每個 worker 都會各自載入模型與整個資料庫索引，
記憶體會乘上 N。要擴充吞吐量請在前面放一層 queue。

## 5. 一次查詢做了什麼

1. **解析查詢對象**，依序比對（都沒中才走完整計算）：
   1. 內容 sha256 命中 `user_data/` → 直接重用之前算好的結果
   2. 內容 sha256 命中 curated 資料庫 → 這顆其實就是庫裡那一顆（即使檔名不同）
   3. 檔名命中 curated 且 sha256 一致 → 同上
   4. 檔名命中但 sha256 不同 → **視為新神經元**，輸出 id 改成 `<檔名>__<hash前8碼>`，並警告
   5. 都沒中 → 新上傳，走完整計算
2. **驗證**：座標數量級、是否落在 standard brain 範圍、節點數、檔名安全性
3. **descriptor**：質心 + 轉動慣量特徵值比 + 特徵向量
4. **三視圖**：`scale_um_per_px=5.0, normalize=p99`（從 config 取值）
5. **候選初篩**：質心距離 → inertia ratio 距離 → rod/disk 方向性，再取前 K
6. **打分**：逐對 pad → 下採樣到 50×50 → Siamese CNN
7. **排序輸出**：取前 n 筆寫成 CSV

---

## 6. 設定

全部集中在 [nrn_service/config.py](nrn_service/config.py)。常改的：

| 項目 | 預設 | 說明 |
|---|---|---|
| `RenderConfig.scale_um_per_px` | `5.0` | **不要動**。資料庫歸檔的三視圖全部是這個值 |
| `RenderConfig.render_version` | `v1_scale5.0_p99` | 改了畫圖參數就要改這個，快取才會失效 |
| `ModelConfig.weights` | `FineTune_Model/FineTune_miniLR_D1-D6_0.weights.h5` | 換模型要一併改 `model_id` |
| `MatchConfig.top_k_candidates` | `0` | 候選上限，`0` = 不截斷（預設）。設非 0 可換回延遲上限，但會改變輸出，不只是省時間（見 §10.8）|
| `MatchConfig.centroid_th / ratio_th` | `100.0 / 0.4` | 與離線 `candidate_matching.py` 對齊 |
| `MatchConfig.rod_angle_th_deg / disk_angle_th_deg` | `35.0 / 30.0` | 方向性過濾的夾角門檻，用 D1-D6 標註校準過（見 §10.7）|
| `ServiceConfig.default_top_n` | `5` | 輸出幾對 |

---

## 7. 使用者上傳的資料怎麼處理

```
user_data/
  uploads.sqlite                     registry（uploads / results 兩張表）
  <sha256 前16碼>/
    meta.json / neuron.swc / descriptor.npz / views.npz
    result_full.csv                  全部候選的排名（快取來源）
    result.csv                       回給使用者的前 n 筆
```

**未經確認的上傳檔不會出現在任何人的候選名單裡。** 服務只從 `data/` 取候選。

人工確認流程：

```bash
python3 tools/promote_upload.py --list              # 看有哪些上傳
python3 tools/promote_upload.py --show <upload_id>  # 檢視細節與結果
python3 tools/promote_upload.py --approve <upload_id>
python3 tools/promote_upload.py --reject  <upload_id> --note "理由"
```

`--approve` 只會複製 SWC 與三視圖到 `data/`。因為 descriptor 存成位置對齊的 `.npy`，
不能單筆 append，所以還要重建索引：

```bash
python3 swc_descriptor_batch.py --input ./data/SWC/FC --out ./data/descriptors_FC --source FC
python3 tools/pack_views.py --side FC
python3 tools/build_curated_index.py --side FC
# 然後重啟服務
```

---

## 8. 維護：什麼時候要重跑 `tools/`

`tools/` 底下的三支程式**不是給網頁端用的**，也不會被 `app.py` 或 `match_cli.py`
在執行時呼叫。它們是你事先跑好、產生衍生資料的工具，兩個入口都吃它們的產物。

| 產物 | 由誰產生 | 誰在用 | 什麼時候要重跑 |
|---|---|---|---|
| `data/view_store/` | `tools/pack_views.py` | **每次查詢**取圖 | 三視圖有新增 / 重畫 / 刪除 |
| `data/index/curated_index_*.parquet` | `tools/build_curated_index.py` | 上傳檔的內容比對 | SWC 有新增 / 修改 / 刪除 |
| （無產物，是管理動作）| `tools/promote_upload.py` | 你手動執行 | 人工確認上傳檔之後 |

```bash
python3 tools/pack_views.py --side FC          # 三視圖變動後
python3 tools/build_curated_index.py --side FC # SWC 變動後
# 然後重啟服務（KDTree 與 view store 是啟動時載入的）
```

> ⚠️ **服務或批次工作正在跑的時候，不要執行 `pack_views.py`。**
> `views_<side>.bin` 是被執行中的程序用 `mmap` 映射進記憶體的，
> 而 `pack_views.py` 會原地截斷並重寫同一個檔案。資料在腳下被抽換的結果是
> memmap 讀到垃圾，**而且不會拋任何錯誤**，只會安靜地算出錯誤的分數。
> 要重新打包就先停掉服務／等批次跑完。
> （改 `.py` 檔則無妨：Python 在啟動時就把模組載進記憶體，
> 執行中的程序不會讀到磁碟上的新版本。）

### 8.1 view store 存的是什麼（常見誤解）

**存的不是 50×50，也不是配對結果。** 存的就是原始三視圖 `(3, H, H)`，
`H` 每顆都不一樣（全庫 10～182，中位 34），像素和 `data/standard_views/*.npz`
**完全相同**，只是從幾萬個小檔案變成一個大檔案 + 偏移表。

`views_<side>.bin` 就是把所有圖的位元組首尾相接：

```
neuron id              offset(位元組)     H    佔用 3*H*H
104198-F-000000                    0    17           867
104198-F-000001                  867    17           867
104198-F-000002                1,734    32         3,072
```

`offset` 是「這顆的資料從檔案第幾個位元組開始」，**跟神經元的空間位置、中心點無關**。
取圖就是 `buf[offset : offset + 3*H*H].reshape(3, H, H)`，一次記憶體切片。

三層資料不要混淆：

| | 存什麼 | 是誰的屬性 | 存在哪 |
|---|---|---|---|
| npz / view store | `(3,H,H)` 原始三視圖 | **一顆神經元**的 | 永久 |
| 50×50 | 模型輸入 | **一對**的（取決於搭配對象，見 §10.6）| 不存，每次現算 0.38 ms |
| `result_full.csv` | 分數與排名 | 一次查詢的 | `user_data/`，同一份檔案重傳才命中 |

### 8.2 忘記重跑 `pack_views` 會怎樣

| 情況 | 行為 |
|---|---|
| 新增了三視圖但沒重打包 | **安全**。取不到的自動退回讀 npz，結果正確，只是慢 |
| 刪除了三視圖但沒重打包 | 安全。打包檔裡的孤兒不會被選為候選（候選來自 descriptor）|
| **既有的圖被重畫但沒重打包** | ⚠️ **危險**。會安靜地回傳打包當時的舊圖，沒有任何錯誤 |

服務啟動時會做一個便宜的檢查（數檔案數 + 比對目錄 mtime，約 7 ms），
過期時發出 `RuntimeWarning`：

```
FC view store 可能已過期（打包時 28612 顆，現在目錄裡有 28613 顆…）
```

但這個檢查**抓不到「檔案數不變、原地覆寫」**的情況。要完全確認請跑逐檔比對：

```bash
python3 tools/pack_views.py --check           # 兩側都驗，FC 約 2 分鐘
python3 tools/pack_views.py --check --side EM
```

輸出範例：

```
[verify] EM: 打包 12767 顆 / 磁碟 12767 顆  未打包 0  已刪除但仍在打包檔 0  內容不符 0
[verify] EM: 一致 ✓
```

### 8.3 這兩個索引都是可選的

拿掉之後服務仍然正常運作，只是退化。實測差異：

| | 沒有索引 | 有索引 |
|---|---|---|
| FC→EM 查詢 | 0.229 s | **0.123 s** |
| EM→FC（1911 對）| 1.519 s | **0.061 s** |
| 上傳檔的解析 | 認不出「同一顆換了檔名」| 用 sha256 正確認出 |

所以萬一索引檔壞掉或被誤刪，服務不會掛，重跑一次工具就好。

---

## 9. 實測數據

環境：本機 GPU（NVIDIA RTX PRO 5000）。

**啟動**：約 3 秒（模型 3.0 s、KDTree 0.09 s、view store 0.03 s）。

**查詢延遲**（60 次隨機查詢，資料庫內既有神經元）：

| 方向 | 候選數中位 | 總延遲中位 | p90 | 最大 |
|---|---|---|---|---|
| FC → EM | 584 | 167 ms | 369 ms | 457 ms |
| EM → FC | 1325 | 270 ms | 364 ms | 489 ms |

**上傳全新檔案**（含算 descriptor + 畫三視圖，各 8 次）：

| 側 | 節點數中位 | 總延遲中位 | 最大 |
|---|---|---|---|
| FC | 342 | 218 ms | 682 ms |
| EM | 4941 | 543 ms | 818 ms |

最壞情況是節點數極多的 EM 骨架（實測最大 181741 節點，光 render 就要 6.2 秒），
原因是 `standard_draw.project_and_rasterize` 對每條 edge 跑 Python 迴圈。
目前接受這個延遲；要優化的話請用「重畫 500 顆歸檔神經元 assert bit-equal」當正確性閘門。

---

## 10. 設計上踩過的坑

### 10.1 `scale_um_per_px` 的預設值

歸檔的三視圖全部是用 `5.0` 畫的（已用 bit-for-bit 重畫驗證：隨機 200 顆 FC 神經元
用 `scale=5.0, normalize=p99` 重畫，200/200 與歸檔完全相同）。
但 `standard_draw.py` 原本的函式預設值是 `1.0`，任何新的呼叫點忘記傳參數就會畫出
完全不同尺度的圖，**而且不會報錯**，只會讓分數失去意義。

現在預設值已改成 `5.0`，而且新產生的 npz 會寫入 `scale_um_per_px / normalize / render_version`。
服務端一律從 `nrn_service/config.py` 取值，不依賴函式預設值。

### 10.2 模型對空白輸入會給高分

實測 `FineTune_miniLR_D1-D6_0` 對全零輸入輸出 **0.678**。
如果某顆神經元 render 出空圖，它會直接排到結果前段。
所以 `validation.validate_views` 會擋掉 `views.max()==0`，候選端也會過濾。

### 10.3 keras 會對每個不同的 batch 大小重新 trace

因為每次查詢的候選數都不一樣，等於**每次查詢都在付 0.8～2.0 秒的編譯成本**
（實測 N=198 第一次 1.13 s、第二次 0.058 s；換成 N=197 又是 0.78 s）。

解法是把 batch 補齊到 `batch_size` 的倍數再送 `model.predict`，模型就永遠只看到
同一種形狀。補上去的是全零列，推論時 BatchNormalization 用 moving statistics，
每一列彼此獨立，不影響真實資料的分數。補齊後所有 N 都是 0.07～0.26 秒。

### 10.4 sqlite 每次重開連線要 150～200 ms

原本每個操作都 `sqlite3.connect` + `PRAGMA journal_mode=WAL`，一次查詢兩個操作就
吃掉 355 ms，是當時最大的固定成本。改成長駐單一連線 + `synchronous=NORMAL` 後降到 ~1 ms。

### 10.5 HTTP header 不能直接放中文

警告訊息是中文，直接塞進 `X-Warnings` 會讓整個回應拋 `UnicodeEncodeError`
（HTTP header 只允許 latin-1），前端會收到 500 而不是結果。
`app.py` 現在一律 `urllib.parse.quote()` 編碼後再放，前端用
`decodeURIComponent()` 還原。同一個原因，`Content-Disposition` 的檔名
也只能用 ASCII —— 這點剛好由 `validation.sanitize_neuron_id` 保證了
（只允許英數字、`.`、`_`、`-`）。

### 10.6 三視圖不能預先算好 50×50 快取

`_pad_to_same_size` 是**成對**做的（補到「這一對裡較大的那張」），
同一顆神經元搭配不同對象，補零後的大小不同。所以只能逐對前處理。
好在只要 0.38 ms/對，不是瓶頸。真正的瓶頸是讀檔，已由 view store 解決
（memmap 取圖 0.0024 ms/張，比讀 npz 快 143 倍）。

### 10.7 方向性過濾的角度門檻是 recall 取捨，不是精度工具

`rod_angle_th_deg` 原本設 30°，用 D1-D6 人工標註（`label >= 0.5` 視為真 pair，
624 對）量測後改成 **35°**。三件事值得記著：

**這道 gate 沒有鑑別力。** 角度作為判別器的 AUC 只有 **0.720**（正樣本中位 7.3°、
負樣本 13.3°）。30° 只擋掉 14% 的人工負樣本 —— 它不是在提升 precision，
它的唯一價值是砍候選數換延遲（實測候選池縮減約 42%）。既然下游 CNN 會把每個
候選都打分，這道門檻就該偏向 recall。

**30° 已經在切掉 ground truth。** 被 gate 管到的 361 對真 pair，角度分佈
中位 7.3° / p95 21.5° / p99 33.4°，30° 會切掉 9 對，其中 **6 對信心度 >= 0.8
（含 2 對 1.0）**，且已確認它們真的不在 `result/*_all_top5.csv` 裡。

**匯率是不對稱的。** 收緊很貴、放寬很便宜：

| θ | 標註 recall | 高信心(>=0.8)損失 | 候選池縮減 FC→EM / EM→FC |
|---|---|---|---|
| 20° | 96.31% | 14 對 | −48.3% / −49.3% |
| 25° | 98.08% | 7 對 | −45.3% / −46.2% |
| 30°（舊） | 98.56% | 6 對 | −42.2% / −42.7% |
| **35°（現在）** | **99.68%** | **0 對** | **−38.9% / −39.1%** |

35° 不是湊的：高信心正樣本的角度上界是 34.1°，35° 正好是「高信心零損失」的
最小門檻。`disk_angle_th_deg` 維持 30°，因為 disk 正樣本角度最大只有 26.7°
（高信心者 15.5°），30° 未殺到任何一對；且 disk pair 僅佔配對約 0.5%，
門檻高低對候選池的影響在 0.1pp 以內。

**為什麼要放寬而不是修邏輯。** v3 在 FC/EM 之間有系統性抖動：被切掉的 pair 中
有 4 對的兩側特徵值一致到 0.03~0.18、v1 一致到 7~17°，v3 卻差 30~44°，
也就是繞著共同的 v1 轉了約 30°。門檻必須蓋過這個抖動幅度。

試過但**沒有採用**的替代方案：只有兩側形狀夠接近才啟用 gate
（`|Δr21| <= 0.15`）。它被單純放寬角度完全支配 —— 候選池成本幾乎一樣
（−38.6% vs −38.9%），但 recall 更低（99.04% vs 99.68%）且仍留 3 對高信心損失。

剩下 2 對 35° 仍擋掉的（信心度 0.5 與 0.6）是真實的形狀差異，不是門檻問題：
`Cha-F-000329 × 579510789` 兩顆共平面（v1 差 11.7°）但面內伸長方向差 67°，
把軸對應排列到最佳後特徵值偏差仍達 0.48。

> ⚠️ 改了這個值就要重跑離線批次，`result/*_all_top5.csv` 才會反映新門檻。

### 10.8 候選上限不只是延遲護欄，它會改變輸出

`top_k_candidates` 原本設 `2000`，用意是保證單次查詢的延遲上限。問題是它**同時改變了
結果**：三段過濾後按 descriptor 距離排序取前 K，被截掉時第 K+1 名之後的候選會遞補
進來打分，所以**截斷版不是完整版的子集** —— 有些神經元的分數會因此變高，有些真配對
會被擠掉。

而且離線全庫掃描走的是同一條路徑（`match_cli.py --batch_side` → `match_one`），
所以歸檔的 `result/*_all_top5.csv` 也一起被這個護欄扭曲了。

實測 150 顆，與「不截斷」的完整答案比較：

| top_k | FC→EM top5 一致 | FC→EM top-1 | EM→FC top5 一致 | EM→FC top-1 |
|---|---|---|---|---|
| 2000（舊預設）| 85.3% | 96.0% | **73.3%** | 91.3% |
| 3000 | 97.3% | 98.7% | 88.7% | 97.3% |
| 4000 | 98.7% | 100.0% | 94.0% | 98.7% |

EM→FC 有超過四分之一的查詢在 `2000` 之下拿到的不是完整答案。

**現在預設改成 `0`（不截斷）。** 依據是全庫實測的單次查詢延遲（不含模型載入）：

| | 中位 | p90 | p99 | 最大 | 超過 1 秒 |
|---|---|---|---|---|---|
| EM→FC（12767 顆全跑）| 352 ms | 862 ms | 1373 ms | **2014 ms** | 805 顆 (6.3%) |
| FC→EM（22286 顆全跑）| 227 ms | — | 996 ms | **1620 ms** | 215 顆 (1.0%) |

超過 2 秒的整個 EM 資料庫只有 1 顆（`976351031`，7958 個候選）。網頁一次只查一顆，
最壞 2 秒可接受；相較之下超大 EM 骨架光 render 就要 6.2 秒（見 §9），打分階段不是
尾巴的主角。副作用是離線 CSV 與線上服務終於會給出一致的結果。

參數保留著：若之後併發成為瓶頸、或資料庫長大很多，設成非 0 就能換回延遲上限。
`match_cli.py` 也有 `--top_k` 可以單次覆蓋，不必動 config。

> ⚠️ 併發是另一回事，`top_k` 解決不了。`app.py` 的 `/match` 是 `async def` 卻同步
> 呼叫 `svc.query()`，會佔住整個 asyncio event loop —— 查詢期間連 `/health` 都回不了。
> 要處理併發得把它改成 `def`（FastAPI 會丟到 threadpool）並對打分加鎖。
> sqlite 那邊已經是安全的（`check_same_thread=False` + `RLock`），但模型目前沒有鎖。
> 尚未處理，留給接手前端整合的人。

---

## 11. 正確性驗證

| 驗證項目 | 結果 |
|---|---|
| `match_one` vs 離線三段過濾 | 40 顆神經元候選集合**完全相同** |
| 服務分數 vs `Model_predict.py` 寫法 | 最大差異 `2.78e-05`，top-20 排名 5/5 一致 |
| view store vs 原始 npz | 600 張**完全相同** |
| `render_single` 預設值 vs 歸檔 | 200/200 **bit-for-bit 相同** |

要重跑這些驗證，見本文件對應章節的說明；建議在改動 config、模型或前處理之後都跑一次。

---

## 12. 已知限制

### 12.1 右腦上傳會回空表（已知，暫不處理）

EM 資料庫只覆蓋單側腦（質心 x 範圍 −268～68），FC 兩側都有（−435～430）。
實測 500 顆隨機 FC 神經元，**117 顆（23%）在 EM 側找不到任何候選**；
把 x 取負（左右腦鏡像）之後只剩 19 顆兩個方向都找不到 —— 也就是說 84% 的
「查無結果」其實是左右腦不對稱造成的。

`NeuronMatchService.query()` 已經預留 `mirror: bool = False` 參數位並寫了完整註解，
目前傳 `True` 會丟 `NotImplementedError`（不會安靜地給錯誤結果）。
要啟用需要：算 descriptor 與畫圖之前先把 `swc.xyz[:, 0]` 取負，並決定鏡像結果要不要標註在輸出裡。

### 12.2 模型只用單一 fold

目前用 `FineTune_miniLR_D1-D6_0`，是 10-fold 交叉驗證的第 0 折。
對**新上傳的神經元**沒有問題；但對當初落在其他 9 折訓練集裡的標註 pair，
分數會偏樂觀。要更穩健可以改成 10 個 fold 取平均（成本從 0.1 s 變 1 s，
還能順便給出不確定度），但目前依討論先用單一模型。

### 12.3 FastAPI 層尚未實際執行

`app.py` 已完成但目前環境沒有安裝 `fastapi`，所以只做過語法檢查，
沒有實際啟動測試過。核心服務（`nrn_service/`、`match_cli.py`、`tools/`）
不需要任何新套件，已經完整測試。

```bash
pip install fastapi uvicorn python-multipart
```

### 12.4 跨網域（CORS）尚未設定

如果前端網頁不是掛在同一個網域，瀏覽器會擋掉對這個服務的請求。
要在 `app.py` 加上 `CORSMiddleware` 並列出允許的來源網址；
因為還不知道前端網頁最後會放在哪裡，目前沒有加。見 §4.5。

### 12.5 只能單一 worker

不要用 `uvicorn --workers N`：每個 worker 都會各自載入模型與整個資料庫索引，
記憶體會乘上 N。要擴充吞吐量請在前面放一層 queue。

---

## 13. 資料庫目前狀態

2026-09-06 補齊之後：

| | descriptor | SWC | views |
|---|---|---|---|
| FC | 28612 | 28612 | 28612 |
| EM | 12767 | 12767 | 12767 |

三個集合完全一致。補齊過程：FC 補了 6344 個 views，EM 補了 1470 個 descriptor、2 個 views，
失敗 0 筆；重算沒有動到既有數值（舊的 11297 個 EM descriptor 全部 bit-for-bit 相同）。
補齊前的 descriptor 備份在 `data/_backup_descriptors_20260906/`。

衍生的索引檔（SWC 有增刪就要重建）：

```
data/view_store/views_{FC,EM}.bin + *_meta.npz      tools/pack_views.py
data/index/curated_index_{FC,EM}.parquet           tools/build_curated_index.py
```
