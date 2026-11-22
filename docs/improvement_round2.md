# 第二輪改進：不堆 loss，改既有的設計

原則：**不新增 loss 項**（已經 11 項）。改的是「訓練怎麼出題」「網路看得到什麼」「用哪份權重」，每一項都是可單獨開關的旗標，可以直接放進消融表。

判斷「有沒有用」的尺度來自 nodds_b8 的兩個 seed（held-out、CR 0.1）：
同準確率下 ArcFace **整體差 ≤ 0.003、單一屬性差 ≤ 0.009**，r18 準確率差 1 到 2 個百分點，都算雜訊。
新的 arm 要贏過這個幅度才算數。

---

## 一、已經寫好、可以直接跑的三項

### 1. EMA 權重（不用訓練，今天就能測）

- **是什麼**：訓練時一直有存一份 EMA 權重（`save_models_ema/`，最後約 1000 步的平均），但評估一直讀的是 `save_models/` 的即時權重。
- **為什麼可能有用**：EMA 權重比較平滑，在生成模型和 flow 上常常比最後一步的權重好；不用多訓練一步。
- **改了什麼**：評估加 `--ema_weights`，改讀 `save_models_ema/`，其他都一樣。曲線的 `.json` 會記下有沒有用 EMA。
- **算不算亮點**：不算（大家都在用），但如果有提升，就是免費的分數，直接併進最終版。

### 2. 訓練時就用 CR（`trcr`）——最推薦

**問題（程式碼裡看得到的）：**

- 訓練時，每個編輯的方向是由**模型自己的 conditioner** 決定（`train_edit_direction=cond`）。所以 flow 從來沒看過「要求的方向」跟「自己的讀值」衝突的情況。
- 但評估時，方向是使用者給的。conditioner 一讀錯，flow 就碰到從沒訓練過的情況，所以才需要推論時的 CR 補救。
- 更糟的是：訓練中 target loss 的方向是看 **r34 老師**的讀值，flow 的方向是看 **conditioner** 的讀值。兩者不一致的那些臉上，flow 被要求往 A 走，loss 卻把輸出拉往 B（log 裡的 `dir_disagree_frac` 就是這個比例）。這是互相矛盾的梯度。

**改法（`--train_consistent_reading 0.1` 搭配 `--train_edit_direction teacher`）：**

1. 方向改由 r34 老師決定，就像評估時由使用者決定。
2. conditioner 讀錯邊時，用**跟推論完全一樣的規則**把讀值移到正確那邊（0.4 / 0.6），再編碼與設目標。
3. target、CLIP、DDS 的方向都跟著同一個方向，矛盾的梯度消失。

**預期會看到：**

- 不加 CR 的曲線會追上加 CR 的曲線（CR 的增益變小，因為模型已經學會了）。
- 加 CR 的曲線不會變差，最好是 Young、Male 這種常讀錯的屬性變好。
- log 的 `train_cr_frac`：規則實際改到的編輯比例。

**為什麼是好亮點：**

- 把「推論時的修正技巧」變成「訓練和推論一致的設計」，正好回應教授「CR 是不是推論技巧／作弊」的疑慮：模型本身就學會了，評估時可以不加 CR。
- 不加任何 loss。
- 要附一個對照組 `teachdir`（只改方向來源、不加 CR 規則），才能分清楚是哪一半有用。

### 2b. 只在「證實讀錯」時才修正（`vercr`）——取代 trcr 當主要版本

trcr 的問題：只要 conditioner 和 r34 讀法不同，就一律以 r34 為準。但 r34 也會讀錯，這樣等於把 conditioner 的判斷整個換掉。

`vercr`（`--train_edit_direction verified --train_consistent_reading 0.1`）：**方向照舊由 conditioner 決定**，只有同時滿足下面四個條件，才判定 conditioner 讀錯並修正：

1. r34 老師看原圖，讀得很清楚（大於 0.65 或小於 0.35）。
2. ConvNeXt 老師也讀得很清楚，而且跟 r34 同一邊。
3. 事先算好的 r34 讀值也同一邊。
4. conditioner 在另一邊（或離 0.5 不到 0.1）。

修正的內容跟推論時的 CR 一樣（方向改成老師的、讀值移到 0.4 / 0.6）。其他所有臉**完全不動**。老師讀得不清楚、或兩個老師意見不同時，代表沒有證據說 conditioner 錯，就不修正。

| conditioner | r34 / ConvNeXt / 預先 r34 | 結果 |
|---|---|---|
| 0.70 | 0.10 / 0.20 / 0.15 | 證實讀錯 → 方向「加」，讀值 0.70 → 0.40 |
| 0.20 | 0.10 / 0.20 / 0.15 | 一致 → 不動 |
| 0.70 | 0.10 / **0.55** / 0.15 | ConvNeXt 不確定 → 不動 |
| 0.70 | 0.10 / 0.20 / **0.60** | 預先 r34 不同意 → 不動 |
| 0.70 | 0.40 / 0.45 / 0.42 | 老師都不確定 → 不動 |

（這張表是用訓練程式裡的同一段規則實際算出來的。）

另外，所有臉的 target、CLIP、DDS loss 都跟著「這次編輯的方向」走，所以沒被修正的臉也不會再有矛盾的梯度。這一半就是既有的 `dircond` arm，所以 **`dircond` 是 vercr 的對照組**：

- dircond：只消除矛盾，不修正任何讀值。
- vercr：消除矛盾，**再加上**修正證實讀錯的臉。
- 兩者的差距 = 「修正讀錯」這件事本身的貢獻。

log：`train_verified_frac`（老師清楚且一致的比例）、`train_cr_frac`（實際被修正的比例）。

### 3. 速度網路知道自己在哪一層（`layeremb`）

**問題（已用程式碼確認）：**速度網路對 W+ 的 18 層用**同一組權重、同一個條件**。它分不出第 3 層（姿勢、臉型）和第 12 層（顏色），只能從那一層本身的數值去猜。但 StyleGAN 不同層管的東西完全不同，眼鏡、笑容、年齡用到的層也不同。

**改法（`--velocity_layer_embed`）：**每一個隱藏層加一個「每一層各自一組」的可學偏置（18 × 隱藏維度），初始化為 0。

- 從舊 checkpoint 接續時，模型的輸出**完全一樣**（已驗證差異為 0），所以 stage-2 消融是公平的：起點相同，只差在能不能學到層的資訊。
- 參數只多一點點（每個網路約 18 × 512 × 隱藏層數），審稿人不能說「只是模型變大」。
- 評估和推論編輯器會自動從 checkpoint 偵測，不用額外加參數。

**預期會看到：**屬性保持（preserve）和區域保持（region）變好，尤其是 Male（目前區域保持只有 73 到 75%）。

**新不新：**StyleFlow、SDFlow 這類 CNF 編輯器的速度網路都是「不分層」的；StyleSpace 等研究已經證明各層語意不同。「把層的資訊給速度網路」是改良別人的方法，新穎度中等。

---

## 二、下一批（要再寫程式，先等上面三項的結果）

| 編號 | 想法 | 改的是哪個既有東西 | 為什麼可能有用 | 會不會加 loss |
|---|---|---|---|---|
| N1 | **ID hinge 門檻自動校準** | ID loss 的門檻（現在手調 0.8，Young 0.68） | 改成每個屬性「最差的那一段」才罰（例如 ID 低於該屬性移動中位數 25% 分位的樣本），不用手調，也不會讓容易的屬性被過度保護 | 不會，改既有 |
| N2 | **多屬性同時編輯評估** | 不改訓練，只加評估 | 條件式 flow 一次編碼、同時改兩個目標再解碼，這是 InterFaceGAN 類方法要逐一疊加才做得到的。可以變成一個「能力」上的亮點（例如「戴眼鏡 + 變老」的成功率與 ID） | 不會 |
| N3 | **用 judge 分歧自動找捷徑** | coupled targets（`nomakeup` 是手動版） | nomakeup 手動指定口紅、濃妝、耳環，r18 反而掉 15 個百分點，代表手動指定不對。改成自動：統計每個屬性編輯時最常連帶改變、而且讓兩個 judge 意見分歧的屬性 | 不會，改既有 |
| N4 | **拿掉 Stable Diffusion** | DDS | 兩個 nodds seed 已經很穩定。若同 batch 下和有 SD 的差距在雜訊內，就整個拿掉：訓練更快、少一個授權問題、少一項 loss（這本身就是「簡化不掉分」的貢獻） | 減少 |
| N5 | **速度報告** | 不改，只量 | ODE 的 NFE 與每張臉編輯時間，審稿人一定會問 | 不會 |

---

## 三、怎麼跑

### 1. EMA（在有最終模型的那台，cchen）

```bash
cd ~/d2f && git pull
```

```bash
O="/home/cchen/桌面/SDFlow-main (副本)/output/SDFlow"; RUN="$O/multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k_long"; ls "$RUN/save_models_ema" | grep prior | tail -n 3
```

看得到 `prior-0130000` 才往下：

```bash
nohup python -u -m scripts.editing_curves --checkpoint_dir "$RUN" --step 130000 --ema_weights --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 --method model --consistent_reading 0.1 --scales 0.2 0.4 0.6 0.8 0.95 1.0 1.25 1.5 --sample_offset 500 --num_samples 500 --out_csv output/curves/ours_ema_cr0.1_off500.csv > output/curves/ours_ema_cr0.1_off500.log 2>&1 &
```

跟原本的權重比（同一批臉）：

```bash
python -m scripts.method_table --target 90 --run "live=output/curves/ours_cr0.1_off500.csv" --run "EMA=output/curves/ours_ema_cr0.1_off500.csv"
```

### 2. 三個訓練 arm（cchen，跟 nodds_b8 同樣的設定才能比）

完整設定在 batch 8 跑不下，所以跟之前一樣疊在 nodds 上、batch 8。對照組就是已經跑完的 nodds_b8 seed 1 和 seed 2：

```bash
cd ~/d2f && git pull && O="/home/cchen/桌面/SDFlow-main (副本)/output/SDFlow"; P=multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k
```

```bash
EXTRA_SETS="batch=8 use_diffusion_guidance=false" TAG=nodds_b8 ARMS="vercr dircond layeremb" bash scripts/launch_stage2_ablation.sh "$O/$P" "0"
```

三個會依序跑，每個和 nodds_b8 一樣久。vercr 最重要，先跑。trcr 和 teachdir 留著當備案（`ARMS="trcr teachdir"`），不用先跑。

**每個 arm 都要有「不加 CR」的曲線**（launcher 只會跑加 CR 0.1 的版本）。nodds_b8 的兩個 seed 也要補：

```bash
for r in nodds_b8 nodds_s2_b8 vercr_nodds_b8 dircond_nodds_b8 layeremb_nodds_b8; do d="$O/${P}_abl_$r"; [ -d "$d/save_models" ] || continue; f="$d/curves_model_s100000_off500_crnone.csv"; [ -f "$f" ] || python -u -m scripts.editing_curves --checkpoint_dir "$d" --step 100000 --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 --method model --scales 0.2 0.4 0.6 0.8 1.0 1.25 1.5 --sample_offset 500 --num_samples 500 --out_csv "$f" > "${f%.csv}.log" 2>&1; done
```

### 3. 判斷

把五個 run 的 CSV（加 CR、不加 CR 各一份）貼給我。要看：

| arm | 跟誰比 | 算成功的條件 |
|---|---|---|
| vercr | nodds_b8 s1、s2，以及 dircond | 不加 CR 時：明顯高於兩個 seed（大於雜訊）；加 CR 時不變差；而且要贏 dircond，才能說「修正讀錯」有貢獻 |
| dircond | nodds_b8 s1、s2 | 只消除矛盾梯度有沒有用 |
| layeremb | nodds_b8 s1、s2 | 同準確率下 ArcFace、preserve、region 至少一項大於雜訊，其他不變差 |

通過的才進下一步：完整訓練 × 3 個 seed（論文的消融表）。
