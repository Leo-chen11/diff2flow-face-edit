# 專案改善路線圖：毛病、解方，以及「三個亮點」怎麼達成

這份文件根據程式碼和已經跑出來的數字寫成（README 和 `docs/` 裡的舊文件已經過期，不當依據）。
最終模型指的是 `multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k_long` 的 step 130000。

文件裡每一項都會標明：**已量到**（有數字）、**已看到**（有圖或程式碼為證）、**還沒量**（只是推測）。
還沒量的東西，一律不能寫進論文。

---

## 0. 一頁結論

**目前站得住的：**

- 在開發用的 500 張臉上，同樣的編輯準確率下，你的模型的 ArcFace 比重訓的原版 SDFlow 高 0.056 到 0.110（已量到，見 1.2）。
- 「方向一致的來源讀值」（consistent reading）不用重訓，強度 1 時準確率從 95.0% 升到 98.4%（已量到）。

**目前站不住的：**

- **「三個以上、真的有提升分數的亮點」這個需求現在還沒達到。** 有數字支持的亮點只有 1 個（consistent reading）。整體模型贏 SDFlow，但還不知道是哪個元件贏的，因為乾淨的消融一個都還沒跑完。
- 論文主表要用的 held-out 數字還沒出來（曲線和對照表正在跑）。
- 整個結果只有 1 個 seed。
- 「ID 0.90」在高準確率下達不到：ArcFace 0.90 只出現在準確率 67% 到 69% 左右（已量到）。

**要做的事，依序：**

1. 跑完 held-out 曲線和對照表（不用訓練）。
2. 做兩個不用訓練的亮點測試（第 3 節的 H1、H7）。
3. 在 cchen 那台跑 stage-2 消融：先量 seed 雜訊，再跑 `noprior`、`nomakeup`、`oneteach`、`ot`。
4. 通過門檻的元件，做完整訓練（full chain）乘以 3 個 seed 的最終消融。
5. 依結果寫論文：有數字的才當亮點，沒通過的拿掉或放進附錄。

---

## 1. 現在的狀態

### 1.1 最終模型實際上是什麼

論文的方法描述必須跟這張表一致（來源：`configs/trained/…_long.json` 和訓練程式）。

| 項目 | 實際設定 |
|---|---|
| 基底 | StyleGAN2 FFHQ 的 W+（18×512）、e4e 反演、條件式 CNF（torchdiffeq dopri5） |
| 屬性 | 5 個：Eyeglasses 15、Male 20、Young 39、Smiling 31、Bangs 5 |
| 速度場 | `dof`：兩個速度網路相加（屬性速度 + 基底速度） |
| 正交懲罰 | `lag_reg_weight = 0`，**關閉** |
| 方向庫（direction bank） | 只在訓練時當老師。50k 到 70k 之間把編輯從方向庫交給 flow，70k 之後 mix = 0，**推論時的編輯完全來自 flow** |
| 方向庫 prior | cosine hinge，權重 `0.1 × (1 − mix)`，50k 到 130k 都有作用，70k 之後是完整的 0.1 |
| 老師 | r34 + ConvNeXt，兩個取 min |
| ID loss | hinge，門檻 0.8（Young 是 0.68），權重 0.35 |
| 其他 loss | DDS（Stable Diffusion 引導）、CLIP 方向性提示、local region（含 head）、counter/preserve attr |
| 訓練 | 兩段：stage 1 到 50k，stage 2 再 80k；batch 4；seed 1 |

**目前啟用的 loss 一共 11 項**：kd、nll、reg、id、counter_attr、preserve_attr、bank_prior、DDS、age DDS、CLIP prompt、local region。

### 1.2 已經量到的數字（開發用的 500 張臉，不能放論文主表）

你的模型（加 consistent reading 0.1）對重訓的原版 SDFlow（5 屬性，step 17549），在同樣平均編輯準確率下的 ArcFace：

| 編輯準確率 | 你的模型 | SDFlow | 差距 |
|---|---|---|---|
| 70% | 0.897 | 0.841 | +0.056 |
| 80% | 0.876 | 0.821 | +0.055 |
| 90% | 0.849 | 0.774 | +0.075 |
| 94% | 0.833 | 0.754 | +0.079 |
| 97% | 0.814 | 0.704 | +0.110 |

其他已量到的：

- consistent reading：強度 1 時準確率 95.0% → 98.4%；「怎麼編都編不動」的臉從 3.3% 降到 0.3%；margin 取 0.05、0.1、0.2 結果一樣。
- 重訓的 SDFlow 可信：3 屬性版和官方釋出版在 50/70/85/95% 準確率下的 ArcFace 是 0.857/0.802/0.741/0.676 對 0.853/0.793/0.726/0.655。
- Male 和 Bangs：R50 judge 和 CelebA r18 judge 的準確率差大約 9 個百分點。
- Male 的區域保持（region preserve）只有 73% 到 75%。

---

## 2. 毛病清單與解方

每一項的格式：**問題** → 為什麼是問題 → 解方。優先度：🔴 不修不能投、🟠 很可能被審稿人問、🟡 加分項。

### A. 論文主張和程式碼對不上

**A1. 🔴「正交分解」不能寫。**
- 問題：最終模型的正交懲罰權重是 0，兩個速度網路只是相加。舊文件 `docs/lag_dof_sdflow_method.md` 還寫著 gate 和正交。
- 為什麼是問題：審稿人看到「正交」會要求看正交程度的數字，而你沒有訓練它。
- 解方（二選一）：
  1. 論文只寫「雙速度場（屬性速度 + 基底速度）」，不提正交。
  2. 跑 `ARMS=orth`（`lag_reg_weight=20`）。有提升才把正交寫進方法。

**A2. 🔴 方法描述要改成「方向庫只是訓練時的老師」。**
- 問題：`docs/related_work_2021_2026.md` 把本方法寫成「方向庫加 residual」，那是舊設計（`replace` 模式）。最終模型在推論時不用方向庫。
- 解方：方法章節寫成「線性方向老師 → 逐步交棒給 flow → 交棒後用 prior 保持方向」。這也比較好當亮點（見 H2）。

**A3. 🟠 最終模型的程式碼版本不在 repo 裡。**
- 問題：config 記錄的 `code_version` 是 `105f1c4`，這個 commit 不在目前的 branch。
- 為什麼是問題：沒辦法證明「這份程式碼訓練得出這個模型」。
- 解方：最終版（v2）一律從目前的 branch 重新訓練，config 會自動記下 commit。舊的 130k 模型只當開發過程的版本。

**A4. 🟠 NLL 的數值不能報。**
- 問題：log 裡的 NLL 有一個固定的 +470.6 偏移（`standard_normal_logprob` 的常數項用了 512 維）。
- 解方：論文不報 NLL 或 bits/dim。只在訓練曲線圖裡用，而且註明是相對值。

**A5. 🟡 README 和 `docs/` 過期。**
- 解方：投稿前重寫 README：環境、資料、訓練兩段的指令、評估三步、每個 baseline 的來源。這份文件可以當底稿。

### B. 模型本身的弱點

**B1. 🔴 Male 編輯走了「化妝捷徑」。**（已看到：render 圖；已量到：R50 和 r18 差 9 個百分點）
- 問題：男變女主要是靠加紅唇、變白、加耳環達成的。R50 judge 會被口紅騙過（例如 face 28、72），r18 比較不會。
- 為什麼是問題：這代表準確率有一部分是「騙過 judge」而不是真的改了性別。審稿人只要看圖就會發現。
- 另一個相關的洞：`PRESERVE40_ALLOW` 允許 Male 編輯時改變 Heavy_Makeup（18）、Earrings（34）、Lipstick（36），所以 Pres 指標看不到這個捷徑。
- 解方：
  1. 訓練：`ARMS=nomakeup`（男變女時，r34 老師讀到的口紅、濃妝、耳環要維持在 0.3 以下）。
  2. 評估：另外報一個「嚴格版 Pres」，Male 的允許清單拿掉 18、34、36。
  3. 先量基準（不用訓練）：`edit_direction_audit` 的 `male_makeup.csv`。

**B2. 🟠 女變男保留長髮。**（已看到）
- 這算不算錯有爭議：保留髮型也可以說是保留身分。
- 解方：放進 limitations，附一張圖。不建議為了這個再加一項 loss。

**B3. 🔴 ID 0.90 的目標要重新定義。**（已量到）
- 問題：ArcFace 0.90 只在準確率 67% 到 69% 出現；準確率 95% 時大約是 0.83。
- 解方：論文不要寫「ID 達到 0.90」這種單點數字。改報「同準確率下的 ID」（70/85/95% 三個點）和整條曲線。這也是比較公平的報法，本來就對你有利。

**B4. 🟠 訓練時留著四個已知的問題沒開修正。**（已看到：訓練啟動時印的 KNOWN ISSUES LEFT ON）

| 問題 | 後果 | 修正的 arm |
|---|---|---|
| Bangs 沒有自己的 CLIP 提示詞 | 移除瀏海時，通用提示詞把圖推向 "a distorted face photo" | `bangsclip` |
| target 方向不是照 flow 拿到的方向 | 訓練目標和實際編輯方向可能不一致 | `dircond` |
| 加眼鏡的遮罩是模糊而不是膨脹 | 遮罩邊界不準 | `truedilate` |
| `id_cond_dropout` 沒有重新縮放 | 訓練和推論的 ID 條件大小不一致 | 還沒有 arm |

- 解方：`ARMS=fixes`（前三個一起開）。有提升就併入 v2，沒差別也併入（因為是修正錯誤）。

**B5. 🟠 loss 太多（11 項）。**
- 為什麼是問題：審稿人會問每一項的必要性；每一項都要一列消融。
- 解方：`nodds`、`noclip`、`nohead` 各跑一次。拿掉後沒有變差的就直接刪掉，最後再用 `custom` 確認「一起刪」也沒事。目標是剩 6 到 7 項。
- 附帶好處：DDS 用的是社群模型 `Realistic_Vision_V5.1`，授權和可重現性都容易被質疑。`nodds` 沒變差的話就可以整個拿掉。

**B6. 🟡 單一屬性的弱點。**
- R50 judge 看不到細框眼鏡（face 7）。解方：Eyeglasses 另外報 parser judge 的準確率（程式已有 `--glasses_judge parser`）。
- face 51（深膚色）沒有被變老。這只是一張圖，還不能下結論。解方見 C6。

### C. 評估的弱點

**C1. 🔴 論文主表的數字還沒有。**
- 現在所有數字都在開發用的 500 張臉上。
- 解方：`scripts/run_heldout_curves.sh` → `comparison_table`（`--sample_offset 500`）。最終版建議把 `--num_samples` 提高到 1000 以上，FID/KID 才比較穩。

**C2. 🔴 只有 1 個 seed。**
- 為什麼是問題：不知道 seed 之間的雜訊有多大，就沒辦法說任何消融「有差」。
- 解方：先跑 `ARMS=ctrl SEED=2` 和 `SEED=3`。這兩次和原本 seed 1 的差距就是雜訊的大小，所有消融都要跟它比。

**C3. 🟠 judge 對你的模型有利。**
- 問題：你的模型是對著 r34 和 ConvNeXt 老師訓練的，這兩個老師和 R50 judge 是同一類分類器、同一類標籤。baseline 沒有這個優勢。
- 解方：主表同時報 R50、r18、兩者共識（consensus）三種準確率，並且用 r18 當匹配準確率的指標再做一次（`--match_metric celeb_success`）。兩種匹配方式結論一致才算數。

**C4. 🟠 baseline 各自的訓練訊號也不同。**
- AdaTrans 訓練時用的 LPIPS 和評估用的是同一個網路，所以它在 LPIPS 上占便宜。
- 解方：在表格下方用註腳寫明每個方法訓練時用了什麼（哪個分類器、哪個 ID 網路、有沒有 LPIPS）。

**C5. 🟠 所有指標都是對「e4e 重建圖」算的，不是對真實照片。**
- 這樣做是對的（排除反演誤差），但審稿人會想知道對原照的結果。
- 解方：附錄加一張表，報編輯結果對真實照片的 ArcFace 和 LPIPS，並寫出 e4e 重建本身的分數當上限。

**C6. 🟡 沒有分組分析和人工評估。**
- 解方：
  1. 依來源屬性分組（男/女、年輕/年長）報成功率，最少要有這個。
  2. 小規模人工評估：每個方法每個屬性 20 張，問「屬性改了嗎」「還是同一個人嗎」。這同時回應 C3。

**C7. 🟡 沒有報速度。**
- CNF 用自適應 ODE 解，比 AdaTrans 和 GradCtrl 慢。審稿人會問。
- 解方：報每張臉的編輯時間和 NFE（訓練 log 已有 `nfe_encode`、`nfe_decode`）。

### D. 訓練流程和可重現性

**D1. 🔴 stage-1 的 checkpoint 只在 cchen 那台。** 壞了就全部重來。解方：現在就備份一份（整個 `multi_v1_min_2teach_seed1` 資料夾）。

**D2. 🟠 檔案散在三個地方**（cchen、WSL 的 `~/d2f`、Windows 桌面資料夾），已經造成好幾次路徑錯誤。解方：決定一台當「評估機」，所有權重和 CSV 都收到那台的 `~/d2f/output/` 底下；其他地方的用 `scripts/pack_eval_bundle.sh` 打包搬過去。

**D3. 🟠 stage-2 消融只拿掉後 80k 步的東西。** 前 50k 步還是有那一項。
- 所以 stage-2 消融只能當「篩選」：便宜、快，用來決定哪些值得做完整訓練。
- 論文裡的消融表必須用完整訓練（`scripts/launch_chain_fix.sh` 加 `S1_SETS`/`S2_SETS`）。

### E. 比較對象

| 年份 | 方法 | 狀態 |
|---|---|---|
| 2020 | InterFaceGAN | 已有 |
| 2021 | Latent Transformer、StyleFlow | 已訓練，等 held-out 曲線 |
| 2022 | GradCtrl | 已移植並訓練，等驗證和曲線 |
| 2022 | PluGeN | 已實作（`training/train_plugen.py`），等訓練和曲線 |
| 2023 | AdaTrans | 已重寫並訓練，等曲線 |
| 2023 | DeltaEdit | 已實作，用官方釋出的權重（不用訓練），等曲線 |
| 2024 | SDFlow | 已重訓並驗證 |
| 2024 | PluGeN4Faces | **不比**（見下） |

- **PluGeN4Faces 不能比**（已讀官方 repo）：沒有釋出權重；訓練需要影片資料集（Celeb-DF 的同一人多個畫面）和一個沒有公開的 9 屬性偵測器；屬性裡也沒有 Bangs。論文在相關研究裡引用，並寫明這個原因。
- **DyStyle 不能比**（已讀官方 repo）：repo 裡只有 README 和圖片，沒有程式碼。
- 🟠 AdaTrans 還沒有跟官方的編輯器權重對過（SDFlow 和 GradCtrl 都有這一步）。官方 README 給的下載連結列出的內容是分類器、RealNVP、e4e、StyleGAN 和 latent，沒有明列編輯器權重。拿得到就在同一批臉上各跑一次；拿不到就在論文寫明「依官方超參數重新實作並訓練，官方編輯器權重未公開」，並列出和官方不同的地方。
- 🟡 2025、2026 沒有找到設定相同又有程式碼的方法。論文用一句話交代範圍即可。

---

## 3. 三個以上的亮點：現況和達成計畫

### 3.1 什麼才算「亮點」

你的條件有三個，缺一不可：

1. **真的提升分數**：拿掉它分數會掉，而且掉的幅度大於 seed 雜訊。
2. **好消融**：可以單獨開關。
3. **夠新**：別人沒這樣用過，或是把別人的方法改良後用在這裡。

下面「新不新」那一欄是我根據目前查過的文獻做的判斷，不是保證。投稿前每一個都要再查一次文獻。

### 3.2 候選清單和現況

| 編號 | 候選亮點 | 分數有提升嗎 | 要不要訓練 | 新不新 |
|---|---|---|---|---|
| H1 | 方向一致的來源讀值（consistent reading） | **已量到**（開發臉 +3.4 個百分點） | 不用 | 我沒查到同樣的做法 |
| H2 | 線性老師逐步交棒 + 方向 prior | 還沒量 | 要 | 交棒的排程我沒查到同樣的做法；「flow 貼近線性方向」的概念需要查 |
| H3 | 雙速度場 `dof` | 還沒量（你印象中稍微好一點） | 要（完整訓練） | 改良 SDFlow 的 CNF |
| H4 | 捷徑抑制（coupled targets，`nomakeup`） | 還沒量（問題已看到） | 要 | 改良既有的屬性保持 loss |
| H5 | 雙老師取 min + ID hinge | 還沒量 | 要 | 各自都有前例，組合是你的 |
| H6 | 運輸正則（RNODE，`ot`） | 還沒量 | 要 | 把 RNODE 用到條件式編輯 flow，屬於「改良別人的方法」 |
| H7 | 邊界投影（`--preserve_boundaries`） | 還沒量 | 不用 | 概念接近 GradCtrl 的遮罩和 InterFaceGAN 的條件投影，新穎度低 |

**結論：現在是 1 個，離 3 個還差 2 個。** 下面是補齊的計畫。

### 3.3 每個候選要怎麼驗證

#### H1 方向一致的來源讀值（最穩，先把它做完整）

- 它解決的問題：模型自己的 conditioner 有時把來源讀錯邊（例如臉明明沒戴眼鏡，讀值卻大於 0.5）。這時「加眼鏡」的指令會變成幾乎不動。修正方法是把讀值移到正確的那一邊再編輯。
- 還缺的證據：
  1. held-out 臉上有沒有同樣的提升。
  2. 在「同準確率」下 ArcFace 有沒有跟著變好（現在只知道同強度下準確率變高）。
- 怎麼做（不用訓練）：跑一條不加 consistent reading 的曲線，跟已經有的 `ours_cr0.1_off500.csv` 比：

```bash
python -m scripts.editing_curves --checkpoint_dir $RUN --step 130000 --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 --method model --scales 0.2 0.4 0.6 0.8 1.0 1.25 1.5 --sample_offset 500 --num_samples 500 --out_csv output/curves/ours_nocr_off500.csv
```

- 可以讓它更強的做法：把同樣的修正套到原版 SDFlow 和 StyleFlow 上。如果它們也變好，H1 就不只是「修自己的 bug」，而是「所有條件式 flow 編輯器都適用的外掛」。這需要我在 `editing_curves.py` 裡替這兩個方法加上同樣的選項（還沒寫）。
- 風險：審稿人可能認為它只是推論技巧。所以它適合當第三個亮點，不適合當第一個。

#### H2 線性老師逐步交棒 + 方向 prior

- 要回答兩個問題：(a) prior 有沒有用；(b) 整個「老師 → 交棒」有沒有用。
- 篩選（stage 2，便宜）：`ARMS=noprior`。只能回答 (a)。
- 定案（完整訓練）：
  1. 完全不用方向庫，flow 從頭自己學。
  2. 用方向庫交棒，但沒有 prior。
  3. 完整版。
- 這三列就是論文消融表裡 H2 的部分。第 1 列需要確認訓練程式在沒有 `--direction_bank_path` 時跑得起來（mechanism report 顯示有這條路，但我沒跑過）。

#### H3 雙速度場

- 只能用完整訓練比：`velocity_field=original` 對 `dof`，其他全部一樣。指令：

```bash
TAG=orig S1_SETS="velocity_field=original" S2_SETS="velocity_field=original" bash scripts/launch_chain_fix.sh <stage1> <stage2> <gpu>
```

- 要注意公平性：`dof` 有兩個速度網路，參數大約是兩倍。審稿人會問「是不是只是模型變大」。解方：加一列「original 但把隱藏層加寬到參數量相同」。
- 你印象中「稍微好一點」。如果差距落在 seed 雜訊內，就不要當亮點，改成一句「我們採用雙速度場」帶過。

#### H4 捷徑抑制

- 篩選：`ARMS=nomakeup`。
- 要看的指標不是 R50 準確率（它可能會下降，因為 R50 本來就被口紅騙），而是：
  1. Male 的 r18 準確率和共識準確率。
  2. R50 和 r18 的差距有沒有從 9 個百分點縮小。
  3. 嚴格版 Pres（不允許化妝屬性改變）。
  4. render 圖上口紅和耳環有沒有消失。
- 這個亮點的說法是「用獨立 judge 之間的分歧找出捷徑，再用成對的保持目標把它關掉」。診斷加上修正一起講，比單講 loss 更有說服力。

#### H5、H6、H7（備案）

- H5：`ARMS="oneteach idcont"`。
- H6：`ARMS=ot`（0.01/0.01）和 `ot_hi`（0.1/0.1）。除了分數，還要看 `nfe_encode`/`nfe_decode` 有沒有下降。NFE 下降代表編輯變快，這本身就是一個可以報的好處（回應 C7）。
- H7（不用訓練）：

```bash
python -m scripts.editing_curves --checkpoint_dir $RUN --step 130000 --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 --method model --consistent_reading 0.1 --preserve_boundaries ./data/attr_boundaries_wplus.pth --scales 0.2 0.4 0.6 0.8 1.0 1.25 1.5 --sample_offset 500 --num_samples 500 --out_csv output/curves/ours_cr0.1_presb_off500.csv
```

### 3.4 「真的有提升」的判定門檻

一個元件要同時滿足這三條才算通過：

1. **幅度**：在 held-out 臉上、同準確率 85% 和 95% 兩個點，ArcFace 至少差 0.01；或是同 ArcFace 下準確率至少差 2 個百分點。（H4 改看 r18/共識準確率和嚴格版 Pres。）
2. **大於雜訊**：差距大於 `ctrl` 三個 seed 之間的最大差距。
3. **方向一致**：3 個 seed 裡至少 2 個同方向，R50 匹配和 r18 匹配的結論也一致。

### 3.5 湊不到三個怎麼辦

我對通過機率的主觀排序（不是量出來的）：H1 > H4 > H2 > H6 > H3 > H5 > H7。

- **最可能的組合**：H1 + H4 + H2。
- **H2 沒過**：用 H6 或 H3 補。
- **只過兩個**：還有三個沒實作的方向可以試，都需要寫新程式，我可以做：
  1. **訓練時就用方向一致的讀值**：現在 H1 只在推論時修正。把同樣的規則放進訓練，讓訓練和推論一致。這是 H1 的延伸，可以合併成一個更完整的亮點。
  2. **用 judge 分歧自動找捷徑**：把 H4 從「手動發現化妝」推廣成「自動找出每個屬性最常連帶改變的屬性，再自動加成對目標」。這樣 H4 就從修一個屬性變成一個通用方法。
  3. **每個屬性各自的 ID 門檻自動校準**：現在 ID hinge 的門檻是手調的（0.8，Young 0.68）。改成依屬性在訓練中自動調整。
- **不建議的補法**：再加新的 loss。現在已經 11 項，再加只會讓 B5 更嚴重。

---

## 4. 執行順序

### 第一階段：不用訓練（WSL，現在就能做）

1. 跑完 held-out 曲線：`bash scripts/run_heldout_curves.sh`。
2. GradCtrl：驗證官方權重（check 兩個 log），再跑它的 held-out 曲線。
3. H1 的對照曲線（不加 consistent reading）和 H7。
4. 對照表：`comparison_table`，`--target_acc 70 85 95`，R50 匹配和 r18 匹配各一次。
5. Male 化妝基準：`edit_direction_audit` 的 `male_makeup.csv`。

這一階段結束就會知道：論文主表長怎樣、H1 在 held-out 上成不成立、H7 有沒有用。

### 第二階段：stage-2 篩選（cchen）

依序跑，前一批的結果決定後一批：

1. 雜訊：`TAG=seed2 SEED=2 ARMS=ctrl`、`TAG=seed3 SEED=3 ARMS=ctrl`。
2. 亮點候選：`ARMS="noprior nomakeup oneteach ot"`。
3. 瘦身和修正：`ARMS="nodds noclip nohead fixes"`。

```bash
ARMS="noprior nomakeup" bash scripts/launch_stage2_ablation.sh output/SDFlow/multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k_long "0"
```

### 第三階段：定案 v2 並做完整訓練

1. 依第二階段決定 v2 的設定：通過的亮點留下、沒用的 loss 刪掉、`fixes` 併入。
2. v2 完整訓練 × 3 個 seed。
3. 每個亮點各一列「拿掉它」的完整訓練（至少 1 個 seed，最好 3 個）。
4. `velocity_field=original` 的對照。

### 第四階段：論文用的東西

1. 用 v2 重跑所有 held-out 曲線和對照表（`--num_samples` 1000 以上）。
2. 附錄：對真實照片的指標、分組分析、速度、人工評估。
3. 圖：每個屬性的比較圖、強度連續變化圖、失敗案例。

---

## 5. 論文可以寫和不能寫的

| 可以寫（已有或即將有證據） | 現在不能寫 |
|---|---|
| 同準確率下 ID 比 SDFlow 高（等 held-out 數字確認） | 「正交分解」 |
| consistent reading 提升準確率（等 held-out 確認） | 「ID 達到 0.90」 |
| 重訓的 baseline 和官方權重相當（SDFlow 已確認） | 任何還沒消融過的元件「有貢獻」 |
| 方向庫是訓練時的老師，推論時只用 flow | NLL、bits/dim 的數值 |
| 限制：長髮保留、細框眼鏡、Male 的化妝傾向（若 H4 沒修好） | 「解耦更好」，除非嚴格版 Pres 支持 |
