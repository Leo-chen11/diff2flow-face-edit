# 相關研究整理（2021–2026）：人臉屬性編輯

本專案的設定是：在 StyleGAN2 的 W+ latent 上，用條件式 CNF 加上方向庫（每屬性 K 個方向）和學到的 residual，編輯真實人臉的屬性。目標是在保住身分（ID）的前提下提高編輯成功率，並延伸到多屬性同時編輯。

下面依「和本專案有多接近」分組。凡是 venue 沒有逐一核對的，都標示 arXiv 編號，投稿前請再查一次。

---

## 第一組：StyleGAN latent 上的條件式／非線性編輯

這組是論文的主要比較對象。

| 年 | 論文 | 做法 | 多屬性 | 和本專案的關係 |
|---|---|---|---|---|
| 2021 | **StyleFlow**（ACM TOG 2021）[arXiv 2008.02401](https://arxiv.org/abs/2008.02401) | 條件 CNF，把 W 對應到以屬性為條件的高斯；改條件再反推 | 依序編輯；論文中也做聯合條件 | 本方法的祖先，必比 |
| 2023 | **SDFlow**（Li, Huang, Shan, Zhang）[arXiv 2309.05314](https://arxiv.org/abs/2309.05314)，[code](https://github.com/phil329/SDFlow) | 語意編碼器加上條件 CNF，把 latent 分解成語意變數和無關變數，用互資訊做解耦 | 聲稱支援單一或多屬性 | **本專案的基底**，必比 |
| 2023 | **AdaTrans**（ICCV 2023）[arXiv 2307.07790](https://arxiv.org/abs/2307.07790)，[code](https://github.com/hzzone/adatrans) | 把編輯拆成多步，每步方向和大小隨屬性與 latent 而變；用 real NVP 約束軌跡 | 依序多屬性 | 和 SDFlow 同一團隊，最強的同類基準 |
| 2021 | **Latent Transformer**（ICCV 2021）[arXiv 2106.11895](https://arxiv.org/abs/2106.11895) | W+ 上學一個 transformer 來改變單一屬性 | 依序多屬性，不限順序 | 常用基準 |
| 2021 | **ISF-GAN / Instance-aware latent space search**（IJCAI 2021）[arXiv 2105.12660](https://arxiv.org/abs/2105.12660) | 每張臉各自搜尋編輯方向 | 有 | 「每張臉不同方向」，概念接近我們的方向庫加 residual |
| 2023 | **ID-Style** [arXiv 2309.14267](https://arxiv.org/abs/2309.14267) | 全域可學方向，加上每張臉的強度預測器（IAIP） | 多屬性，強調保 ID | 和我們的「方向庫 + per-face magnitude」最像 |
| 2023 | **Latent Traversals as Potential Flows**（ICML 2023） | 把 latent 的移動當成學到的位勢流 | — | flow 型的非線性移動，相關工作 |
| 2020–21 | **InterFaceGAN**、**GANSpace**、**StyleSpace**（CVPR 2021） | 線性方向，或 S 空間的通道 | 相加 | 對應我們的 `scripts/linear_baseline.py` |

---

## 第二組：專門處理多屬性疊加

| 年 | 論文 | 做法 | 指出的問題 |
|---|---|---|---|
| 2023 | **DyStyle**（WACV 2023）[arXiv 2109.10737](https://arxiv.org/abs/2109.10737) | 結構隨樣本改變的動態網路，加上多屬性對比學習 | **依序編輯會累積誤差；多屬性在 latent 中互相糾纏** |
| 2022 | **Latent-to-Latent mapper**（WACV 2022）[IEEE 9706683](https://ieeexplore.ieee.org/document/9706683/) | 學一個 latent 到 latent 的映射，同時編輯多個屬性並保 ID | 直接相加各屬性方向會互相干擾 |
| 2021 | **Talk-to-Edit**（ICCV 2021） | 對話式、在語意場上連續多輪編輯 | 多輪編輯的一致性 |
| 2022 | **FLAME**（ACM MM 2022）[arXiv 2207.09855](https://arxiv.org/abs/2207.09855) | 找出解耦的線性方向，另外處理屬性的風格 | — |

這組論文的共同結論：**單純把各屬性的方向相加會互相干擾，依序編輯會累積誤差。** 這正是本專案 `scripts/eval_multi_attr.py` 要量化的兩件事。

---

## 第三組：擴散模型時代

這組是背景，用來說明為什麼還在 StyleGAN 上做。

| 年 | 論文 | 重點 |
|---|---|---|
| 2022 | Diffusion Autoencoders（CVPR 2022）、DiffusionCLIP（CVPR 2022） | 擴散模型的語意 latent，可做線性屬性編輯 |
| 2023 | Asyrp（ICLR 2023） | 擴散模型 h-space 的語意方向 |
| 2024 | **Concept Sliders**（ECCV 2024）[arXiv 2311.12092](https://arxiv.org/abs/2311.12092) | LoRA 滑桿，**可以組合多個**；年齡、表情、臉型等人臉屬性。是「多屬性組合」在擴散模型上的代表 |
| 2025 | RigFace [arXiv 2502.02465](https://arxiv.org/abs/2502.02465) | 3DMM 控制訊號加上微調的 Stable Diffusion，加身分編碼器 |
| 2025 | InstaFace [arXiv 2502.20577](https://arxiv.org/abs/2502.20577) | 單張圖推論，身分保留模組 |
| 2025 | FaceCrafter [arXiv 2505.15313](https://arxiv.org/abs/2505.15313) | 以身分為條件的擴散模型，姿勢、表情、情緒分開控制 |
| 2025 | 3D-aware few-shot 屬性編輯 [arXiv 2510.18287](https://arxiv.org/abs/2510.18287) | 3D 感知生成模型上，少樣本、保 ID 的屬性編輯 |
| 2026 | **LaTo**（ICLR 2026）[arXiv 2509.25731](https://arxiv.org/abs/2509.25731) | 以臉部特徵點為 token 的擴散 transformer，細粒度、保 ID 的編輯 |

**定位上的說法：** 擴散模型的畫質好，但每次編輯要跑幾十步，身分也要額外模組才能保住。StyleGAN latent 編輯是單次前向傳播，ID 和屬性可以分開量化，而且有大量可比較的基準和公開協定。

2024–2026 年 StyleGAN latent 編輯的新論文較少，多半發在期刊。建議在 Google Scholar 看 **AdaTrans 和 SDFlow 的「被引用」列表**，補齊最新的同類工作。

---

## 本專案的定位

- **和第一組的差異：** 方向庫（每屬性 K 個方向，加上 gate）、學到的 residual、ControlNet 注入；評估一律在**相同 ID 下**比較編輯率，判定用**獨立的 R50**，不用訓練用的分類器。
- **已有的實驗證據：**
  - 比線性基準好的部分，來自 residual：眼鏡 +30、Bangs +16–20、Male +10–16。
  - gate 和 ControlNet 在消融實驗中沒有可量測的貢獻。
- **屬性疊加要對齊的比較對象：**
  - StyleFlow、Latent Transformer、AdaTrans：依序編輯，對應我們的 `--compose seq`。
  - DyStyle、Latent-to-Latent：同時編輯，對應 `--compose sum` 和 `orth`。

---

## 屬性疊加（最多三個屬性）的擴充設計

### 第一步：先量現況，不用訓練（已實作）

腳本是 `scripts/eval_multi_attr.py`。

**比較三種組合方式：**

| 方式 | 做法 | 對應的文獻做法 |
|---|---|---|
| `sum` | 每個屬性從同一張原圖各算一次，把校準過的 W+ delta 和 ControlNet skips 相加 | 直接相加（Latent-to-Latent 指出的干擾問題） |
| `orth` | 相加前，逐層把每個 delta 在「其他屬性 delta 張成的空間」上的分量去掉 | 不用訓練的去干擾做法 |
| `seq` | 改一個屬性、重新生成、從新的臉重新讀條件，再改下一個 | StyleFlow、Latent Transformer、AdaTrans 的依序編輯（DyStyle 指出的誤差累積） |

**指標：**
- 每個屬性的 R50 成功率。
- **全部成功率**：每個被編輯的屬性都成功的比例。
- 同一批臉「一次只改一個屬性」時的成功率，以及兩者的差，即**疊加損耗**。
- ID_ind、LPIPS。
- 沒被編輯的屬性變了多少。

**組合：**
- 預設是 5 個屬性的全部 10 組雙屬性。
- 再加三組三屬性：
  - 眼鏡 + 笑 + 瀏海（局部 × 局部）
  - 性別 + 年齡 + 笑（全域 × 全域 × 表情）
  - 性別 + 眼鏡 + 年齡（混合）

### 第二步：如果疊加損耗大，再改訓練（只寫設計，尚未實作）

- 新參數 `--multi_edit_prob p`、`--multi_edit_max 3`：訓練時以機率 p 讓一個樣本同時編輯 2～3 個屬性。組合方式用第一步勝出的那種，讓訓練和推論一致。
- 損失：
  - 每個被編輯的屬性各算一個目標損失（hinge）。
  - preserve 損失只套在沒被編輯的屬性上。
  - ID 損失照舊。
- 程式上要把 `train_sdflow.py` 的 `mid_idx`（每個樣本一個屬性）改成 (B, A) 的遮罩。
- **可寫成論文貢獻的版本：** 在相同 ID 下，比較 sum、orth、seq 和「訓練過疊加」的全部成功率。若訓練版明顯較好，就是「疊加感知訓練」。
