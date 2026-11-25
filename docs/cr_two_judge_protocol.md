# Consistent reading（CR）的公平評估：R50 與 r18 一起用

> **2026-11-25 更新：SOTA 比較一律不加 CR。** 我們、SDFlow、StyleFlow 都用各自原本的讀值直接編輯，主表、7 點表、折線圖預設都不含 CR 的欄位。CR 的版本（`WITH_CR=1`、`--variants cr/both`）只留作分析 CR 這條規則本身，不是比較的一部分。下面的共識臉設計仍然適用。

目的：讓「我們的方法 + CR」的分數，沒有任何一個地方可以被說成是「拿評分的 judge 去幫自己作弊」。

---

## 1. 疑慮到底在哪裡（要先講清楚）

目前的 held-out 曲線（`--face_set direction`，至今所有曲線都是這個）：

1. **R50 讀原圖**，決定「使用者要求」：R50 說有戴眼鏡 → 要求「拿掉」；說沒戴 → 要求「戴上」。
2. 只用 **R50 讀得很清楚的臉**（讀值 < 0.35 或 > 0.65）。
3. 編輯後再用 **R50 打分**。

CR 的作用是「模型一定照要求的方向編」。問題在於：**要求本身是 R50 定的**。如果 R50 把一張臉讀錯（例如把中年人讀成年輕），要求就會是「變老」，CR 讓模型乖乖變老，最後又由同一個 R50 打分，它當然說「成功」。這時 R50 的分數有一部分是「模型跟 R50 意見一致」，不是「真的編對」。

數據也看得出這個傾向：long → long+CR（強度 0.95）R50 +3.5、r18 只有 +1.4；Young 的 R50 +8.7。r18 的增益比較小，就是因為 r18 沒有參與決定要求。

**所以疑慮只發生在 R50 和 r18 讀法不一樣的那些臉上。** 兩個 judge 讀法一致、又都很清楚的臉，要求是誰定的都一樣，沒有偏袒問題。

---

## 2. 原則：judge 永遠不進入方法

| 東西 | 可以用 | 不可以用 |
|---|---|---|
| CR 規則（推論時） | 使用者要求的方向、**模型自己的** conditioner 讀值、固定 margin 0.1 | 任何 judge（R50、r18、ArcFace、CLIP）的讀值 |
| margin | 在開發臉上定好（0.05/0.1/0.2 結果相同），之後不再改 | 在 held-out 上挑 |
| judge | 只用來**定義測試題目**（哪些臉、要求哪個方向）與**打分** | 放進模型或 CR |

**為什麼不能把 R50/r18 放進 CR：** 那等於推論時偷看考官的答案，任何方法這樣做準確率都會變高，這才是真的作弊。所以「把 R50 跟 r18 一起用到 CR 裡」正確的做法**不是**把它們塞進 CR 規則，而是**一起用來出題**（下一節）。

---

## 3. 設定：共識臉（`--face_set consensus`）

**只在兩個 judge 都讀得很清楚、而且讀法一致的臉上評估：**

- R50 讀值 < 0.35 或 > 0.65，**且** r18 讀值 < 0.35 或 > 0.65，**且** 兩者同一邊。
- 要求的方向由這個一致的讀法決定 → **不管由哪個 judge 當「使用者」，題目都一樣**，R50 沒有主場優勢。
- 每個屬性、每個方法用**完全相同的臉**（只取決於原圖與兩個 judge，跟方法無關）。
- 兩個 judge 的準確率、ArcFace、屬性保持全部在這批臉上算。
- 表格的說明文字會自動寫出每個屬性用了幾張臉（覆蓋率要報）。

**預期：** CR 在共識臉上的增益會變小。**剩下的增益才是真的**：要求不含糊，CR 修正的只是**模型自己**把原圖讀錯的情況。這正是 CR 要解決的問題，也最難被質疑。

為什麼不用「R50 和 r18 平均」當讀值：平均之後，一個 judge 的錯誤仍然可以決定要求，只是被稀釋；共識臉直接把有爭議的臉排除，比較乾淨，也比較好解釋。

---

## 4. 再加上四個檢查（各自擋掉一種質疑）

| 檢查 | 擋掉的質疑 | 怎麼做 |
|---|---|---|
| A. 有 CR 和沒 CR **都報** | 「你只挑對自己有利的版本」 | 主表放 Ours（no CR）與 Ours+CR 兩欄；CR 當作獨立消融的貢獻 |
| B. CR **也給 SDFlow 和 StyleFlow** | 「CR 是只給自己的推論技巧」 | 這兩個 baseline 也是「先讀原圖再編」，同樣會讀錯；用同一條規則（各自的讀值、同一個 margin）。其他方法（InterFaceGAN、GradCtrl、PluGeN、AdaTrans、DeltaEdit、LT）本來就**無條件照要求方向移動**，CR 對它們沒有東西可改 —— CR 只是讓條件式 flow 拿到這些方法本來就有的性質 |
| C. 交換測試（`--direction_judge r18`） | 「換一個 judge 出題結果就不同」 | 在舊的 face set 上，改由 r18 出題、R50 變成完全獨立的考官；CR 的增益若仍在，就不是 R50 偏袒。共識臉上這個測試自動成立 |
| D. 事先固定 | 「看了結果才調設定」 | margin 0.1、共識門檻 0.35/0.65、held-out offset 500 都在跑之前定好、寫在這份文件；不在 held-out 上改 |

可選：第三個 judge（CLIP zero-shot，訓練資料與 R50/r18 完全不同）當額外佐證。

---

## 5. 論文 / 報告上怎麼寫（一句話版）

> 編輯要求由兩個獨立屬性分類器（R50、ResNet-18）共同決定，只評估兩者都清楚且一致的臉，兩者的準確率都報告。CR 只使用要求方向與模型自己的讀值，不使用任何評估分類器；同一規則也套用在 SDFlow 與 StyleFlow 上，並報告有無 CR 的結果。

---

## 6. 之前訓練的 SOTA 要不要重訓？——**不用**

| 問題 | 答案 |
|---|---|
| 我改了自己模型的 batch size（4 → 8）／img_size，baseline 會受影響嗎？ | 不會。baseline 的訓練只用 W+ latent 與 r34 標籤，跟我們模型的 batch、img_size 完全無關，權重不用動 |
| 評估會受影響嗎？ | 不會。所有方法的評估一樣：StyleGAN2 生成 1024×1024，judge 一律 area 縮到 256，同一組 judge、同一批 held-out 臉。img_size 只影響**我們模型訓練時**算 loss 的縮圖 |
| 那什麼要重跑？ | **評估曲線**（只是推論，不是訓練）。只要評估規則改了（這次是改成共識臉），所有方法都要在新規則下重跑曲線，否則不是同一批臉，不能比。`method_table` 現在會檢查，臉不同就拒絕出表 |
| 要注意的地方 | batch 8 的消融只能跟 batch 8 的對照組比（不能拿 batch 4 的 long 當對照）。SOTA 主表用的是最終模型，跟消融的 batch 無關 |

---

## 7. 指令

全部在跑 held-out 曲線的那台機器上（模型、baseline、`data/` 都在的資料夾），用 `scripts/run_heldout_curves.sh`，`FACE_SET=consensus` 的結果放在 `output/curves_consensus/`，不會覆蓋舊的曲線：

```bash
FACE_SET=consensus METHODS="ours ours_nocr sdflow5 sdflow5_cr styleflow styleflow_cr adatrans gradctrl plugen deltaedit lt interfacegan" bash scripts/run_heldout_curves.sh
```

出表（同準確率比 ID，兩個 judge 各一次；再加固定強度）：

```bash
python -m scripts.method_table --curves output/curves_consensus --target 90
python -m scripts.method_table --curves output/curves_consensus --target 90 --judge r18
```

畫圖：

```bash
CURVES=output/curves_consensus bash scripts/plot_all_methods.sh
```
