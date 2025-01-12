# 羽球雙打選手移動預測

[海報介紹](https://github.com/user-attachments/files/18389109/badminton.2.pdf)

[羽球雙打選手移動預測論文](https://github.com/user-attachments/files/18389111/default.pdf)

---

## 摘要
透過數據分析來提升球員表現的需求日益增加。然而，針對羽球球員移動預測的相關研究多集中於單打，缺乏雙打的相關研究。

本研究提出了基於LSTM、Attention和ResNet的深度學習模型——Double ShuttleNet，旨在預測雙打比賽中的選手座標、回球種類及回球座標。

---

## 背景與動機
羽球是一項高速度且具高度戰術性的運動。雙打比賽中，選手間的協調與配合是影響比賽結果的關鍵。然而，現有研究多聚焦於單打分析，對雙打的深入探討相對缺乏。因此，本研究專注於雙打比賽，開發預測模型並視覺化結果。

---

## 方法概述

### 模型輸入
- 前τ球四位球員的座標
- 前τ球的回球種類
- 前τ球的回球座標

### 1. 模型架構
**Double ShuttleNet**使用以下核心技術：

- **LSTM（長短期記憶模型）**：
  - 捕捉時序資料中的關聯性，學習並記憶前面時刻的移動軌跡與擊打資訊。

- **Multi-head Self-Attention**：
  - 通過Query、Key、Value計算Attention Score，聚焦序列中的重要部分，並利用多頭機制捕捉不同回球類型的影響。

- **ResNet（殘差網絡）**：
  - 引入Skip Connection，減少梯度消失和過擬合，提升模型的訓練效率與準確性。

### 2. 資料來源
- **比賽資料**：[2022年印尼羽毛球大師賽雙打比賽數據](https://github.com/HuangYuHsien/S2DoublesDataset)
- **資料內容**：當前回合數、擊球次數、球員位置、回球種類及得失分原因。

### 3. 預測目標
- **座標預測**：預測下一回合中四位選手的位置與回球座標。
- **回球種類預測**：分類預測下一次擊球的技術類型。

---

## 實驗與結果

### 1. 評估方法
- **損失函數**：
  - **MSE（均方誤差）**：用於座標預測。
  - **Weighted Cross-Entropy（加權交叉熵）**：用於回球種類預測。
- **驗證方法**：Stratified KFold 交叉驗證，確保每個類別的比例一致。

### 2. 模型表現
| 模型                      | ACC (↑)   | MSE (↓)   | MAE (↓)   |
|---------------------------|-----------|-----------|-----------|
| Seq2Seq                  | 0.435     | 0.061     | 0.219     |
| LSTM                     | 0.510     | 0.026     | 0.152     |
| Multi-head Self-Attention| 0.505     | 0.050     | 0.200     |
| **Double ShuttleNet**    | **0.525** | **0.014** | **0.092** |

---

## 視覺化範例
![視覺化結果](https://github.com/user-attachments/assets/e0f125c9-e4b9-4164-a587-34a1932d3d5c)

利用前五球預測第六球的四位選手位置、回球座標與回球類型。

- **標誌解釋**：
  - A、B、C、D：四位球員的位置。
  - `@`：選手的擊球位置。
  - 紅色的T：回球座標。
  - 藍色的AP、BP、CP、DP：A、B、C、D四位球員的預測位置。
  - NB：模型預測的回球座標。
