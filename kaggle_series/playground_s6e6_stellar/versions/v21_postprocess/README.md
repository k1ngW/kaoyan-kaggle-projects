# v21 — 四模型后处理优化

**前提**: 4 个模型的 OOF 文件齐全（v8 / v17_cat / v18_realmlp / v20_tabm）

**特点**: 纯 OOF 操作，不动 GPU，不动数据/特征/模型训练

---

## 输入

```
oof/
├── v8_oof_preds.npy        (N, 3)   XGB            CV 0.96665
├── v17_cat_oof.npy          (N, 3)   CatBoost       CV 0.96887
├── v18_realmlp_oof.npy      (N, 3)   RealMLP-5      CV 0.96904
└── v20_tabm_oof.npy         (N, 3)   TabM           CV ?
```

---

## 后处理方案

### 1. 4 模 Grid Search（基线，1 min）

```
blend = w1*v8 + w2*cat + w3*realmlp + w4*tabm
Grid Search w1~w4, sum=1, 步长 0.02
目标: max Balanced Accuracy
```

**目的**: 确认 TabM 加入后的 CV 天花板，和 v19 三模对比。

---

### 2. 12 权重（每模型 × 每类独立，10 min）

```
GALAXY: w1_g*v8_g + w2_g*cat_g + w3_g*realmlp_g + w4_g*tabm_g
QSO:     w1_q*v8_q + w2_q*cat_q + w3_q*realmlp_q + w4_q*tabm_q
STAR:    w1_s*v8_s + w2_s*cat_s + w3_s*realmlp_s + w4_s*tabm_s

每类独立 Grid Search，12 个权重
```

**目的**: 解决 v19 "一个权重管所有类" 的问题。TabM QSO 强但 STAR 可能弱，每类独立分配权重。

---

### 3. XGB Stacking（非线性融合，20 min）

```
X_meta = [oof_v8, oof_cat, oof_realmlp, oof_tabm]  # (N, 12) 拼接
y_meta = y

5-fold:
  fold 1-4 OOF → 训 XGB meta model → 预测 fold 5
  fold 1-4 OOF → 训 XGB meta model → 预测 test（平均）

XGB 参数: max_depth=3~5, n_estimators=300, 防过拟合
```

**目的**: LogReg 只能学线性组合，XGB 能学非线性交互（比如"TabM 和 realmlp 同时给高分才选 STAR"）。

---

### 4. 阈值调优（argmax 前缩放，10 min）

```
scipy.minimize 搜 3 个缩放因子 [t_galaxy, t_qso, t_star]
scaled = blend * [t_g, t_q, t_s]
pred = scaled.argmax(1)

目标: max Balanced Accuracy
```

**目的**: 调整 argmax 判定边界，适合类别不平衡或某类置信度系统性偏高的情况。

---

### 5. 每类独立 XGB Stacking（终极方案，30 min）

```
QSO 通道:   XGB([所有模型的QSO概率]) → 融合后的 QSO 概率
STAR 通道:  XGB([所有模型的STAR概率]) → 融合后的 STAR 概率
GALAXY 通道: XGB([所有模型的GALAXY概率]) → 融合后的 GALAXY 概率

三路拼回 → 归一化 → argmax
```

**目的**: 和 v20 notebook 的 run_blend() 同思路，但 meta model 从 LogReg 升级到 XGB，非线性表达能力更强。

---

## 执行顺序

```
① Grid Search (4模) → 确立基线
② 12 权重 → 升级 v19 逻辑
③ XGB Stacking → 如果 ② 提升不够
④ 阈值调优 → 叠加到 ②/③ 的融合结果上
⑤ 每类 XGB Stacking → 如果 ③ 提升不够，终极试探
```

## 评估标准

- CV 提升 > 0.0003 → 提交
- CV 提升 < 0.0001 → 放弃该方案
- LB 噪音大，以 CV 为主、LB 为辅

---

## 前置条件

- [ ] v20 TabM notebook 跑完
- [ ] 下载 oof.npy + test_preds.npy 到 versions/v20_tabm/
- [ ] 下载 experiment_log_v20_tabm.json 到 experiment_logs/
