# v8.1 — ❌ 死胡同（Chris OvR 堆叠器，不是纯 XGBoost）

> **决定日期**: 2026-06-15 | **放弃日期**: 2026-06-15
> **死因**: Chris Deotte XGBoost OvR notebook 本质是 XGBoost + 外部 OOF 的 OvR 堆叠方案
> 纯 XGBoost (ADD_EXT=False) 仅 CV 0.9609，不如旧 v8 多分类方案 (0.96665)
> **继任**: v8.2 — 基于旧 v8 多分类 XGBoost 加特征

---

## 为什么 v8.1

```
v23 Cell 4 权重分析 →
  三类 Top-5 全有 v8，GALAXY #1、QSO #1/#2
  但 v8 CV 0.96665 / LB 0.96757 ← 五模倒数第一
  树模型唯一代表，LogReg 别无选择

v8.1 逻辑:
  把树模型从 0.96757 拉到 0.969+ → ×3 seed
  → Stacker 有了强锚点 → CV 0.969+ / LB 0.971+
```

---

## 技术方案

### 起点：Chris Deotte XGBoost OvR

| 属性 | 旧 v8 | Chris XGBoost OvR |
|------|------|------|
| Kaggle Kernel | (待查) | [PS6e6 XGBoost](https://www.kaggle.com/code/cdeotte/ps6e6-xgboost) |
| 策略 | 多分类 XGBoost | **3 个二分类 OvR** |
| 特征数 | 170+ (TE + 分箱) | **26** (纯手工特征) |
| 训练方式 | 5 fold CV | 3 seed × 5 fold + 内部 LogReg blending |
| **CV** | 0.96665 | **0.9686** |
| **LB** | 0.96757 | **0.9695** |

### v8.1 改进计划

| 改动 | 预期 ∆LB | 来源 |
|------|:--:|------|
| 10 对颜色差 + mag_stats (mean/std/min/max/range) | +0.0005 | STRATEGY.md |
| 红移×波段 + 波段/红移 (补到 10 个) | +0.0003 | STRATEGY.md |
| 通量 flux_* + log_flux | +0.0003 | STRATEGY.md |
| 天空坐标 sky_x/y/z + sin/cos | +0.0002 | |
| seed 数 3 → 5 | +0.0003 | |
| **目标** | **LB 0.971+** | |

架构不动：OvR + 内部 LogReg blending 保留，只在特征工程层加特征。

---

## 运行记录

| 日期 | 版本 | 描述 | CV | LB |
|------|------|------|------|------|
| 2026-06-15 | v8.1-base | Chris OvR 原版 (ADD_EXT=False, CV 0.9609) | ❌ 放弃 | — |

**发现：** Chris notebook 本质是 XGBoost + 外部 OOF 的 OvR 堆叠器，不是纯 XGBoost。关掉外部模型只剩 CV 0.9609，不如旧 v8 多分类方案 (0.96665)。v8.1 回退到旧 v8 多分类 XGBoost 加特征路线。

---

## 文件

```
versions/v8.1/
├── README.md                    ← 本文件
├── s6e6-xgb-ovr-v8-1.ipynb      ⬜ 从 Kaggle 下载
├── submission_v8-1_xxx.csv      ⬜
├── oof.npy                      ⬜
└── test_preds.npy               ⬜
```

## 状态

- ⏸ 等待 Kaggle notebook 跑通 baseline
