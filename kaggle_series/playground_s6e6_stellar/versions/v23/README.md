# v23 — Chris LogReg Stacker（终结传统融合，升级 Level 2 活规则）

> **决定日期**: 2026-06-14
> **前提**: v19/v21 确认 CSV 混合天花板 LB 0.97050，v22 确认单模天花板 ~0.969
> **方向**: 从模型层/特征层迭代 → 转向后处理层升级

---

## 为什么 v23

```
你的单模水平                        你的融合水平
─────────────────────────────────   ─────────────────────────────
v18 RealMLP       CV 0.96904       v19 Grid Search 3权重      LB 0.97054
v17 CatBoost      CV 0.96887       v21 4模CSV混合               LB 0.97050
v20 TabM          CV 0.96862       ← 全是 Level 1 死规则
v8  XGBoost       CV 0.96665
─────────────────────────────────
单模 ≈ 榜上 0.971 选手的水平        融合落后一个时代
                                    差距：0.001-0.002
```

**v23 只做一件事：把 0.969 水平的单模，配上 0.971 水平的融合。**

---

## 技术方案

### Chris Deotte LogReg Stacker

```python
# 核心流程
oof_logit = logit(clip(oof, 1e-15, 1-1e-15))   # ① prob → logit 变换
test_logit = logit(clip(test_preds, 1e-15, 1-1e-15))

X_meta = hstack([所有模型的 oof_logit])           # ② 拼接 meta 特征
y_meta = y

for seed in range(5):                             # ③ 5 seed × 5 fold = 25 次
    for fold in range(5):
        LogReg = Linear(n_models*3, 3)            # ④ PyTorch 单层 + L2
        loss = CrossEntropy + L2(weight_decay)
        → 训 → 预测

最终 = mean(25 次预测)                             # ⑤ 降 meta 层方差
```

### 5 个关键组件

| 组件 | 作用 | 不做会怎样 |
|------|------|------|
| prob → logit | 让概率空间线性可分 | raw 概率喂 LogReg = 退步 |
| L2 正则 | 自动压缩冗余特征权重 | 同框架模型高度相关 → 过拟合 |
| 5 seed × 5 fold | 降 meta 层方差 | 单 seed 的融合规则不稳定 |
| 类别权重 | 防多数类主导 meta model | GALAXY 65% → LogReg 偏向它 |
| 模型权重可解释 | 知道哪个模型在哪类上被信任 | Grid Search 只有 3 个数 |

---

## 输入 OOF（Kaggle Dataset 路径）

### 已有（5 数据集，Kaggle 已验证 ✅）

| 模型 | Dataset | Owner | 文件（扁平目录） |
|------|------|------|------|
| v8 XGBoost (旧) | `s6e6-v8-oof` | weywuy | `oof.npy` · `test_preds.npy` |
| **v8.2 XGBoost** | `s6e6-v8-2-oof` | weywuy | `oof.npy` · `test_preds.npy` |
| v17 CatBoost | `s6e6-cat-oof` | weywuy | `oof.npy` · `test_preds.npy` |
| v18 RealMLP | `s6e6-realmlp5-oof` | weywuy | `oof.npy` · `test_preds.npy` |
| v20 TabM | `s6e6-tabm-oof` | weywuy | `oof.npy` · `test_preds.npy` · `oof_bias.npy` |
| v22a CatBoost | `s6e6-v22a-cat-oof` | weywuy | `v22a-cat_oof.npy` · `v22a-cat_test_preds.npy` |

### v23 Notebook 实际路径（Kaggle 已验证 ✅）

所有 OOF 挂载在 `/kaggle/input/datasets/weywuy/` 下，比赛数据在 `/kaggle/input/competitions/playground-series-s6e6/`。

```python
# 实际 Kaggle 路径（2026-06-15 验证）
OOF_BASE = '/kaggle/input/datasets/weywuy'
OOF_DATASETS = {
    'v8':    f'{OOF_BASE}/s6e6-v8-oof',             # 旧 XGBoost (LB 0.96757)
    'v8-2':  f'{OOF_BASE}/s6e6-v8-2-oof',           # v8.2 XGBoost (CV 0.96767)
    'v17':   f'{OOF_BASE}/s6e6-cat-oof',            # CatBoost (CV 0.96887)
    'v18':   f'{OOF_BASE}/s6e6-realmlp5-oof',       # RealMLP (CV 0.96904)
    'v20':   f'{OOF_BASE}/s6e6-tabm-oof',           # TabM (CV 0.96862)
    'v22a':  f'{OOF_BASE}/s6e6-v22a-cat-oof',       # CatBoost v22a (CV ~0.964)
}
COMP_ROOT = '/kaggle/input/competitions/playground-series-s6e6'
```

### 待跑（多 seed）

| 计划 | 改什么 | 预计 CV | 上线后 Dataset 名 |
|------|------|:--:|------|
| v17 seed=142 | `SEED=142` + `N_SPLITS=5`（改 seed，折数不变） | ~0.9688 | `s6e6-cat-oof-seed142` |
| v17 seed=242 | `SEED=242` | ~0.9688 | `s6e6-cat-oof-seed242` |
| v18 seed=142 | v18 notebook 改 `SEED=142` | ~0.9690 | `s6e6-realmlp5-oof-seed142` |
| v18 seed=242 | v18 notebook 改 `SEED=242` | ~0.9690 | `s6e6-realmlp5-oof-seed242` |

### 最终通道（v23.1）

```
v8-2_xgb     × 1 seed   = 1 通道  (CV 0.96767 ✅)
v17_cat      × 3 seed   = 3 通道  (42 ✅, 142 ⬜, 242 ⬜)
v18_realmlp  × 3 seed   = 3 通道  (42 ✅, 142 ⬜, 242 ⬜)
v20_tabm     × 1 seed   = 1 通道
v22a_cat     × 1 seed   = 1 通道  (退役: 被 v17 替代)
─────────────────────────────────
             9 通道, 4 架构, 6 数据集
```

---

## 预期收益

| 来源 | 预期 ∆LB |
|---|---|
| Grid Search → LogReg Stacker | +0.0010 |
| 3 模 → 5 模 (9 通道) | +0.0005 |
| logit 变换 + L2 正则 | +0.0003 |
| 5 seed × 5 fold 平均 | +0.0002 |
| OOF 置信度/共识度特征 | +0.0002 |
| **合计** | **+0.0020** |
| **目标 LB** | **0.9725+** |

---

## 与传统融合对比

| | v19 Grid Search | v21 CSV 混合 | **v23 Chris Stacker** |
|---|---|---|---|
| 融合方式 | 死权重 | 死权重 × 4 | **活权重（动态）** |
| 输入变换 | 无 | 无 | **logit** |
| Meta model | 无 | 无 | **PyTorch LogReg** |
| 正则化 | 无 | 无 | **L2 weight_decay** |
| 多 seed | 1 | 1 | **5 seed × 5 fold = 25** |
| 每类差异化 | 否 | Grid Search 尝试过 | **自动学到** |
| 模型权重可解释 | 3 个数 | 4 个数 | **27 个系数，每类不同** |
| 你的 LB | 0.97054 | 0.97050 | **?** |
| Chris 验证 | — | — | **CV 0.97028 (19 模)** |

---

## 参考

- [Chris-Deotte-3层Stacking冠军方案.md](../../knowledge/Chris-Deotte-3层Stacking冠军方案.md) — Chris 同款方案
- [后处理方案对比.md](../../knowledge/后处理方案对比.md) — 传统 vs Chris Stacker 详细对比
- [KGMON-Playbook-冠军方案参考.md](../../knowledge/KGMON-Playbook-冠军方案参考.md) — 验证 LogReg+L2 是 meta learner 最优解
- `ref_notebooks/gpu-logistic-regression-stacker.ipynb` — Chris 原始代码参考

---

## v22 最终结论（写入此处）

> v22a: 去掉 CTR3 + 哈希交互 → 严重过拟合 (learn-test gap=0.007), CV ~0.9644
> v22b: (待记录)
> v22c: 加回 CTR3 + hash 但特征底子不同 → (待记录)
>
> **结论: v17 的 153cat + CTR3 + hash + SDSS + depth=8 是联动体系，不能拆改。**
> **多样性不应靠改特征体系，应靠多 seed + 多架构 + Stacker 融合。**
>
> 传统 CSV 混合方案（v19/v21）在此终结。Level 1 到此为止。

---

## v23 实际结果（2026-06-15）

### 提交记录

| 日期 | Kaggle ID | 描述 | CV | LB |
|------|:--:|------|------|------|
| 2026-06-15 | 53706215 | 5通道 + StandardScaler + L2 | **0.96713** | **0.96809** |
| 2026-06-15 | 53705688 | (炸) class_weights 导致 GALAXY 塌陷 | 0.95398 | 0.95372 |

### 写入文件

```
versions/v23/
├── README.md                             ← 本文件
├── s6e6-chris-logreg-stacker-v23.ipynb   ← Kaggle Notebook
├── submission.csv                        ← LB 0.96809
├── v23_meta_oof.npy                      ← (577347, 3)
└── v23_meta_test.npy                     ← (247435, 3)
```

### 关键结论

| 发现 | 证据 |
|------|------|
| 5 通道 Stacker CV 0.96713，不如最优单模 v18 (0.96904) | 通道太少 + v8 底子弱 |
| v8 (XGBoost) 是权重 MVP，但本身 LB 仅 0.96757 | 树模型唯一代表，LogReg 别无选择 |
| class_weights 是毒药 | logit 空间样本不均衡已被消解 |
| v17 被 L2 压死（与 v22a 共线），不说明其无效 | 补 seed 后权重会自然转移 |

### 后续路线

| 版本 | 内容 | 策略 | 状态 |
|------|------|------|:--:|
| **v23.1** | v8.2 + v17×3 seed + v18×3 seed + v20 | 扩至 11 通道，LogReg 活权重 | **下一步** |
| **v23.2** | Chris OvR Stacker | 每类独立 LogReg（三头） | 备选 |

**v8.2 已完成：** CV 0.96767 / LB 0.96801。单模特征天花板确认（370→380 不涨）。OOF 已上传为 `s6e6-v8-2-oof`，可直接导入 v23.1。v8.2 替代旧 v8 成为 v23.1 的树模型通道。

**v23.2 说明：** Chris OvR notebook 架构有价值（三头 vs 单头），如果 v23.1 仍不超单模则切换。
