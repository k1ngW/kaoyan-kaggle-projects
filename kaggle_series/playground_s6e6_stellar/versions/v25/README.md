# v25 — S6E6 收尾阶段：低开发成本高收益 🔥

> **决定日期**: 2026-06-18
> **前提**: v24 验证第三层范式，无论成败，v25 做最后冲刺
> **方向**: 不动新模型，在现有输出上做后处理优化 + 伪标签回炉

---

## 总体策略

不用再训新单模或新 Stacker——v8.2/v17/v18/v20 单模天花板已到，v23/v24 融合方案已上。v25 只做三件事：

```
① 后处理提纯（不碰模型，优化已有输出）
② 伪标签回炉（用强模型打标签，喂回训练）
③ 任务结构重构（二进制链替代三分类）
```

---

## v25.1 阈值优化 🔥🔥

> **成本**: 30 秒 | **预期**: +0.0003-0.0008 | **前置**: v24 或 v19 OOF 任意一份

### 原理

每类的最优决策边界不是等概率 1/3。你的分析已确认 STAR 被系统性低估（bias=-0.74），阈值调高 1.2× → STAR 召回率 +5%。

### 实现

```python
from scipy.optimize import minimize

def optimize_thresholds(oof_probs, y_true):
    def objective(t):
        preds = np.argmax(oof_probs * t, axis=1)
        return -balanced_accuracy_score(y_true, preds)

    result = minimize(objective, x0=[1.0, 1.0, 1.0],
                      bounds=[(0.5, 2.0)]*3, method='L-BFGS-B')

    print(f'Optimal: GALAXY={result.x[0]:.3f} QSO={result.x[1]:.3f} STAR={result.x[2]:.3f}')
    return result.x

# 应用最优阈值
final_probs = test_probs * optimal_thresholds
final_preds = np.argmax(final_probs, axis=1)
```

### 关键

- 在 **OOF** 上搜阈值，不在 LB 上搜（会过拟合 LB）
- 搜到后在 test 上应用
- 如果搜完的 CV > 原始 CV → 阈值有效 → 提交

---

## v25.2 堆叠器融合 🔥

> **成本**: 10 分钟 | **预期**: +0.0002-0.0004 | **前置**: v23 + v24 都有 OOF

### 原理

两个 Stacker 的预测错误不相关 → 简单加权就可能涨。

### 实现

```python
# v23 test 预测 + v24 test 预测 → 加权
v23_test = np.load('v23_meta_test.npy')      # v23 LogReg
v24_test = np.load('v24_test.npy')           # v24 TabPFN

for w in np.linspace(0.3, 0.7, 9):  # 搜 v23 权重
    blended = w * v23_test + (1-w) * v24_test
    # 在 OOF 上搜最优 w（不是 test！）
    cv_score = balanced_accuracy_score(y, blended.argmax(1))
    print(f'v23_w={w:.2f} CV={cv_score:.5f}')

# 如果最优点在 0.4-0.6 之间 → 有增益
# 如果在 0.0 或 1.0 → 有一个 Stacker 纯粹拖后腿，放弃
```

### 条件

- v23.1 和 v24 必须都跑过，有双方 OOF
- 如果一方明显弱于另一方 → 跳过，只用强的

---

## v25.3 伪标签回炉 🔥🔥🔥

> **成本**: 0.5 天 | **预期**: +0.0005-0.0015 | **前置**: 有 LB ≥ 0.97 的强模型

### 原理

```
57万 train + 24万 test
    ↓ 用 v24/v19 预测 test
    ↓ 挑置信度 > 0.95 的样本
    ↓ 打标签 → 混入训练集
    ↓ v17/v18 重训
```

### 实现

```python
def generate_pseudo_labels(model, test, threshold=0.95):
    probs = model.predict_proba(test)
    confidence = probs.max(axis=1)
    mask = confidence >= threshold
    pseudo = test[mask].copy()
    pseudo['class'] = probs[mask].argmax(axis=1)
    print(f'Pseudo labeled: {mask.sum()} / {len(test)} ({mask.mean():.1%})')
    return pseudo

# 1. 用 v24（或 v19）生成伪标签
pseudo = generate_pseudo_labels(clf_v24, test_raw, threshold=0.95)

# 2. 混入训练集，伪标签权重递减
train_extended = pd.concat([train, pseudo], ignore_index=True)
sample_weight = np.ones(len(train_extended))
sample_weight[len(train):] = 0.5  # 伪标签半信任

# 3. 用扩展训练集重新训 v17 CatBoost / v18 RealMLP
#    → 产新版单模 OOF → 喂回 Stacker
```

### 风险控制

| 措施 | 目的 |
|------|------|
| 阈值 ≥ 0.95 | 只取极确定样本，不喂毒 |
| 伪标签权重 0.5 | 即使标错了影响也减半 |
| OOF 对比 CV | 加了伪标签后 CV ↑ 才是真涨 |
| 最多 1 轮 | 不自举循环（伪标签标伪标签 = 爆炸） |

### 条件

- 打标签的模型 LB ≥ 0.97 → 伪标签可信
- 不是用 v24 的预测 → 喂回 v24 自己（等于自欺欺人）
- 正确做法：**v24 打标签 → 喂 v17/v18**（跨架构交叉验证式伪标签）

---

## v25.4 二进制链 🔥🔥

> **成本**: 1 天 | **预期**: +0.0005-0.0010 | **前置**: 前三件做完还差最后一口

### 原理

帖子 #704885 验证的思路：

```
三分类:                          二进制链:
GALAXY / QSO / STAR              ┌─ 第一关: GALAXY vs (QSO+STAR)
       ↓                         │          如果是 GALAXY → 结束
  一个模型辨三种                  │
                                 └─ 第二关: QSO vs STAR
                                         最终分类
```

每关只做二分类，模型专注度更高。第一关受益于 GALAXY 和 non-GALAXY 的物理本质差异（吸收线 vs 发射线），第二关不需要同时跟多数类竞争。

### 实现

```python
# 第一关: GALAXY vs Rest
y_binary1 = (y_train == 'GALAXY').astype(int)
clf1 = XGBClassifier / CatBoostClassifier
clf1.fit(X_train, y_binary1)

# 第二关: QSO vs STAR（只在非 GALAXY 样本上训）
mask = y_train != 'GALAXY'
y_binary2 = (y_train[mask] == 'QSO').astype(int)
clf2 = XGBClassifier / CatBoostClassifier
clf2.fit(X_train[mask], y_binary2)

# 预测
prob_galaxy = clf1.predict_proba(X_test)[:, 1]
is_not_galaxy = prob_galaxy < 0.5  # 或搜最优阈值

final = np.full(len(X_test), 'GALAXY')
final[is_not_galaxy] = clf2.predict(X_test[is_not_galaxy])  # QSO or STAR
```

### 注意事项

- 不限于 XGB/Cat，RealMLP 和 TabM 也能做二进制链
- 二进制链的 OOF 也可以喂回 v23/v24 Stacker
- 如果前三件已经冲够了，这步可以留到下一场 Playground 再做

---

## v25.5 OOF 共识度特征 🔥

> **成本**: 0.5 天 | **预期**: +0.0002-0.0005 | **前置**: 有 6+ 模型的 OOF

### 原理

```
"6 个模型都说它是 GALAXY" vs "3 个说 GALAXY、3 个说 QSO"
    → 前者的预测更值得信任

量化这个信任度作为 meta 特征喂给 Stacker
```

### 实现

```python
# 6 模型的 OOF prob: (N, 3) × 6 → 特征
all_oofs = np.stack([oof_xgb0, oof_xgb1, oof_realmlp0, ...])  # (6, N, 3)

# 共识度特征
probs_std = all_oofs.std(axis=0)            # 6 人在每类上的分歧
confidence = all_oofs.max(axis=2).mean(0)   # 平均自信度
entropy = -(probs * np.log(probs)).sum(2).mean(0)  # 平均熵
disagreement = (all_oofs.argmax(2) != all_oofs.argmax(2)[0]).mean(0)  # 不同意率

# 拼接进 X_train
X_train_enhanced = np.concatenate([X_train, probs_std, confidence[:,None], ...], axis=1)
```

## 堆叠器现在就能预测"自己有多不确定"

### 架构优势
- 如果共识度高（所有基模型一致）→ 几乎一定对
- 如果共识度低（基模型分化）→ 需要 Stacker 发挥
Stacker 可以看到这些模式，给不确定样本分配更保守的概率。

---

## 优先级 & 依赖

```
v24 跑出结果
    │
    ├── v25.1 阈值优化         ← 30 秒，随时可跑
    │
    ├── v25.2 堆叠器融合       ← 需要 v23 + v24 双 OOF
    │
    ├── v25.3 伪标签回炉       ← 需要 v24 (或 v19) 强到能打标签
    │     └── 产出新版 v17/v18 → 可选：喂回 v23 Stacker
    │
    ├── v25.5 OOF 共识度特征   ← 有 6 模 OOF 即可
    │
    └── v25.4 二进制链         ← 前三件做完还差最后一口时启用
```

## 预估总收益

| 版本 | 组件 | 预期单件 ∆LB | 累积 LB |
|------|------|:--:|------|
| v19 | 当前最高 | — | 0.97054 |
| +v25.1 | 阈值优化 | +0.0005 | 0.97104 |
| +v25.2 | 堆叠器融合 | +0.0003 | 0.97134 |
| +v25.3 | 伪标签回炉 | +0.0010 | 0.97234 |
| +v25.5 | 共识度特征 | +0.0003 | 0.97264 |
| +v25.4 | 二进制链 | +0.0007 | **0.97334** |

> 单个预期可叠加（每件互不干扰），但边际效应递减。保守估计全套跑完 **LB ≥ 0.9715**，乐观冲 **0.973+**。

---

## 运行记录

| 日期 | Kaggle ID | 版本 | 描述 | CV | LB |
|------|:--:|------|------|------|------|
| ⬜ | ⬜ | v25.1 | 阈值优化 | ⬜ | ⬜ |
| ⬜ | ⬜ | v25.2 | 堆叠器融合 | ⬜ | ⬜ |
| ⬜ | ⬜ | v25.3 | 伪标签回炉 | ⬜ | ⬜ |
| ⬜ | ⬜ | v25.5 | OOF 共识度特征 | ⬜ | ⬜ |
| ⬜ | ⬜ | v25.4 | 二进制链 | ⬜ | ⬜ |
