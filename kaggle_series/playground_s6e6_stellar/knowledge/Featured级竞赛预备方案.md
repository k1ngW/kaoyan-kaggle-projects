# Featured 级竞赛预备方案

> 基于 S6E6 经验，为未来 Featured 竞赛准备的第四层技术手册。

---

## 个人研究 → Featured 的差距

S6E6 是 Playground（入门级），Featured 的区别：

| | Playground | Featured |
|---|---|---|
| 参赛人数 | 2,000-3,000 | 1,000-3,000 |
| 竞争烈度 | Top 10% 有银牌 | Top 5% 有银牌 |
| 关键差距 | 特征工程 + 几个强单模 | **全流程自动化 + 多层堆叠** |
| 你的位置 | 当前 LB 0.97054，能拿银牌 | 同样方案在 Featured ≈ Top8-10% |

---

## Featured 选手的四种赢法

不是所有人都走四层框架。真实榜单上风格各异：

### 四种流派

| 风格 | 核心武器 | 例子 | 适用谁 |
|------|---------|------|------|
| **领域专家型** | 外人不知道的领域公式特征 + 一个强单模 | 天文学家打天体分类，加消光/金属丰度/色温公式 → XGB 直接银牌 | 你有该领域学术背景 |
| **单一杀手型** | 对一个框架控制到极限 | 把 CatBoost depth/l2_leaf_reg/ctr 调到极致 → 一个模进金牌区 | 你对某框架心理模型极深 |
| **黑马洞见型** | 发现别人忽略的 trick | 数据泄露/标签噪声/采样偏差 → 一个 trick 拉开所有人 | 运气 + 对数据的极端敏感 |
| **工程自动化型** | **四层框架** | 50 通道 + 深堆 + 伪标签 + 阈值优化 → 稳定金牌 | **你——无该领域学术背景时** |

### 四层框架在金牌区的定位

```
铜牌 | 银牌 | ───── 金牌 ───── | ── Top 1% ── | 冠军
      ↑          ↑                    ↑            ↑
   你的现状    +伪标签+阈值        +深堆3层     +独特insight
              +四层框架起步       +50通道      +全自动化
```

| 段位 | 标配 | 你差什么 |
|------|------|------|
| 银牌 (~Top20%) | 多架构单模 + 加权集成 | ✅ 已到 (v19 LB 0.97054) |
| 金牌 (~Top5%) | + 伪标签 + 阈值优化 | ⬜ 差两次 Playground 验证 |
| Top 1% | + 深堆 3 层 + 50+ 通道 | ⬜ Featured 再开 |
| 冠军 | + 领域独特 insight | — |

### 哪条路最适合你

你是**非天文学背景**打天文比赛，没有领域知识优势。所以自然走工程自动化——靠架构互补 + Stacker + 多通道，用计算力换领域知识。

这不是弱的路线，是**对你最正确的路线**。Chris Deotte（5 届 Kaggle Grandmaster）打 S6E4 就是这样赢的——用别人的公共 notebook 当通道，自己不写一行特征工程，靠 Stacker 层融合拿到金牌。四层框架就是这条路线的完整版。

**未来如果遇到你本专业领域的比赛（比如你学的东西），切领域专家型 + 工程自动化 = 两条腿一起跑，直接冲冠军。**

---

---

## 第四层四件套

### 1. 🔥🔥🔥 伪标签自训练（Pseudo-labeling）

**原理**：测试集中置信度最高的样本，当准标签喂回训练。

```python
def pseudo_label_boost(train, test, model_fn, threshold=0.95, rounds=3):
    for r in range(rounds):
        model = model_fn()
        model.fit(train.drop('class',1), train['class'])

        # 预测 test，选高置信度样本
        test_probs = model.predict_proba(test)
        confidence = test_probs.max(axis=1)
        pseudo_mask = confidence >= threshold
        pseudo = test[pseudo_mask].copy()
        pseudo['class'] = test_probs[pseudo_mask].argmax(axis=1)

        # 混入训练集，weight 递减
        train = pd.concat([train, pseudo], ignore_index=True)
        sample_weight = np.ones(len(train))
        sample_weight[len(train)-len(pseudo):] = 0.5 ** (r+1)  # 递减信任

        threshold *= 0.98  # 每轮降低门槛

    return train, model
```

**为什么有效**：57 万 train + 24 万 test，test 里天然有简单样本，喂回去让模型看到更多数据分布。

**S6E6 适用性**：test 里距离已知类别中心近的样本 → 伪标签几乎一定对 → 扩展训练集。

**前提条件**：基模型 LB ≥ 0.968 才值得做（准确性不够反哺是喂毒药）。

---

### 2. 🔥🔥 阈值优化（Threshold Optimization）

**原理**：每类的决策边界不是等概率 1/3，而是随类别先验和模型偏好偏移。

```python
from scipy.optimize import minimize

def optimize_thresholds(oof_probs, y_true):
    """搜索最优阈值，最大化 CV balanced accuracy"""
    def objective(thresholds):
        # thresholds: [t_GALAXY, t_QSO, t_STAR]
        preds = np.argmax(oof_probs * thresholds, axis=1)
        return -balanced_accuracy_score(y_true, preds)

    result = minimize(objective, x0=[1.0, 1.0, 1.0],
                      bounds=[(0.5, 2.0)]*3, method='L-BFGS-B')
    return result.x  # 最优阈值
```

**S6E6 经验**：STAR 类 bias=-0.74（被系统性低估），阈值调高 1.2× → STAR 召回率 +5%。

---

### 3. 🔥 对抗验证（Adversarial Validation）

**原理**：训练一个分类器判断"这条样本来自比赛数据还是原始 SDSS"，如果它能区分 → 有分布偏移 → 需要处理。

```python
# 构造对抗数据集
train['is_test'] = 0
test['is_test'] = 1
adv_data = pd.concat([train, test], ignore_index=True)

# 训练对抗分类器
adv_model = XGBClassifier()
adv_scores = cross_val_score(adv_model, adv_data.drop('is_test',1), adv_data['is_test'])
print(f"Adversarial AUC: {adv_scores.mean():.4f}")

# AUC > 0.65 → 显著偏移
if adv_scores.mean() > 0.65:
    # 方案 A: 按对抗分数加权样本
    adv_probs = adv_model.predict_proba(...)[:, 1]
    sample_weight = 1 / (adv_probs + 0.1)  # 偏移大的样本降权

    # 方案 B: 删除漂移特征
    drift_cols = top_adv_features(adv_model)  # 贡献最大的 N 个特征
    train = train.drop(drift_cols, axis=1)
```

**S6E6 适用性**：SDSS 原始数据 vs 竞赛数据，分布差异天然存在。

---

### 4. 🔥🔥🔥 深度堆叠（3-Layer Deep Stacking）

```
Layer 0: 50+ 单模
├── XGBoost × 5 seed × 2 特征集 = 10
├── CatBoost × 5 seed × 2 特征集 = 10
├── LightGBM × 5 seed × 2 特征集 = 10
├── RealMLP × 3 seed = 3
├── TabM × 3 seed = 3
├── TabPFN-3 × 1 = 1
└── 其他（AutoML等）= 13+
           ↓ 每模型产出 3 维 OOF

Layer 1: 族 Stacker（按架构分族，族内融合）
├── Tree Stacker (XGB+Cat+LGB 的 OOF → LogReg/XGB)
├── NN Stacker (RealMLP OOF → LogReg)
├── Attention Stacker (TabM+TabPFN OOF → LogReg)
           ↓ 每族产出 3 维

Layer 2: Super Stacker
└── 族 Stacker 输出 + raw features → TabPFN-3 / LogReg
           ↓ 最终融合
```

**S6E6 对比**：你当前是 Layer 0 (4 模 × 1 seed) + Layer 1 (5 通道 Stacker)。差 10× 的通道容量。

---

## 实施优先级

| 顺序 | 组件 | 开发成本 | 预期增益 | 适用 |
|:--:|------|:--:|:--:|------|
| 1 | 伪标签自训练 | 0.5 天 | +0.0005-0.0015 | 有强单模后 |
| 2 | 阈值优化 | 0.5 天 | +0.0003-0.0008 | 已集成后 |
| 3 | 多 seed × 多特征集 | 2 天（跑） | +0.001-0.003 | 全套 |
| 4 | 深度堆叠 3 层 | 3 天 | +0.002-0.005 | Featured |
| 5 | 对抗验证 | 1 天 | +0.0002-0.0005 | 有原始数据集时 |

---

## S6E6 可以立马跑的

### 按优先级

1. **阈值优化** — 已经拿到 v19 的 OOF，不改任何模型，纯数学搜索，30 秒跑完。直接试。

2. **伪标签 (需先跑 v24)** — v24 如果 CV > 0.9688，用它打伪标签，混入 v17/v18 训练。

3. **轻量多特征集** — 不是跑 50 个模型，是 v17 跑两份：一份 241 特征（已有），一份去掉哈希交互变 150 特征。两份 OOF 给 Stacker → 同架构伪多样性。

### 路线

```
S6E6 快速验证：
  v24 跑完 → 阈值优化 → 提交。          （0.5 天，可能涨 0.0005）
  v24 OOF + v23.1 OOF → 伪标签 → 回炉。  （1 天，可能涨 0.001）

下一场 Featured：
  全流程四件套 = 你的标准出场配置
  Layer 0: 20+ 单模 → Layer 1: 族 Stacker → Layer 2: TabPFN Super Stacker
```

---

## 关键心态

S6E6 是 Playground，目的不是冲 Top 10，是**把四层武器库都验一遍**——伪标签能不能涨、阈值优化稳不稳定、TabPFN 做 Super Stacker 行不行。验证完这些，下一场 Featured 你就是满配状态入场。

---

## 路线图

```
S6E6 (分类) ✅ 进行中
    │  验证: 树/NN/Attention 三架构 + 传统 Stacker + Foundation Model Stacker
    │
    ▼
Playground #2 (理想: 回归任务)
    │  目标: 把四件套在回归场景全部重走一遍
    │  验证: 伪标签对回归有没有效、TabPFN-3 regressor 模式
    │
    ▼
Playground #3 (回归或分类，查缺补漏)
    │  目标: 巩固薄弱项，尝试深度堆叠 3 层
    │
    ▼
Featured 🏆
    满配: 四层架构 + 全流程自动化 + 两部分类一回的经验
```
