# v22 — CatBoost OOF-Safe Encoding（A/B/C 三版本对照实验）

## 决策变更

| 原计划 | 实际情况 | 决策 |
|------|------|------|
| LGBM 单模 | Kaggle LGBM 4.6.0 未编译 CUDA，GPU 不可用 | **换 CatBoost** |
| 354 特征体系 | 跑起来实际 ~304（v22a）/ ~360（v22b） | 不硬凑数量，由版本参数化决定 |

## 技术方案

### 核心思路
用 OOF-safe 频率编码 + 目标编码替代 CatBoost 原生 CTR3 + 哈希交互，
创造和 v17 不同的特征体系 → 预测差异 → 集成多样性。

### 三个版本

| | v22a（保守） | v22b（激进） | v22c（回归 v17 写法） |
|---|---|---|---|
| 静态数值 | 104 | 130 | 130（同 b） |
| 静态类别 | 40 | 50 | 50（同 b） |
| OOF 编码列 | 40 | 70+ | 70+ |
| 哈希交互 | ❌ | ❌ | ✅ 105 PAIR + 3 COMBO |
| CTR3 | ❌ | ❌ | ✅ |
| depth/lr | 7/0.02 | 7/0.02 | 8/0.042 |
| l2_leaf_reg | 3.0 | 3.0 | 8.0 |
| random_strength | 0.5 | 0.5 | 1.2 |

## 实验日志

### 2026-06-14 最终运行结果

```
v22a: fold BA = [0.96462, 0.96432, 0.96308, 0.96356, 0.96413]
      OOF CV = 0.96394  |  feats=304  |  CM: GALAXY→STAR 13056 次误判

v22b: fold BA = [0.96385, 0.96317, 0.96313, 0.96315, 0.96323]
      OOF CV = 0.96331  |  feats=425  |  更多特征 → 更差

v22c: fold BA = [0.95816, 0.95897, 0.95770, 0.95755, 0.95867]
      OOF CV = 0.95821  |  feats=965  |  弱底子 + CTR3 + hash = 负优化
```

**赢家 v22a，CV 0.96394，比 v17 低 0.00493。**

### 三条铁律

1. **v17 体系不可拆改。** 153cat + CTR3 + hash + SDSS + depth=8 是联动体系。把 cat 从 153 砍到 50，CTR3 和 hash 失去基数——960+ 特征全是噪音，CV 反而最低。

2. **特征数量和 CV 负相关。** 304 → 425 → 965，CV 0.96394 → 0.96331 → 0.95821。不是"多一定好"，底子垮了堆再多没用。

3. **单模天花板已被证实。** 8 个原始波段的信号基础，v17 241 特征 + CTR3 体系已是最优解。继续在特征层/模型层迭代的边际收益为零。

### v22a OOF 是否保留进 v23？

保留。CV 0.96394 虽然弱，但从混淆矩阵看**错误模式不同**——GALAXY→STAR 13056 次误判（v17 只有 9581 次）。Stacker 可能在某些类别上给它非零权重。

## 产出

- [x] v22a_oof.npy / v22a_test_preds.npy（CV 0.96394）→ 上传为 `s6e6-v22a-cat-oof`
- [x] v22b_oof.npy / v22b_test_preds.npy（CV 0.96331，废案存档）
- [x] v22c_oof.npy / v22c_test_preds.npy（CV 0.95821，废案存档）
- [x] 实验结论：v17 体系不可拆改，特征层/模型层天花板已到
- [x] 战略转向：v23 终结 Level 1 混合，升级 Chris LogReg Stacker

## 踩坑

1. Kaggle LGBM 4.6.0 `device='gpu'` 实际跑 CPU，`device='cuda'` 报 CUDA 未编译
2. CatBoost `task_type='GPU'` 正常，T4×2 五折约 10-15min
3. **目录未创建导致第一个版本保存失败**：`train_oof/` / `test_preds/` / `subs/` 需要 `mkdir` 再 `np.save`
4. Notebook 内 A/B/C 连续跑若不提前建目录，前一个版本白跑（OOF 在内存但写不进去）
5. **OOF-safe 编码在 fold 内做，每个 fold 都要对 encode_cols 重新计算 freq/TE → 5-fold 内存开销翻倍**

---

## ★ 战略转折：v22 → v23

### 决定

**v23 终结传统 CSV 混合方案，转向 Chris Deotte GPU LogReg Stacker。**

### 为什么

v19 / v21 反复验证：Grid Search 死权重、每类独立权重、阈值调优、XGB Stacking——所有 Level 1 方案的天花板就是 CV 0.96996 / LB 0.97050。

v22 单模实验确认：这个数据的单模天花板（~0.969）已经被 v17/v18 摸到，继续在特征层和模型层迭代的边际收益 < 0.0005。

**唯一剩下的有效增量在融合层**：Chris Stacker 的 logit 变换 + L2 正则 + 5seed×5fold 自动学到每模型×每类的动态权重，这是 Grid Search 做不到的。

### v23 计划

```
输入: v8 + v17 + v18 + v20 + (v22 赢家) → 5 模 OOF
方法: prob→logit → PyTorch LogReg + L2 → 5seed × 5fold
目标: CV > 0.9703 / LB > 0.9710

参考: Chris Deotte GPU LogReg Stacker (ref_notebooks/)
      知识库/后处理方案对比.md — Chris Stacker 的 5 个要点
```

### 传统混合方案的终结声明

> Level 1 方法（Grid Search / 阈值调优 / 12 权重 / CSV 混合）在 S6E6 上已穷尽。
> 从 v23 开始，只保留 Grid Search 作为基线对照，不再作为主方案提交。
> 后处理层升级为 Level 2 活规则（LogReg/Ridge Stacking）+ 多 seed 平均。
