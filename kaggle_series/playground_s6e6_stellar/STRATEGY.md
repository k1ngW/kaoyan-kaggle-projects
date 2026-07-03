# S6E6 提分策略 & 版本分析

## 当前状态

| 版本 | 模型 | OOF | LB | 排名 |
|------|------|-----|-----|------|
| **v17-cat** | CatBoost GPU | 0.96887 | 0.96978 | 树 |
| **v18-realmlp** | RealMLP-5 NN | **0.96904** | **0.96979** | 神经网络 |
| v8 | XGBoost CPU | 0.9666 | 0.96757 | 树 |

**差距**: v8 距离比赛顶级单模型 (~0.969) 差 ~0.0013

---

## v8 vs CatBoost cat-3 本质差异

```
v8 (XGBoost CPU):                    cat-3 (CatBoost GPU):
─────────────────                    ────────────────────
手动 Target Encoding (171列)         原生 CTR (max_ctr_complexity=3)
1个字符串交互                         几百个哈希 PAIR/TRIO
class_weights 自动                    class_weights=[1, 3.25, 5]
无原始SDSS混入                        原始SDSS weight=0.06
仅171精选特征                         241特征 + 153类别
-------------------------------------------------------------------
OOF 0.9666                           OOF 0.96892 (+0.0023)
```

---

## cat-3 特征工程详解

### 1. 颜色差 + 星等统计 (20+7=27个数值特征)
- `u-g`, `g-r`, `r-i`, `i-z` 等 10 对颜色差 + 绝对值
- `mag_mean/std/min/max/range` + `mag_argmin/max`（最亮/最暗波段）

### 2. 光谱物理特征 (5个)
- `mag_slope`: 5波段线性斜率
- `mag_curvature` = u - 2r + z (标准天文曲率)
- `blue_curvature` = u - 2g + r
- `red_curvature` = r - 2i + z
- `color_temp_gr_proxy` = 4600×((1/(0.92gr+1.7))+(1/(0.92gr+0.62))) (色温代理)

### 3. 红移变换 (7个)
`redshift_abs/log1p_abs/sq/cbrt` + `redshift_is_neg/lt_002/gt_07` (布尔)

### 4. 比值交互 (10个)
`g/i/z_over_redshift`, `z_over_g`, `z2_over_g2`, `log_z_over_log_g`, `sqrt_z_over_sqrt_g`, `redshift_x_{u,g,r,i,z}`

### 5. 天空坐标 (7个)
`alpha_sin/cos`, `delta_sin/cos`, `sky_x/y/z` (球面→直角坐标)

### 6. 通量 (16个)
`flux_{u,g,r,i,z}`, `log_flux_{...}`, `flux_mean/std/range`, `flux_ratio_{...}`

### 7. 🔥 分箱编码 → CatBoost 类别特征 (120+个)
这是 v8 完全缺失的核心技巧：

| 类别家族 | 例子 | 捕获什么 |
|---------|------|---------|
| `q32/q100/q500_cat` | `redshift_q100_cat` | 值的分位相对位置 |
| `floor_cat` | `u_floor_cat` | 整数部分 (= 粗粒度分组) |
| `roundN_cat` | `u_round2_cat` | 固定精度分组 |
| `mod10/mod100` | `redshift_mod10_cat` | 数字尾数模式 |
| `frac20` | `redshift_frac20_cat` | 小数部分分桶 |
| `decimal1000` | `redshift_decimal1000_cat` | 细粒度小数编码 |

**为什么有效**: CatBoost 的 `max_ctr_complexity=3` 自动学习这些类别与 target 的三阶统计关系，完全替代手动 Target Encoding。

### 8. 🔥🔥 哈希交互 (几百个)
```python
hash(a, b) = ((a+1000003) * 1000003 + (b+9176)) % 2147483647
```
- `COMBO_*`: 9 个手动选择的组合对
- `PAIR_*`: C(10,2)=45 个自动两两组合
- `TRIO_*`: 3 个三特征组合

**为什么用哈希**: 字符串拼接 → 百万级类别 → 内存爆炸。哈希 → int32 → 紧凑 + CatBoost 极快处理。

---

## 后续提分方向

### 方向 A: 特征层
- [x] 全颜色差 + 绝对值 (cat-3 已有)
- [x] sky_bin (帖子 #705303)  → cat-3 用 `alpha_floor_cat × delta_floor_cat` 哈希替代
- [ ] 天文物理公式特征 (更多色温/消光/金属丰度代理)
- [ ] 异常检测分数作为特征 (Isolation Forest + OOF)

### 方向 B: 模型层
- [x] CatBoost 单模 (cat-3, CV 0.96887 → LB 0.96978)
- [x] RealMLP-5 单模 (NN n_ens=8, CV 0.96904 → LB 0.96979)
- [ ] XGBoost GPU + 同样 241 特征 → 对照实验
- [x] TabM (kirill0212, CV 0.96862) — attention 架构，第三种互补
- [x] TabPFN-3 堆叠器 (🔥 v24) — Transformer meta-learner 替代 LogReg，+raw features

### 方向 C: 训练层
- [x] 原始 SDSS 混入 weight=0.06 (cat-3 已用)
- [ ] 伪标签: 高置信度 test 样本 → 加入训练
- [ ] 对抗验证: 检查竞赛数据 vs 原始 SDSS 分布偏移

### 方向 D: 后处理 & 集成
- [x] v17+v8 软投票 → ∆~0（同架构无意义）
- [x] v17+v18+v8 三模跨架构集成 → **LB 0.97054** (+0.00076)
- [ ] v17+v18+v8+TabM 四模集成 → 目标 0.971
- [ ] 阈值调优 (v16 已有 scipy.optimize 方案)

---

## 版本迭代记录

| # | 来源 | 核心改进 | OOF | LB |
|---|------|---------|-----|-----|
| **v24** 🔥 | TabPFN-3 Stacker | 6公共OOF+raw feat→TabPFN(Transformer meta) | ⬜ | ⬜ |
| **v19-ensemble** 🏆 | 三模集成 | realmlp×0.47+cat×0.48+v8×0.05 | 0.96997 | **0.97054** |
| v18-realmlp | RealMLP-5 单模 | 神经网络 n_ens=8 | 0.96904 | 0.96979 |
| v17-cat | CatBoost 单模 | 241feat+原生CTR+哈希交互 | 0.96887 | 0.96978 |
| v8 | XGBoost 单模 | 171 TE + XGB lossguide | 0.9666 | 0.96757 |

**关键发现**: 树+树融合 (v17+v8) = 几乎零提升。树+神经网络 (v17+v18) = +0.00076 LB。架构互补才是核心。

---

## 2026-06-15 更新：v8.2 & v23 经验

### v8.2 特征迭代方法论

| 版本 | 起点 | 特征 | CV | LB |
|------|------|:--:|------|------|
| 旧 v8 | Chris XGBoost v1 | 240 选中 / ~400 工程 | 0.96665 | 0.96757 |
| **v8.2 baseline** | Chris RAPIDS v5 (style="full") | 370 / ~500 | **0.96770** | **0.96801** |
| v8.2-v1 (计划) | +10 缺失特征 + TOP_N 380 | 380 / ~510 | 目标 0.968+ | ⬜ |

**核心发现**：
- style="full" 多出来的 130 工程特征（sky/absmag/color_plane/phys_bin 等）贡献了 +0.001 CV
- XGBoost 参数除了 max_bin 512→960，其余完全一致 → 参数调优空间几乎为零
- 特征迭代 = 池扩 → 自动 gain 排序 → 选 TOP_N，每次弱特征自然被挤出
- v8.1 放弃：Chris OvR notebook 本质是堆叠器（XGBoost + 外部 OOF → LogReg），纯 XGBoost 仅 CV 0.9609

### v23 LogReg Stacker 实战教训

- class_weights 在 logit 空间是毒药 → CV 0.954 vs 0.967
- L2 自动压共线特征 → 需要多架构多样性
- Cell 4 权重分析直接指导迭代方向（补谁的 seed、退役谁）
- 5 通道太少 → 9 通道才能超单模天花板
- OvR Stacker 作为 v23.2 备选方案

### 当前版本全景

| 版本 | 模型 | CV | LB | 状态 |
|------|------|------|------|:--:|
| v18 | RealMLP-5 | 0.96904 | 0.96979 | 单模天花板 |
| v17 | CatBoost cat-3 | 0.96887 | 0.96978 | |
| v20 | TabM | 0.96862 | 0.96895 | |
| **v8.2** | XGBoost v5 | **0.96770** | **0.96801** | 封版 |
| v19 | Grid Search 3模 | 0.96997 | **0.97054** | 融合天花板 |
| v23 | LogReg Stacker 5通 | 0.96713 | 0.96809 | ⬅ 待扩通道 |
| **v24** 🔥 | TabPFN-3 Stacker | ⬜ | 0.97038 | 封版，未超 v19 (0.97054) |
| **v25** 🔥🔥 | 收尾冲刺五件套 | — | — | **方案已定，v25.1 随时可跑** |

---

### v24 TabPFN-3 Stacker 🔥🔥🔥

> **决定日期**: 2026-06-18
> **前提**: v23 系列 LogReg Stacker 受限于线性 meta-learner + 纯 OOF 输入
> **方向**: 替换 meta-learner 为 TabPFN-3 Transformer，并拼入原始特征

#### 与 v23 系列的断代差异

```
v23 系列 (LogReg lineage)              v24 系列 (TabPFN lineage)
─────────────────────────────          ─────────────────────────────
meta-learner: Linear + L2              meta-learner: Transformer (in-context)
输入:        纯 OOF logit              输入:        OOF logit + raw features
正则化:      显式 L2                   正则化:      内置 attention dropout
训练:        5seed×5fold CV = 25次     训练:        1次 fit，无 CV 循环
非线性:      不能                      非线性:      能学交叉条件推理
共线处理:    L2 压死 (如 v17 vs v22a)  共线处理:    attention 自然分配
```

#### 核心改进

1. **输入空间炸开** — 不只喂"别人怎么判断"，也喂"数据本身长什么样"
   ```python
   X_train = np.concatenate([6个模型的 OOF logit, raw features], axis=1)
   ```
   基模型集体犯错时，TabPFN 能从原始特征自救。

2. **Transformer meta-learner** — TabPFN-3 在百万合成表格数据集上预训练，支持 100 万行 in-context learning。非线性交叉条件推理能力远超 Linear 层。

3. **零 CV 负担** — 无需 5seed×5fold 的 25 轮训练来降方差，一次 fit 出结果。

4. **无 L2 毒副作用** — L2 在 LogReg 里随机压死共线通道。TabPFN attention 自然处理高度相关输入。

#### 基模型来源（公共 Kaggle Notebook）

| 模型 | 来源 | Owner |
|------|------|------|
| xgb-0 | xgb-v0-for-s6e6 | cdeotte |
| xgb-1 | xgb-v1-for-s6e6 | cdeotte |
| realmlp-0 | ps-s6-e6-realmlp-pytorch | yekenot |
| realmlp-1 | realmlp-v1-for-s6e6 | cdeotte |
| tabm-0 | s6e6-tabm | donmarch14 |
| cat-0 | cat-v0-for-s6e6 | cdeotte |

共 6 模型 × 3 类 = 18 维 OOF logit + 原始特征列。

#### 后续路线

| 版本 | 内容 | 策略 | 状态 |
|------|------|------|:--:|
| **v24** | 6 公共 OOF + raw feat → TabPFN-3 | 帖子复现，验证范式可行性 | 🔄 运行中 |
| **v24.1** | 你的 9 通道 + 你的 241 特征 → TabPFN-3 | 用自己的 OOF 替换公共 OOF | 计划 |
| **v24.2** | 加 cat-3 同款哈希交互特征 → TabPFN-3 | 特征升级 | 计划 |

#### 风险

- TabPFN-3 在 57 万行上的行为不完全可控（pretrained prior 可能不适合天文数据）
- 57 万 × ~300 维的显存占用需确认 GPU 是否 hold 住
- 如果结果不如 v23.1 LogReg，说明这个任务 meta 层不需要非线性

#### 参考

- 帖子: Stacking with TabPFN-3 (基于 Chris GPU Logistic Regression Stacker)
- TabPFN-3 Technical Report: https://arxiv.org/abs/2605.13986
- GitHub: https://github.com/PriorLabs/TabPFN

| # | 标题 | 关键信息 |
|---|------|---------|
| — | Stacking with TabPFN-3 | **v24** 基座: 6公共OOF+raw feat→TabPFN-3 meta-learner |
| 704527 | Single Model or Ensemble? | 集成 0.97028, 单模 0.96862-0.96904 |
| 704885 | Binary Classification Chain | GALAXY vs Rest → QSO vs STAR |
| 705303 | Sky Positions Features | sin/cos + sky_bin |
| 703686 | TabPFN-3 baseline | LB 0.964 (单模分类，非堆叠) |

---

## 🏆 最终结果（2026-07-01）

| 指标 | 数值 |
|------|------|
| **Competition** | Playground Series S6E6 |
| **截止** | 2026-06-30 |
| **排名** | **532 / 2812 (Top 18.9%)** |
| **最佳分数** | **0.97054** (v19 三模 Grid Search 集成) |
| **最佳单模** | v18 RealMLP-5 (0.96979) |
| **v25 五件套** | 未跑 — 竞赛结束，方案已归档到 STRATEGY.md，下场比赛复用 |

### v25 未跑原因

竞赛结束后再提交不计入排名。v25 四件套（阈值优化/伪标签/Stacker 融合/二进制链/OOF 共识度）方案完整保留，S7/S8 Playground 上验证。

