# v24 — TabPFN-3 Stacker 🔥

> **决定日期**: 2026-06-18
> **前提**: v23 系列 LogReg Stacker 受限于线性 meta-learner + 纯 OOF 输入
> **方向**: Transformer meta-learner + OOF + raw features，断代升级

---

## 断代差异

```
v23 系列 (LogReg lineage)              v24 系列 (TabPFN lineage)
─────────────────────────────          ─────────────────────────────
meta-learner: Linear + L2              meta-learner: Transformer (in-context)
输入:        纯 OOF logit              输入:        OOF logit + raw features
正则化:      显式 L2                   正则化:      内置 attention dropout
训练:        5seed×5fold CV = 25次     训练:        1次 fit
非线性:      不能                      非线性:      能学交叉条件推理
共线处理:    L2 压死                   共线处理:    attention 自然分配
```

## 核心改进

1. **输入空间炸开** — OOF logit + raw features：基模型集体犯错时能从原始特征自救
2. **Transformer meta-learner** — TabPFN-3 预训练于百万合成表格数据集
3. **零 CV 负担** — 无需 25 轮训练降方差
4. **无 L2 毒副作用** — attention 不压死共线通道

## 基模型（公共 Kaggle Notebook）

| 模型 | 来源 | Owner |
|------|------|------|
| xgb-0 | xgb-v0-for-s6e6 | cdeotte |
| xgb-1 | xgb-v1-for-s6e6 | cdeotte |
| realmlp-0 | ps-s6-e6-realmlp-pytorch | yekenot |
| realmlp-1 | realmlp-v1-for-s6e6 | cdeotte |
| tabm-0 | s6e6-tabm | donmarch14 |
| cat-0 | cat-v0-for-s6e6 | cdeotte |

共 6 模型 × 3 类 = 18 维 OOF logit + 原始特征 (~8 列) = ~26 维输入。

## 后续路线

| 版本 | 内容 | 策略 | 状态 |
|------|------|------|:--:|
| **v24** | 6 公共 OOF + raw feat → TabPFN-3 | 帖子复现 | 🔄 Kaggle 运行中 |
| **v24.1** | 你的 9 通道 + 你的 241 特征 → TabPFN-3 | 换自己的 OOF | ⬜ |
| **v24.2** | 加 cat-3 同款哈希交互特征 | 特征升级 | ⬜ |

## 运行记录

| 日期 | Kaggle ID | 描述 | CV | LB |
|------|:--:|------|------|------|
| ⬜ | ⬜ | v24 TabPFN-3 Stacker 初跑 | ⬜ | ⬜ |

## ⚠️ 可复现性注意

**TabPFN-3 模型权重必须钉死，严禁自动下载。**

```python
# ✅ 正确：从 Kaggle dataset 读取，版本钉死
os.environ["TABPFN_MODEL_CACHE_DIR"] = "/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1"

# ❌ 错误：删掉这行让 TabPFN 自动下载 → Hugging Face 最新版，可能与原帖不同
```

- 原帖 Input 挂了 `prior-labsai/tabpfn-3` dataset，copy 时必须确保该 dataset 也挂载
- 如果不挂载，TPF 会尝试 `mkdir /kaggle/input/models/` → 炸 `Read-only file system`
- 自动下载（方案 A）拉的是 Hugging Face 最新版，Prior Labs 更新权重后结果不可复现
- 所有 v24.x 后续版本统一使用同一个 TabPFN-3 dataset 版本

## 风险

- TabPFN-3 在 57 万行天文数据上行为不完全可控
- 显存 ~300 维 × 57 万行需确认 GPU 容量
- 如果结果不如 v23.1 LogReg → 这个任务 meta 层不需要非线性

## 参考

- 帖子: Stacking with TabPFN-3 (基于 Chris GPU Logistic Regression Stacker)
- TabPFN-3 Paper: https://arxiv.org/abs/2605.13986
- GitHub: https://github.com/PriorLabs/TabPFN
