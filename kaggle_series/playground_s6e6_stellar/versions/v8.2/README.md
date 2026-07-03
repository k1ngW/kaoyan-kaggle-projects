# v8.2 — XGBoost 改进版

> **决定日期**: 2026-06-15
> **前提**: v23 权重分析确认 v8 (XGBoost) 是 Stacker 最依赖的通道，但 LB 仅 0.96757
> **方向**: 基于 Chris Deotte RAPIDS XGBoost GPU v5 (370 特征) 继续加特征，目标 CV 0.969+

---

## 为什么 v8.2（跳过 v8.1）

v8.1 测试了 Chris XGBoost OvR 方案 → 发现它本质是堆叠器，纯 XGBoost 仅 CV 0.9609 → 放弃。
v8.2 回到多分类 XGBoost 路线，起点是 Chris 更新的 RAPIDS v5 (370 特征, CV 0.96770)。

---

## baseline 对比

| | 旧 v8 | **v8.2 baseline (Chris v5)** | 目标 |
|------|:--:|:--:|:--:|
| 特征数 | 240 | 370 | — |
| CV | 0.96665 | **0.96770** | 0.969+ |
| LB | 0.96757 | **0.96801** | 0.969+ |

差距 CV -0.00117 / LB -0.00177。

---

## 特征差异分析（待 notebook 下载后完成）

vs STRATEGY.md v17 特征清单：

| 特征类别 | v17 有 | v8.2 baseline | 状态 |
|------|:--:|:--:|:--:|
| 颜色差 (10 对) | ✅ | ⬜ | 待查 |
| mag_stats (mean/std/min/max/range) | ✅ | ⬜ | 待查 |
| mag_slope + curvature | ✅ | ⬜ | 待查 |
| 红移变换 (sq/cbrt/lt_002/gt_07) | ✅ | ⬜ | 待查 |
| 波段/红移 比值交互 | ✅ | ⬜ | 待查 |
| 通量 flux_* + log_flux + stats | ✅ | ⬜ | 待查 |
| 天空坐标 sky_x/y/z | ✅ | ⬜ | 待查 |

---

## 运行记录

| 日期 | 版本 | 描述 | CV | LB |
|------|------|------|------|------|
| 2026-06-15 | v8.2-baseline | Chris RAPIDS XGBoost GPU v5 (370 feat) | **0.96770** | **0.96801** |
| 2026-06-15 | v8.2-v1 | +10特征 + TOP_N=380 | **0.96767** | 0.96801 |

**结论：** XGBoost 单模特征天花板已到。370→380 特征不涨反平。v8.2 CV 0.96767 作为单模终版，封档。

---

## Kaggle Dataset

| Dataset 名 | 内容 |
|------|------|
| `s6e6-v8-2-oof` (weywuy) | `oof.npy` + `test_preds.npy` |

### v23.1 路径引用

```python
'v8-2': f'{OOF_BASE}/s6e6-v8-2-oof',    # v8.2 XGBoost (CV 0.96767)
```

---

## 文件

```
versions/v8.2/
├── README.md
├── s6e6-xgb-v8-2-baseline.ipynb       ✅
├── s6e6-xgb-v8-2-v1.ipynb             ⬜ 从 Kaggle 下载
├── submission_v8-2_baseline.csv        ⬜
├── submission_v8-2_v1.csv              ⬜
├── oof_v8-2_v1.npy                     ⬜
└── test_v8-2_v1.npy                    ⬜
```

## 状态

- ✅ v8.2 封版，转入 v23.1 多 seed 堆叠
