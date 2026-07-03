# v20 — TabM One-vs-Rest

## 状态

| 项目 | 值 |
|------|------|
| Notebook | ps6e6-one-vs-rest-tabm.ipynb（kirill0212 方案复现） |
| 环境 | Kaggle Notebook, GPU T4×2 |
| TabM CV (raw) | 0.961011 |
| TabM CV (tuned) | 0.968617（+0.0076 bias调优后） |
| 三类召回 | GALAXY 0.9574 / QSO 0.9772 / STAR 0.9712 |
| 外部 OOF 融合 | ❌ 未成功 — STACKING_FILES 路径多了 datasets/weywuy/ |
| 四模简单融合 CV | 0.969505（TabM 用 raw 概率拖了后腿） |

## 踩坑

1. Kaggle 数据集路径是 `/kaggle/input/datasets/weywuy/s6e6-xxx-oof/`，不是 `/kaggle/input/s6e6-xxx-oof/`
2. Cell 14 改 STACKING_FILES 后 Cell 28-33 不会自动重跑，`add_oof` 仍为空
3. 最终导出 Cell 用 `np.load()` 直接读文件绕过了 add_oof，但此时 run_blend() 已跑完
4. 导出的 oof.npy 是 raw 版（0.961），bias 将在 v21 Stacking 时自动学习

## 后续

- [ ] 提交 TabM 单模看 LB
- [ ] 下载 oof.npy + test_preds.npy + experiment_log 到本地
- [ ] v21 后处理：每类独立 Stacking 融合四模型

## 文件清单

Kaggle Output 需下载的文件：

| 文件 | 说明 | 必要 |
|------|------|:--:|
| oof.npy | TabM raw OOF 概率 (577347, 3) | ✅ |
| test_preds.npy | TabM raw Test 概率 (247435, 3) | ✅ |
| experiment_log_v20_tabm.json | 实验日志 | ✅ |
| submission.csv | notebook 提交文件（TabM tuned 单模） | ✅ 看 LB |
