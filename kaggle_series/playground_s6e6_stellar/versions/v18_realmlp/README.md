# v18 — RealMLP-5 单模型
#
# 来源: Chris Deotte "RealMLP-5 for Playground Series S6E6"
# 架构: PyTorch tabular neural network, n_ens=8
# 特征: color diffs + redshift ratios + fold-safe TE
# 预期 CV: ~0.96928
#
# 产出来自 Kaggle notebook, 跑完后下载到本目录:
#   oof.npy         — OOF 预测 (577347 × 3)
#   test_preds.npy  — 测试集预测 (247435 × 3)
#   submission.csv  — 提交文件
#   log.json        — 实验记录 (CV + LB + 参数)
