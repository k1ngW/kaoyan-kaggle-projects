# House Prices — Advanced Regression Techniques

> Kaggle Getting Started · 永久开放 · 5,900+ 队伍

## 竞赛信息

- 任务：79 个特征预测房价 SalePrice
- 评估：RMSE（对数空间）
- 数据：1,460 训练 / 1,459 测试
- 链接：https://www.kaggle.com/c/house-prices-advanced-regression-techniques

## 方法

1. **EDA**：偏度分析 → Target log 变换（偏度 1.88 → 0.12）
2. **特征工程**：缺失值填充 + One-hot 编码 + 新建特征（TotalSF/TotalBath/HouseAge）
3. **模型对比**：Ridge / Lasso / ElasticNet / XGBoost，5 折 CV
4. **模型诊断**：VIF 共线性检验 / 残差分析 / 特征重要性

## 结果

| 模型 | CV RMSE |
|------|:------:|
| Ridge (alpha=10) | 0.1482 |
| Lasso (alpha=0.001) | 0.1491 |
| XGBoost | **0.1288** |

**Kaggle Public Score**: 0.12471

## 运行

```bash
pip install pandas numpy scikit-learn xgboost
# 下载 train.csv test.csv → data/
python run_all.py
```
