# 考研复试 Kaggle 项目库

> 深圳大学 应用统计专硕(025200) · 复试项目准备
> 每个竞赛一个文件夹，所有代码和文档集中管理

## 项目列表

| 项目 | 竞赛 | 类型 | 对标432 | 状态 |
|------|------|------|---------|:--:|
| [House Prices](kaggle_series/house_prices/) | [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) | 回归分析 | 回归/变量选择/共线性 | 🟢 进行中 |
| Store Sales | [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/) | 时间序列 | ARIMA/指数平滑 | ⏳ 10月 |

## 目录结构

```
kaoyan_projects/
├── README.md
├── kaggle_series/                  # 所有竞赛项目
│   ├── kaggle入门指南.md
│   ├── 竞赛选题策略.md
│   └── house_prices/              # 房价预测
│       ├── 项目计划书.md
│       ├── run_all.py             # 一键运行
│       └── data/                  # gitignore（自行下载）
│
└── project2_store_sales/          # 10月启动
    └── 项目计划书.md
```

## 快速开始

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# 下载数据（需先配置 Kaggle API）
# https://www.kaggle.com/settings/account → Create New Token

cd kaggle_series/house_prices
python run_all.py
```

## 复试价值

- Kaggle 排名可在线验证 → 比"我做过xx项目"有说服力
- 每个项目直接对应 432 统计学考点 → 做项目 = 复习专业课
- GitHub 仓库可直接发给面试导师 → 展示工程能力和统计思维

## 参赛记录

| 日期 | 竞赛 | 分数 | 排名 |
|------|------|------|------|
| 2026-06-09 | House Prices | 0.12471 (XGBoost baseline) | — |
