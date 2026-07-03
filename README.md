# Kaggle Projects

> Kaggle 竞赛项目 & 机器学习知识库 | [Kaggle 入门指南](kaggle_series/kaggle入门指南.md)

## 项目

| # | 项目 | 竞赛 | 类型 | 指标 | 语言 | 排名 |
|:--:|------|------|------|------|:--:|:--:|
| 1 | [S6E6 Stellar Classification](kaggle_series/playground_s6e6_stellar/) | Playground S6E6 | 3分类 | Balanced Accuracy | Python | **Top 8.34%** (#235/2817) |
| 2 | [S6E7 Student Health Risk](kaggle_series/playground_s6e7_health/) | Playground S6E7 | 分类 | Balanced Accuracy | Python | 🟢 进行中 |
| — | [House Prices](kaggle_series/house_prices/) | Getting Started | 回归 | RMSLE | Python | 已放弃 |

## 目录结构

```
├── kaggle_series/
│   ├── playground_s6e6_stellar/     # S6E6: 3分类 — Top 8.34%
│   │   ├── optimize_v*.py           # 多版本迭代 (v1→v24)
│   │   ├── ensemble_cells.py        # 跨架构集成
│   │   ├── experiment_logger.py     # 实验日志体系
│   │   ├── STRATEGY.md
│   │   └── versions/                # 各版本 Notebook 归档
│   │
│   ├── playground_s6e7_health/      # S6E7: 分类 — 进行中
│   │
│   ├── knowledge/                   # 共享知识库
│   │   ├── 机器学习基础概念.md
│   │   ├── 模型原理与优化思路详解.md
│   │   ├── 集成学习与Bias-Variance.md
│   │   ├── 数据层/特征层/模型层/后处理层优化方案.md
│   │   ├── Public LB陷阱与最终选择策略.md
│   │   ├── Chris-Deotte-3层Stacking冠军方案.md
│   │   └── KGMON-Playbook-冠军方案参考.md
│   │
│   ├── kaggle入门指南.md
│   └── 代码模板/
│
├── sql_practice/                    # SQL 练习
├── data_collection/                 # 数据采集示例
└── reports/                         # 项目报告
```

## 参赛记录

| 日期 | 竞赛 | Public | Private | 排名 |
|------|------|:--:|:--:|:--:|
| 2026-06-30 | S6E6 Stellar Classification | 0.97054 | **0.97009** | **#235/2817 (Top 8.34%)** |

## 技术栈

```
Python: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, matplotlib, seaborn
R:     tidyverse, glmnet, xgboost, caret, rmarkdown
```

## 快速开始

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn

# 下载数据（需配置 Kaggle API）
cd kaggle_series/playground_s6e6_stellar
python run_all.py
```
