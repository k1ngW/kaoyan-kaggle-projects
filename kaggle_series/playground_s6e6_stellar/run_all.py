"""
Kaggle Playground S6E6 — Predicting Stellar Class
===================================================
竞赛: https://www.kaggle.com/competitions/playground-series-s6e6
类型: 3分类（GALAXY / QSO / STAR）
评估: Balanced Accuracy
对标: 432统计学 — 逻辑回归 / 卡方检验 / 交叉验证 / 特征选择

运行: python run_all.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    except: pass

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SUBMISSION_PATH = os.path.join(os.path.dirname(__file__), "submission.csv")

# ═══════════════════════════════════════
# Step 1: 数据加载
# ═══════════════════════════════════════
print("=" * 60)
print("  Step 1/5: 数据加载")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"  Train: {train.shape[0]:,} x {train.shape[1]}  |  Test: {test.shape[0]:,} x {test.shape[1]}")

# 分离
train_id = train['id']; test_id = test['id']
y = train['class']
train.drop(['id', 'class'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

# 标签编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"  类别: {dict(zip(le.classes_, range(3)))}")
for cls, count in zip(le.classes_, np.bincount(y_enc)):
    print(f"    {cls}: {count:,} ({count/len(y)*100:.1f}%)")

# ═══════════════════════════════════════
# Step 2: 特征工程
# ═══════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Step 2/5: 特征工程")
print("=" * 60)

# 检查并处理类别特征
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()

print(f"  数值特征: {len(num_cols)}  |  类别特征: {len(cat_cols)}")

# 合并处理
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)
n_train = len(train)

# 类别特征编码
for col in cat_cols:
    all_data[col] = all_data[col].astype('category').cat.codes

# 新建交互特征
if len(num_cols) >= 4:
    # 颜色差值（天文常用特征）
    all_data['u_g'] = all_data.get('u', 0) - all_data.get('g', 0)
    all_data['g_r'] = all_data.get('g', 0) - all_data.get('r', 0)
    all_data['r_i'] = all_data.get('r', 0) - all_data.get('i', 0)
    all_data['i_z'] = all_data.get('i', 0) - all_data.get('z', 0)
    # 统计特征
    for prefix in ['u', 'g', 'r', 'i', 'z']:
        if prefix in all_data.columns:
            all_data[f'{prefix}_sq'] = all_data[prefix] ** 2

# 分离
train_processed = all_data[:n_train].copy()
test_processed = all_data[n_train:].copy()
print(f"  处理后特征: {train_processed.shape[1]}")

# ═══════════════════════════════════════
# Step 3: 模型训练对比
# ═══════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Step 3/5: 模型对比（5折 Stratified CV）")
print("=" * 60)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

X = train_processed.values
X_test = test_processed.values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, multi_class='multinomial', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=0, n_jobs=-1),
    'XGBoost (深度)': xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8,
                                         subsample=0.8, colsample_bytree=0.7,
                                         random_state=42, verbosity=0, n_jobs=-1),
}

results = {}
for name, model in models.items():
    if 'Logistic' in name:
        scores = cross_val_score(model, X_scaled, y_enc, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    else:
        scores = cross_val_score(model, X, y_enc, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    results[name] = (scores.mean(), scores.std())
    print(f"  {name:<25} Balanced Acc: {scores.mean():.5f} (+/- {scores.std():.5f})")

best_name = max(results, key=lambda k: results[k][0])
print(f"\n  Best: {best_name} (CV: {results[best_name][0]:.5f})")

# ═══════════════════════════════════════
# Step 4: 训练最佳模型 + 预测
# ═══════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Step 4/5: 训练最佳模型 + 生成提交")
print("=" * 60)

best_model = models[best_name]
if 'Logistic' in best_name:
    best_model.fit(X_scaled, y_enc)
    preds_enc = best_model.predict(X_test_scaled)
    preds_proba = best_model.predict_proba(X_test_scaled)
else:
    best_model.fit(X, y_enc)
    preds_enc = best_model.predict(X_test)
    preds_proba = best_model.predict_proba(X_test)

preds_class = le.inverse_transform(preds_enc)

# 生成提交文件
submission = pd.DataFrame({
    'id': test_id,
    'class': preds_class
})
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"  预测分布:")
for cls, count in zip(le.classes_, np.bincount(preds_enc)):
    print(f"    {cls}: {count:,} ({count/len(preds_enc)*100:.1f}%)")
print(f"\n  Submission: {SUBMISSION_PATH}")

# ═══════════════════════════════════════
# Step 5: 特征重要性
# ═══════════════════════════════════════
print(f"\n{'=' * 60}")
print("  Step 5/5: 特征重要性 + 要点")
print("=" * 60)

# 用 XGBoost 看特征重要性
xgb_model = models.get('XGBoost') or models.get(list(models.keys())[0])
if hasattr(xgb_model, 'feature_importances_'):
    if not hasattr(xgb_model, '_Booster'):
        xgb_model.fit(X, y_enc)
    imp = pd.DataFrame({
        'feature': train_processed.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n  Top 10 特征重要性:")
    for _, row in imp.head(10).iterrows():
        bar = '#' * int(row['importance'] * 50)
        print(f"    {row['feature']:<20} {bar} {row['importance']:.4f}")

print(f"""
  要点:
  - 3分类问题，训练集 {len(train):,} 行，类别不平衡（GALAXY 65%）
  - 用 StratifiedKFold 保持验证集中类别比例一致
  - 评估指标 Balanced Accuracy = 各类别召回率的算术平均
  - LogisticRegression 用标准化数据，树模型用原始数据
  - 特征工程关键是颜色差值（天文学上区分恒星/星系/类星体的核心特征）
""")
print("  Done! 上传 submission.csv 到 Kaggle")
