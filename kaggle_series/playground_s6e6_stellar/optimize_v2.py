"""
S6E6 Optimization v2 — 特征增强 + LightGBM + Ensemble
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
OUT = os.path.join(os.path.dirname(__file__), "submission.csv")

# Load
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_id, test_id = train['id'], test['id']
y = train['class']
train.drop(['id', 'class'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ── Enhanced Feature Engineering ──
all_data = pd.concat([train, test], sort=False).reset_index(drop=True)
n_train = len(train)

cat_cols = all_data.select_dtypes(include=['object']).columns
for col in cat_cols:
    all_data[col] = all_data[col].astype('category').cat.codes

bands = ['u', 'g', 'r', 'i', 'z']
# 颜色差值
for i in range(len(bands)):
    for j in range(i+1, len(bands)):
        b1, b2 = bands[i], bands[j]
        all_data[f'{b1}_{b2}'] = all_data[b1] - all_data[b2]

# 比率特征
for i in range(len(bands)):
    for j in range(i+1, len(bands)):
        b1, b2 = bands[i], bands[j]
        all_data[f'{b1}_div_{b2}'] = all_data[b1] / (all_data[b2] + 0.001)

# 多项式 (关键波段)
for b in bands:
    all_data[f'{b}_sq'] = all_data[b] ** 2

# 红化相关
if 'reddening' in all_data.columns:
    for b in bands:
        all_data[f'{b}_dered'] = all_data[b] - all_data['reddening']

train_p = all_data[:n_train].copy()
test_p = all_data[n_train:].copy()

X, X_test = train_p.values, test_p.values
print(f"Features: {X.shape[1]}")

# ── Models ──
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

scaler = StandardScaler()
X_s = scaler.fit_transform(X)
X_test_s = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'XGB': xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7,
                              subsample=0.8, colsample_bytree=0.7,
                              random_state=42, verbosity=0, n_jobs=-1),
    'XGB_v2': xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=8,
                                 subsample=0.8, colsample_bytree=0.6,
                                 random_state=43, verbosity=0, n_jobs=-1),
    'LGBM': lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=7,
                                subsample=0.8, colsample_bytree=0.7,
                                random_state=42, verbose=-1, n_jobs=-1),
    'LGBM_v2': lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=9,
                                   subsample=0.8, colsample_bytree=0.6,
                                   random_state=43, verbose=-1, n_jobs=-1),
    'RF': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
}

results = {}
fitted = {}
for name, model in models.items():
    use_scaled = 'RF' not in name and 'XGB' not in name and 'LGBM' not in name
    scores = cross_val_score(model, X_s if use_scaled else X, y_enc,
                              cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    results[name] = (scores.mean(), scores.std())
    print(f"  {name:<12} CV: {scores.mean():.5f} (+/- {scores.std():.5f})")
    model.fit(X_s if use_scaled else X, y_enc)
    fitted[name] = model

best_name = max(results, key=lambda k: results[k][0])
best_score = results[best_name][0]
print(f"\n  Best single: {best_name} ({best_score:.5f})")

# ── Ensemble: Soft Voting ──
print("\n--- Ensemble ---")
top_models = sorted(results, key=lambda k: results[k][0], reverse=True)[:3]
print(f"  Using: {top_models}")

estimators = [(name, fitted[name]) for name in top_models]
ensemble = VotingClassifier(estimators, voting='soft')
ensemble.fit(X, y_enc)
preds_enc = ensemble.predict(X_test)
# Also compare single best
best_preds = fitted[best_name].predict(X_test)

# ── Check if ensemble prediction is different from best single ──
diff_pct = (preds_enc != best_preds).mean()
print(f"  Ensemble differs from {best_name} on {diff_pct*100:.1f}% of test")

# Use ensemble
preds_class = le.inverse_transform(preds_enc)

pd.DataFrame({'id': test_id, 'class': preds_class}).to_csv(OUT, index=False)

print(f"\n  Distribution: {dict(zip(*np.unique(preds_class, return_counts=True)))}")
print(f"  Done! v2 ready. Score improvement: {best_score - 0.95311:+.5f} vs v1 CV")
