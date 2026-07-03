"""
S6E6 v3 — Voting Ensemble (fast, no stacking CV overhead)
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
if sys.platform=="win32":os.environ["PYTHONIOENCODING"]="utf-8";sys.stdout.reconfigure(encoding="utf-8");sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR=os.path.join(os.path.dirname(__file__),"data")
train=pd.read_csv(os.path.join(DATA_DIR,"train.csv"))
test=pd.read_csv(os.path.join(DATA_DIR,"test.csv"))
tid,test_id=train['id'],test['id']; y=train['class']
train.drop(['id','class'],axis=1,inplace=True); test.drop(['id'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder(); y_enc=le.fit_transform(y)

# Features (same as v2)
all_data=pd.concat([train,test],sort=False).reset_index(drop=True); n_train=len(train)
cat_cols=all_data.select_dtypes(include=['object']).columns
for col in cat_cols: all_data[col]=all_data[col].astype('category').cat.codes
bands=['u','g','r','i','z']
for i in range(len(bands)):
    for j in range(i+1,len(bands)):
        b1,b2=bands[i],bands[j]; all_data[f'{b1}_{b2}']=all_data[b1]-all_data[b2]
for b in bands: all_data[f'{b}_sq']=all_data[b]**2
if 'reddening' in all_data.columns:
    for b in bands: all_data[f'{b}_clean']=all_data[b]-all_data['reddening']
all_data['brightness_sum']=all_data[bands].sum(axis=1)

train_p=all_data[:n_train]; test_p=all_data[n_train:]
X,X_test=train_p.values,test_p.values
print(f"Features: {X.shape[1]}")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier

cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

# Best individual models
models = {
    'XGB1': xgb.XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=7,subsample=0.8,colsample_bytree=0.7,random_state=42,verbosity=0,n_jobs=-1),
    'XGB2': xgb.XGBClassifier(n_estimators=300,learning_rate=0.04,max_depth=8,subsample=0.7,colsample_bytree=0.6,random_state=43,verbosity=0,n_jobs=-1),
    'LGB1': lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,max_depth=7,subsample=0.8,colsample_bytree=0.7,random_state=42,verbose=-1,n_jobs=-1),
    'LGB2': lgb.LGBMClassifier(n_estimators=300,learning_rate=0.04,max_depth=9,subsample=0.7,colsample_bytree=0.6,random_state=43,verbose=-1,n_jobs=-1),
    'Cat': CatBoostClassifier(iterations=300,learning_rate=0.05,depth=7,random_seed=42,verbose=0,thread_count=-1),
    'RF': RandomForestClassifier(n_estimators=150,max_depth=12,random_state=42,n_jobs=-1),
}

fitted = {}
for name, m in models.items():
    s = cross_val_score(m, X, y_enc, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    print(f"  {name:<8} CV: {s.mean():.5f} (+/- {s.std():.5f})")
    m.fit(X, y_enc); fitted[name] = m

# Voting ensemble
top = sorted(models.keys(), key=lambda k: cross_val_score(models[k],X,y_enc,cv=cv,scoring='balanced_accuracy',n_jobs=-1).mean(), reverse=True)[:4]
print(f"\n  Voting: {top}")
vote = VotingClassifier([(n, fitted[n]) for n in top], voting='soft')
vote.fit(X, y_enc)
preds = vote.predict(X_test)
preds_class = le.inverse_transform(preds)

# Compare distribution
for cls in le.classes_:
    print(f"  {cls}: {(preds_class==cls).sum()} ({(preds_class==cls).mean()*100:.1f}%)")

pd.DataFrame({'id':test_id,'class':preds_class}).to_csv(os.path.join(os.path.dirname(__file__),"submission.csv"),index=False)
print("V3 done!")
