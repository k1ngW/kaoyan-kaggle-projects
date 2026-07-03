"""
S6E6 v5 — GPU accelerated XGBoost + LightGBM
"""
import os, sys, warnings; import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
if sys.platform=="win32":os.environ["PYTHONIOENCODING"]="utf-8";sys.stdout.reconfigure(encoding="utf-8");sys.stderr.reconfigure(encoding="utf-8")
DATA_DIR=os.path.join(os.path.dirname(__file__),"data")
train=pd.read_csv(os.path.join(DATA_DIR,"train.csv")); test=pd.read_csv(os.path.join(DATA_DIR,"test.csv"))
tid,test_id=train['id'],test['id']; y=train['class']
train.drop(['id','class'],axis=1,inplace=True); test.drop(['id'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder; le=LabelEncoder(); y_enc=le.fit_transform(y)
# Features
all_data=pd.concat([train,test],sort=False).reset_index(drop=True); n_train=len(train)
for col in all_data.select_dtypes(include=['object']).columns: all_data[col]=all_data[col].astype('category').cat.codes
bands=['u','g','r','i','z']
for i in range(len(bands)):
    for j in range(i+1,len(bands)): b1,b2=bands[i],bands[j]; all_data[f'{b1}_{b2}']=all_data[b1]-all_data[b2]; all_data[f'{b1}_r_{b2}']=all_data[b1]/(all_data[b2]+0.001)
for b in bands: all_data[f'{b}_sq']=all_data[b]**2
if 'reddening' in all_data.columns:
    for b in bands: all_data[f'{b}_clean']=all_data[b]-all_data['reddening']
all_data['brightness_sum']=all_data[bands].sum(axis=1)
train_p=all_data[:n_train]; test_p=all_data[n_train:]; X,X_test=train_p.values,test_p.values
print(f"Features: {X.shape[1]}")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb, lightgbm as lgb
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

print("GPU models (n_estimators=1000)...")
models = {
    'XGB_GPU1': xgb.XGBClassifier(n_estimators=1000,learning_rate=0.03,max_depth=7,subsample=0.8,colsample_bytree=0.7,tree_method='hist',random_state=42,verbosity=0),
    'XGB_GPU2': xgb.XGBClassifier(n_estimators=1000,learning_rate=0.02,max_depth=8,subsample=0.7,colsample_bytree=0.6,tree_method='hist',random_state=43,verbosity=0),
    'LGB_GPU1': lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.03,max_depth=7,subsample=0.8,colsample_bytree=0.7,device='gpu',random_state=42,verbose=-1),
    'LGB_GPU2': lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.02,max_depth=9,subsample=0.7,colsample_bytree=0.6,device='gpu',random_state=43,verbose=-1),
}

fitted={}
for name,m in models.items():
    s=cross_val_score(m,X,y_enc,cv=cv,scoring='balanced_accuracy',n_jobs=-1)
    print(f"  {name:<12} CV: {s.mean():.5f} (+/- {s.std():.5f})")
    # Fit on full data
    m.fit(X,y_enc); fitted[name]=m

# Best 4 voting
top=sorted(fitted.keys(),key=lambda n:cross_val_score(models[n],X,y_enc,cv=cv,scoring='balanced_accuracy',n_jobs=-1).mean(),reverse=True)[:4]
print(f"\n  GPU Ensemble: {top}")
vote=VotingClassifier([(n,fitted[n]) for n in top],voting='soft')
s=cross_val_score(vote,X,y_enc,cv=cv,scoring='balanced_accuracy',n_jobs=-1)
print(f"  GPU Ensemble CV: {s.mean():.5f}")
vote.fit(X,y_enc)
preds=vote.predict(X_test)
preds_class=le.inverse_transform(preds)
pd.DataFrame({'id':test_id,'class':preds_class}).to_csv(os.path.join(os.path.dirname(__file__),"submission.csv"),index=False)
print("v5 GPU done!")
