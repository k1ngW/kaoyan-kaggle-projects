"""
S6E6 v6 — TabPFN-3 + 公式特征 + 最佳 ensemble
关键发现（来自Discussion）:
  1. spectral_type = cut(r-g, [-inf,-1,-0.5,0,inf]) → 用原始r-g值代替编码
  2. galaxy_population = cut(u-r, [-inf,2.2,inf]) → 用原始u-r值代替编码
  3. TabPFN-3 单模型可达 0.964 LB
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

from sklearn.preprocessing import LabelEncoder; le=LabelEncoder(); y_enc=le.fit_transform(y)

# ── 公式特征工程 ──
# 根据Discussion: spectral_type = cut(r-g), galaxy_population = cut(u-r)
# 所以不要用编码后的类别，直接用原始的 color 差值
all_data=pd.concat([train,test],sort=False).reset_index(drop=True); n_train=len(train)

# 光谱类型相关的核心特征 (r-g)
all_data['r_minus_g'] = all_data['r'] - all_data['g']
all_data['r_div_g'] = all_data['r'] / (all_data['g'] + 0.001)

# 星系群体相关的核心特征 (u-r)
all_data['u_minus_r'] = all_data['u'] - all_data['r']
all_data['u_div_r'] = all_data['u'] / (all_data['r'] + 0.001)

# 保留类别编码作为辅助（但不依赖它们）
for col in all_data.select_dtypes(include=['object']).columns:
    all_data[col] = all_data[col].astype('category').cat.codes

# 颜色差特征
bands=['u','g','r','i','z']
for i in range(len(bands)):
    for j in range(i+1,len(bands)):
        b1,b2=bands[i],bands[j]; all_data[f'{b1}_{b2}']=all_data[b1]-all_data[b2]

# 天球坐标特征 (Discussion Sky Positions)
if 'alpha' in all_data.columns and 'delta' in all_data.columns:
    alpha_rad = np.deg2rad(all_data['alpha'])
    delta_rad = np.deg2rad(all_data['delta'])
    all_data['sin_delta'] = np.sin(delta_rad)
    all_data['cos_delta'] = np.cos(delta_rad)
    all_data['sin_alpha'] = np.sin(alpha_rad)
    all_data['cos_alpha'] = np.cos(alpha_rad)

for b in bands: all_data[f'{b}_sq']=all_data[b]**2
if 'reddening' in all_data.columns:
    for b in bands: all_data[f'{b}_clean']=all_data[b]-all_data['reddening']
all_data['brightness_sum']=all_data[bands].sum(axis=1)

train_p=all_data[:n_train]; test_p=all_data[n_train:]
X,X_test=train_p.values,test_p.values
print(f"Features: {X.shape[1]}")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, lightgbm as lgb

scaler=StandardScaler(); X_s=scaler.fit_transform(X); X_test_s=scaler.transform(X_test)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

# ── TabPFN-3 ──
from tabpfn import TabPFNClassifier
tabpfn = TabPFNClassifier(device='cpu', random_state=42)
print("TabPFN loaded. Training...")
tabpfn.fit(X_s, y_enc)
tabpfn_preds = tabpfn.predict(X_test_s)
tabpfn_scores = cross_val_score(tabpfn, X_s, y_enc, cv=cv, scoring='balanced_accuracy', n_jobs=1)
print(f"  TabPFN-3     CV: {tabpfn_scores.mean():.5f}")

# ── GPU Ensemble (with more trees now) ──
models = {
    'XGB1': xgb.XGBClassifier(n_estimators=2000,learning_rate=0.02,max_depth=7,subsample=0.7,colsample_bytree=0.6,tree_method='hist',random_state=42,verbosity=0),
    'XGB2': xgb.XGBClassifier(n_estimators=2000,learning_rate=0.02,max_depth=8,subsample=0.6,colsample_bytree=0.5,tree_method='hist',random_state=43,verbosity=0),
    'LGB1': lgb.LGBMClassifier(n_estimators=2000,learning_rate=0.02,max_depth=7,subsample=0.7,colsample_bytree=0.6,device='gpu',random_state=42,verbose=-1),
    'LGB2': lgb.LGBMClassifier(n_estimators=2000,learning_rate=0.02,max_depth=9,subsample=0.6,colsample_bytree=0.5,device='gpu',random_state=43,verbose=-1),
}

fitted={}
for name,m in models.items():
    s=cross_val_score(m,X,y_enc,cv=cv,scoring='balanced_accuracy',n_jobs=-1)
    print(f"  {name:<8} CV: {s.mean():.5f}")
    m.fit(X,y_enc); fitted[name]=m

# ── TabPFN + XGB/LGB Ensemble ──
# 用软投票，TabPFN 权重翻倍（因为它单模型就很强）
print("\n  Building TabPFN ensemble...")
all_models = [('tabpfn', tabpfn)] + [(n,fitted[n]) for n in ['LGB1','XGB1','LGB2','XGB2']]
vote = VotingClassifier(all_models, voting='soft', weights=[2,1,1,1,1])
s=cross_val_score(vote,X_s,y_enc,cv=cv,scoring='balanced_accuracy')
print(f"  TabPFN Ensemble CV: {s.mean():.5f}")

vote.fit(X_s, y_enc)
preds=vote.predict(X_test_s)
preds_class=le.inverse_transform(preds)

pd.DataFrame({'id':test_id,'class':preds_class}).to_csv(os.path.join(os.path.dirname(__file__),"submission.csv"),index=False)
print(f"\n  Pred distribution: {dict(zip(*np.unique(preds_class,return_counts=True)))}")
print("v6 TabPFN done!")
