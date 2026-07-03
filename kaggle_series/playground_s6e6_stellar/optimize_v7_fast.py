"""
S6E6 v7 Fast — Chris Deotte features + GPU XGBoost (no prior)
"""
import os, sys, warnings; import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
if sys.platform=="win32":os.environ["PYTHONIOENCODING"]="utf-8";sys.stdout.reconfigure(encoding="utf-8");sys.stderr.reconfigure(encoding="utf-8")
DATA_DIR=os.path.join(os.path.dirname(__file__),"data")
train=pd.read_csv(os.path.join(DATA_DIR,"train.csv")); test=pd.read_csv(os.path.join(DATA_DIR,"test.csv"))
tid,test_id=train['id'],test['id']; y=train['class']
train.drop(['id','class'],axis=1,inplace=True); test.drop(['id'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder; le=LabelEncoder(); y_enc=le.fit_transform(y)
CLASSES=le.classes_

# ── Chris Deotte 风格特征工程 (CPU, 但很快) ──
def add_features(df):
    out=df.copy(); EPS=1e-6; B=['u','g','r','i','z']
    # 10对颜色差
    for a,b in [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('u','i'),('u','z'),('g','i'),('g','z'),('r','z')]:
        out[f'{a}_{b}']=(out[a]-out[b]).astype('float32')
    # 星等统计
    bv=out[B].values.astype('float32'); out['mag_mean']=bv.mean(axis=1); out['mag_std']=bv.std(axis=1,ddof=1)
    out['mag_min']=bv.min(axis=1); out['mag_max']=bv.max(axis=1); out['mag_range']=out['mag_max']-out['mag_min']
    # 弯曲度
    out['mag_curvature']=(out['u']-2*out['r']+out['z']).astype('float32')
    out['blue_curvature']=(out['u']-2*out['g']+out['r']).astype('float32')
    out['red_curvature']=(out['r']-2*out['i']+out['z']).astype('float32')
    # Flux (10^(-0.4*mag))
    fd={}; fb=np.column_stack([10**(-0.4*np.clip(out[b].values,-30,30)) for b in B])
    for i,b in enumerate(B): out[f'flux_{b}']=fb[:,i].astype('float32')
    out['flux_mean']=fb.mean(axis=1); out['flux_std']=fb.std(axis=1,ddof=1); out['flux_range']=fb.max(axis=1)-fb.min(axis=1)
    # Redshift交互
    for b in B:
        out[f'{b}_x_redshift']=(out[b]*out['redshift']).astype('float32')
        out[f'{b}_over_redshift']=(out[b]/(out['redshift'].abs()+EPS)).astype('float32')
    out['redshift_abs']=out['redshift'].abs(); out['redshift_is_neg']=(out['redshift']<0).astype('int8')
    # 天球坐标 → 3D
    ar,dr=np.deg2rad(out['alpha']),np.deg2rad(out['delta'])
    out['sin_a']=np.sin(ar); out['cos_a']=np.cos(ar); out['sin_d']=np.sin(dr); out['cos_d']=np.cos(dr)
    out['sky_x']=(np.cos(dr)*np.cos(ar)).astype('float32')
    out['sky_y']=(np.cos(dr)*np.sin(ar)).astype('float32')
    out['sky_z']=np.sin(dr).astype('float32')
    # 颜色平面半径
    if 'u_g' in out.columns and 'g_r' in out.columns:
        out['color_r_ug_gr']=np.sqrt(out['u_g']**2+out['g_r']**2).astype('float32')
    return out

train=add_features(train); test=add_features(test)
# 编码类别
for col in train.select_dtypes(include=['object']).columns:
    all_vals=pd.concat([train[col],test[col]]).astype('category').cat.codes
    train[col]=all_vals[:len(train)].values; test[col]=all_vals[len(train):].values

X,X_test=train.values.astype('float32'),test.values.astype('float32')
from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit_transform(X); X_test=StandardScaler().fit_transform(X_test)
print(f"Features: {X.shape[1]}")

# ── GPU XGBoost (Chris Deotte params) ──
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
import xgboost as xgb

base_kwargs=dict(objective='multi:softprob',num_class=3,eval_metric='mlogloss',
    tree_method='hist',device='cuda',learning_rate=0.012,n_estimators=5000,
    max_depth=0,max_leaves=72,grow_policy='lossguide',max_bin=512,min_child_weight=10,
    gamma=0.20,reg_alpha=0.30,reg_lambda=4.0,subsample=0.82,colsample_bytree=0.74,
    colsample_bylevel=0.86,verbosity=0)

# OOF预测
oof=np.zeros((len(X),3),dtype='float32')
test_preds=np.zeros((len(X_test),3),dtype='float32')

print("OOF training...")
for fold,(tr_idx,va_idx) in enumerate(cv.split(X,y_enc),1):
    print(f"  Fold {fold}/5...",end='',flush=True)
    X_tr,X_va=X[tr_idx],X[va_idx]; y_tr,y_va=y_enc[tr_idx],y_enc[va_idx]
    m=xgb.XGBClassifier(random_state=42+fold*100,**base_kwargs)
    m.fit(X_tr,y_tr,eval_set=[(X_va,y_va)],verbose=False)
    oof[va_idx]=m.predict_proba(X_va).astype('float32')
    test_preds+=m.predict_proba(X_test).astype('float32')/5
    acc=balanced_accuracy_score(y_va,oof[va_idx].argmax(axis=1))
    print(f" Acc: {acc:.5f}")

cv_score=balanced_accuracy_score(y_enc,oof.argmax(axis=1))
print(f"OOF: {cv_score:.5f}")

preds=le.inverse_transform(test_preds.argmax(axis=1))
pd.DataFrame({'id':test_id,'class':preds}).to_csv(os.path.join(os.path.dirname(__file__),"submission.csv"),index=False)
print("v7 fast done!")
