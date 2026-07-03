"""
S6E6 v7 — Chris Deotte 核心技巧 CPU 适配版
关键创新:
  1. 原始SDSS数据集 → prior特征
  2. Flux转换 (10^(-0.4*mag))
  3. 天球坐标(sky_x/y/z) + sin/cos
  4. 10对颜色差 + 光谱弯曲度
  5. XGBoost: max_leaves+lossguide+early_stopping
  6. OOF预测保存(用于stacking)
"""
import os, sys, warnings; import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
if sys.platform=="win32":os.environ["PYTHONIOENCODING"]="utf-8";sys.stdout.reconfigure(encoding="utf-8");sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR=os.path.join(os.path.dirname(__file__),"data")
# Competition data
train=pd.read_csv(os.path.join(DATA_DIR,"train.csv")); test=pd.read_csv(os.path.join(DATA_DIR,"test.csv"))
tid,test_id=train['id'],test['id']; y=train['class']
# Original SDSS data
orig=pd.read_csv(os.path.join(DATA_DIR,"star_classification.csv"))
print(f"Train: {train.shape}  Test: {test.shape}  Original: {orig.shape}")

from sklearn.preprocessing import LabelEncoder; le=LabelEncoder(); y_enc=le.fit_transform(y)
CLASSES=le.classes_

# ── 处理原始SDSS数据集 ──
# 给原始数据也生成spectral_type和galaxy_population
def spectral_type_fn(g,r):
    return pd.cut(r-g, [-np.inf,-1,-0.5,0,np.inf], labels=['M','G/K','A/F','O/B']).astype(str)
def galaxy_pop_fn(u,r):
    return pd.cut(u-r, [-np.inf,2.2,np.inf], labels=['Blue_Cloud','Red_Sequence']).astype(str)

if 'spectral_type' not in orig.columns:
    orig['spectral_type'] = spectral_type_fn(orig['g'],orig['r'])
if 'galaxy_population' not in orig.columns:
    orig['galaxy_population'] = galaxy_pop_fn(orig['u'],orig['r'])
orig_y = orig['class']; orig.drop('class',axis=1,inplace=True)
orig_y_enc=np.array(le.transform(orig_y))

# ── 特征工程 ──
def add_all_features(df):
    out=df.copy()
    EPS=1e-6; BANDS=['u','g','r','i','z']
    # 10对颜色差
    pairs=[('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('u','i'),('u','z'),('g','i'),('g','z'),('r','z')]
    for a,b in pairs:
        out[f'{a}_{b}']=(out[a]-out[b]).astype('float32')
    # 星等统计
    band_vals=out[BANDS].values.astype('float32')
    out['mag_mean']=band_vals.mean(axis=1); out['mag_std']=band_vals.std(axis=1,ddof=1)
    out['mag_min']=band_vals.min(axis=1); out['mag_max']=band_vals.max(axis=1)
    out['mag_range']=out['mag_max']-out['mag_min']
    # 光谱弯曲度
    out['mag_curvature']=(out['u']-2*out['r']+out['z']).astype('float32')
    out['blue_curvature']=(out['u']-2*out['g']+out['r']).astype('float32')
    out['red_curvature']=(out['r']-2*out['i']+out['z']).astype('float32')
    # 星等斜率
    x_centered=np.arange(5)-2; denom=np.sum(x_centered**2)
    out['mag_slope']=(band_vals-x_centered).dot(x_centered)/denom
    # redshift交互
    for b in BANDS:
        out[f'{b}_x_redshift']=(out[b]*out['redshift']).astype('float32')
        out[f'{b}_over_redshift']=(out[b]/(out['redshift'].abs()+EPS)).astype('float32')
    out['redshift_abs']=out['redshift'].abs().astype('float32')
    out['redshift_log1p_abs']=np.log1p(out['redshift_abs']).astype('float32')
    out['redshift_is_neg']=(out['redshift']<0).astype('int8')
    # Flux转换 (10^(-0.4*mag))
    flux_dict={}
    for b in BANDS:
        clipped=np.clip(out[b].values,-30,30)
        flux=10**(-0.4*clipped)
        out[f'flux_{b}']=flux.astype('float32'); flux_dict[b]=flux
    flux_arr=np.column_stack(list(flux_dict.values()))
    out['flux_mean']=flux_arr.mean(axis=1); out['flux_std']=flux_arr.std(axis=1,ddof=1)
    out['flux_min']=flux_arr.min(axis=1); out['flux_max']=flux_arr.max(axis=1)
    out['flux_range']=out['flux_max']-out['flux_min']
    # 天球坐标
    alpha_rad=np.deg2rad(out['alpha']); delta_rad=np.deg2rad(out['delta'])
    out['sin_alpha']=np.sin(alpha_rad); out['cos_alpha']=np.cos(alpha_rad)
    out['sin_delta']=np.sin(delta_rad); out['cos_delta']=np.cos(delta_rad)
    out['sky_x']=np.cos(delta_rad)*np.cos(alpha_rad).astype('float32')
    out['sky_y']=np.cos(delta_rad)*np.sin(alpha_rad).astype('float32')
    out['sky_z']=np.sin(delta_rad).astype('float32')
    # 颜色平面
    if 'u_g' in out.columns and 'g_r' in out.columns:
        out['color_radius_ug_gr']=np.sqrt(out['u_g']**2+out['g_r']**2).astype('float32')
    # 计算的类别
    out['spectral_type_calc']=spectral_type_fn(out['g'],out['r'])
    out['galaxy_pop_calc']=galaxy_pop_fn(out['u'],out['r'])
    return out

train=add_all_features(train); test=add_all_features(test); orig=add_all_features(orig)
train.drop(['id','class'],axis=1,inplace=True); test.drop(['id'],axis=1,inplace=True)

# ── 原始数据prior特征 (关键! 简化版) ──
for col in ['spectral_type','galaxy_population','spectral_type_calc','galaxy_pop_calc']:
    if col not in train.columns: continue
    train[col]=train[col].astype(str).fillna('NA')
    test[col]=test[col].astype(str).fillna('NA')
    orig[col]=orig[col].astype(str).fillna('NA')
    # 用crosstab快速计算每个类别×class的分布
    ct=pd.crosstab(orig[col],orig_y,normalize='index')
    for cls_name in CLASSES:
        rates=ct[cls_name].to_dict() if cls_name in ct.columns else {}
        train[f'prior_{col}_{cls_name}']=train[col].map(rates).fillna(0.33).astype('float32')
        test[f'prior_{col}_{cls_name}']=test[col].map(rates).fillna(0.33).astype('float32')
    # Count
    vc=orig[col].value_counts().to_dict()
    train[f'{col}_count']=train[col].map(vc).fillna(1).astype('float32')
    test[f'{col}_count']=test[col].map(vc).fillna(1).astype('float32')

# ── 全部编码 ──
all_data=pd.concat([train,test],sort=False).reset_index(drop=True); n_train=len(train)
for col in all_data.select_dtypes(include=['object']).columns: all_data[col]=all_data[col].astype('category').cat.codes.astype('int16')
train_p=all_data[:n_train]; test_p=all_data[n_train:]

# 删掉id列(如果还有)
for c in ['id','rerun_ID','field_ID','spec_obj_ID','plate','MJD','fiber_ID']:
    if c in train_p.columns: train_p.drop(c,axis=1,inplace=True)
    if c in test_p.columns: test_p.drop(c,axis=1,inplace=True)

X,X_test=train_p.values.astype('float32'),test_p.values.astype('float32')
print(f"Final features: {X.shape[1]}")
from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit_transform(X); X_test=StandardScaler().fit_transform(X_test)

# ── XGBoost with Chris Deotte params ──
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

import xgboost as xgb; import lightgbm as lgb

# XGBoost GPU with Chris Deotte params
xgb_m=xgb.XGBClassifier(
    objective='multi:softprob', num_class=3, eval_metric='mlogloss',
    tree_method='hist', device='cuda',  # GPU
    learning_rate=0.012, n_estimators=7000,
    max_depth=0, max_leaves=72, grow_policy='lossguide',
    max_bin=512, min_child_weight=10,
    gamma=0.20, reg_alpha=0.30, reg_lambda=4.0,
    subsample=0.82, colsample_bytree=0.74, colsample_bylevel=0.86,
    random_state=42, verbosity=0
)
s=cross_val_score(xgb_m,X,y_enc,cv=cv,scoring='balanced_accuracy',n_jobs=-1)
print(f"XGB GPU (Chris params) CV: {s.mean():.5f}")

# ── OOF预测 (Chris Deotte方法核心) ──
print("\nGenerating OOF predictions...")
oof=np.zeros((len(X),3),dtype='float32')
test_preds=np.zeros((len(X_test),3),dtype='float32')

for fold,(tr_idx,va_idx) in enumerate(cv.split(X,y_enc),1):
    print(f"  Fold {fold}/5...",end='')
    X_tr,X_va=X[tr_idx],X[va_idx]; y_tr,y_va=y_enc[tr_idx],y_enc[va_idx]
    # Train
    m=xgb.XGBClassifier(objective='multi:softprob',num_class=3,eval_metric='mlogloss',
        tree_method='hist',device='cuda',learning_rate=0.012,n_estimators=7000,
        max_depth=0,max_leaves=72,grow_policy='lossguide',max_bin=512,min_child_weight=10,
        gamma=0.20,reg_alpha=0.30,reg_lambda=4.0,subsample=0.82,colsample_bytree=0.74,
        colsample_bylevel=0.86,random_state=42+fold*100,verbosity=0)
    m.fit(X_tr,y_tr,eval_set=[(X_va,y_va)],verbose=False)
    oof[va_idx]=m.predict_proba(X_va).astype('float32')
    test_preds+=m.predict_proba(X_test).astype('float32')/5
    acc=(oof[va_idx].argmax(axis=1)==y_va).mean()
    print(f" Val acc: {acc:.5f}")

from sklearn.metrics import balanced_accuracy_score
cv_score=balanced_accuracy_score(y_enc,oof.argmax(axis=1))
print(f"\nOOF Balanced Accuracy: {cv_score:.5f}")

# ── Stacking: 用OOF训练LogisticRegression元模型 ──
print("\nTraining meta-model (LogisticRegression on OOF)...")
from sklearn.linear_model import LogisticRegression
meta=LogisticRegression(max_iter=2000,multi_class='multinomial',random_state=42)
meta.fit(oof,y_enc)
final_preds=meta.predict_proba(test_preds).argmax(axis=1)
final_class=le.inverse_transform(final_preds)

preds_class=le.inverse_transform(test_preds.argmax(axis=1))
pd.DataFrame({'id':test_id,'class':preds_class}).to_csv(os.path.join(os.path.dirname(__file__),"submission.csv"),index=False)
print(f"v7 done! OOF: {cv_score:.5f}")
