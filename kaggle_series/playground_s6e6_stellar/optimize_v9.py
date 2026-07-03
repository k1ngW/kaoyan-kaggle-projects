"""
S6E6 v9 — GPU加速 五模型 Stacking
===================================
XGBoost + LightGBM + CatBoost + FastMLP → LogisticRegression 元模型
全部运行在 RTX 3060 GPU 上

策略:
  1. 复用 v8 特征工程 (Quantile Binning + Fold-Safe TE + Prior + etc.)
  2. 5折CV训练4个异构模型，全部GPU加速
  3. OOF预测 → LogisticRegression 元模型
  4. 测试集: 各模型预测 → 元模型融合 → 最终分类

运行: C:/Users/Lenovo/kaggle_env/python -s optimize_v9.py
"""
import os, sys, warnings, gc, site
site.ENABLE_USER_SITE = False  # 使用 conda env 的包

import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    except: pass

# ═══════════════════════════ 常量 ═══════════════════════════
SEED = 42; N_SPLITS = 5
TARGET, ID_COL = 'class', 'id'
CLASSES = ['GALAXY', 'QSO', 'STAR']
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}
EPS = 1e-6
RAW_NUM_COLS = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
BANDS = ['u', 'g', 'r', 'i', 'z']
TE_SMOOTH, TE_INNER_SPLITS = 20.0, 5

DATA_DIR = Path(__file__).parent / "data"
SUBMISSION_PATH = Path(__file__).parent / "submission.csv"
OOF_DIR = Path(__file__).parent / "oof"; OOF_DIR.mkdir(exist_ok=True)
ORIG_PATH = DATA_DIR / "star_classification.csv"

from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb, lightgbm as lgb, catboost as cb
# FastMLP: 轻量 PyTorch MLP 替代 RealMLP (快 4x, GPU 加速)

# ═══════════════════════════ 240 TOP_FEATURES ═══════════════════════════
TOP_FEATURES = [
    'redshift_u', 'u_over_redshift', 'z_over_redshift', 'g_over_redshift',
    'g_z', 'redshift_g', 'g_i', 'u_i', 'u_r_abs',
    'TE_redshift_qbin64__x__mag_mean_qbin64_QSO',
    'i_over_redshift', 'u_r', 'g_i_abs',
    'TE_redshift_qbin16_GALAXY', 'redshift_z',
    'TE_redshift_qbin64__x__mag_mean_qbin64_GALAXY',
    'redshift_abs', 'TE_redshift_qbin64_GALAXY',
    'orig_g_qbin64_prior_QSO', 'redshift_log1p_abs',
    'orig_g_qbin16_prior_QSO', 'orig_redshift_qbin64_prior_GALAXY',
    'redshift', 'TE_redshift_qbin64_QSO', 'mag_slope',
    'TE_u_r_qbin64_GALAXY', 'TE_alpha_qbin64__x__delta_qbin64_STAR',
    'r_over_redshift', 'flux_g', 'redshift_i', 'flux_std', 'g',
    'TE_alpha_qbin64__x__delta_qbin64_GALAXY',
    'TE_u_g_qbin64__x__g_r_qbin64_STAR', 'redshift_is_neg',
    'g_qbin16', 'orig_g_qbin256_prior_QSO', 'TE_u_r_qbin64_QSO',
    'flux_range', 'mag_std', 'orig_g_qbin16_prior_GALAXY',
    'redshift_r', 'orig_redshift_qbin64_prior_STAR',
    'orig_u_r_qbin16_prior_QSO', 'u_z',
    'orig_redshift_qbin64__x__mag_mean_qbin64_prior_QSO',
    'TE_redshift_qbin64_STAR', 'TE_g_qbin64_QSO',
    'orig_redshift_qbin16_prior_GALAXY', 'TE_g_qbin16_QSO',
    'u_g', 'z', 'orig_alpha_qbin64__x__delta_qbin64_prior_GALAXY',
    'g_i_x_redshift', 'orig_z_qbin16_prior_QSO',
    'orig_alpha_qbin64__x__delta_qbin64_prior_STAR',
    'color_plane_radius_ug_gr', 'i', 'flux_z', 'TE_i_qbin64_QSO',
    'TE_g_qbin64_GALAXY', 'orig_redshift_qbin256_prior_GALAXY',
    'r', 'TE_u_r_qbin16_GALAXY', 'flux_i', 'r_i_x_redshift',
    'flux_r', 'r_z', 'orig_i_qbin16_prior_QSO', 'r_z_x_redshift',
    'g_r_x_redshift', 'orig_mag_range_qbin16_prior_STAR',
    'r_z_abs', 'mag_max', 'TE_g_qbin16_GALAXY',
    'orig_mag_range_qbin64_prior_STAR', 'TE_i_qbin16_QSO',
    'flux_min', 'TE_u_g_qbin64_STAR', 'orig_u_qbin16_prior_QSO',
    'TE_redshift_qbin64__x__mag_mean_qbin64_STAR',
    'orig_u_g_qbin16_prior_STAR', 'flux_max',
    'orig_z_qbin64_prior_QSO', 'TE_redshift_qbin16_STAR',
    'mag_range', 'TE_u_g_qbin64__x__g_r_qbin64_QSO',
    'g_r', 'orig_redshift_qbin64__x__mag_mean_qbin64_prior_GALAXY',
    'redshift_qbin16', 'mag_mean_qbin16',
    'TE_u_g_qbin16_STAR', 'TE_z_qbin64_QSO', 'u_g_abs',
    'orig_u_r_qbin64_prior_QSO', 'mag_min',
    'orig_r_qbin16_prior_QSO', 'redshift_qbin64__x__mag_mean_qbin64',
    'u_r_x_redshift', 'orig_i_qbin64_prior_QSO', 'u', 'flux_u',
    'TE_redshift_qbin16_QSO', 'flux_mean',
    'redshift_qbin64__x__mag_mean_qbin64_freq',
    'TE_alpha_qbin64__x__delta_qbin64_QSO',
    'redshift_qbin64__x__mag_mean_qbin64_freq_log1p',
    'u_g_x_redshift', 'u_qbin16', 'TE_g_r_qbin64_GALAXY',
    'color_plane_radius_ri_iz', 'orig_z_qbin256_prior_QSO',
    'redshift_qbin256', 'TE_mag_range_qbin64_QSO',
    'g_r_abs', 'orig_mag_range_qbin256_prior_STAR',
    'orig_g_qbin64_prior_GALAXY', 'orig_mag_range_qbin16_prior_GALAXY',
    'r_i', 'r_qbin16', 'TE_r_qbin64_STAR', 'TE_g_r_qbin64_STAR',
    'TE_u_r_qbin16_QSO', 'orig_u_qbin16_prior_STAR',
    'alpha_sin', 'TE_u_g_qbin64_QSO', 'orig_spectral_x_pop_prior_QSO',
    'r_qbin64', 'sky_y', 'u_g_qbin16', 'mag_range_qbin256',
    'TE_r_i_qbin64_QSO', 'TE_mag_range_qbin64_GALAXY',
    'orig_alpha_qbin256_prior_STAR', 'alpha_qbin256',
    'z_qbin16', 'delta_cos', 'orig_u_qbin64_prior_QSO',
    'g_qbin64', 'TE_r_qbin16_QSO', 'TE_z_qbin16_QSO',
    'color_plane_angle_ug_gr', 'mag_range_qbin16',
    'TE_g_qbin64_STAR', 'TE_g_r_qbin64_QSO',
    'orig_u_g_qbin64_prior_STAR', 'TE_r_i_qbin16_QSO',
    'blue_curvature', 'TE_r_i_qbin64_GALAXY',
    'TE_u_qbin64_STAR', 'TE_u_qbin64_QSO', 'mag_mean',
    'TE_i_qbin16_GALAXY', 'TE_u_g_qbin16_QSO',
    'TE_u_g_qbin64_GALAXY', 'delta', 'delta_sin', 'alpha',
    'sky_x', 'sky_z', 'i_qbin16', 'redshift_qbin64',
    'TE_g_r_qbin16_STAR', 'mag_curvature',
    'TE_g_qbin16_STAR', 'TE_alpha_qbin16_QSO',
    'TE_mag_range_qbin16_QSO', 'TE_u_g_qbin64__x__g_r_qbin64_GALAXY',
    'TE_mag_range_qbin16_STAR',
]

# ═══════════════════════════ 特征工程 ═══════════════════════════
def cat_key(s): return s.astype(str).fillna('__NA__')

def spectral_type(g, r):
    return pd.cut(r - g, [-np.inf, -1, -0.5, 0, np.inf],
                   labels=['M', 'G/K', 'A/F', 'O/B']).astype(str)

def galaxy_population(u, r):
    return pd.cut(u - r, [-np.inf, 2.2, np.inf],
                   labels=['Blue_Cloud', 'Red_Sequence']).astype(str)

def add_public_features(df):
    out = df.copy()
    for c in RAW_NUM_COLS: out[c] = pd.to_numeric(out[c], errors='coerce').astype('float32')
    for a, b in [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('u','i'),('u','z'),('g','i'),('g','z'),('r','z')]:
        out[f'{a}_{b}'] = (out[a] - out[b]).astype('float32')
    bv = out[list(BANDS)].values.astype(np.float32)
    out['mag_mean'] = bv.mean(axis=1).astype('float32')
    out['mag_std'] = bv.std(axis=1, ddof=1).astype('float32')
    out['mag_min'] = bv.min(axis=1).astype('float32')
    out['mag_max'] = bv.max(axis=1).astype('float32')
    out['mag_range'] = (out['mag_max'] - out['mag_min']).astype('float32')
    out['mag_argmin'] = bv.argmin(axis=1).astype('int16')
    out['mag_argmax'] = bv.argmax(axis=1).astype('int16')
    for b in BANDS:
        out[f'redshift_{b}'] = (out['redshift'] * out[b]).astype('float32')
        out[f'{b}_over_redshift'] = (out[b] / (out['redshift'].abs() + EPS)).astype('float32')
    ar = np.deg2rad(out['alpha'].values.astype(np.float32))
    dr = np.deg2rad(out['delta'].values.astype(np.float32))
    out['alpha_sin'] = np.sin(ar).astype('float32'); out['alpha_cos'] = np.cos(ar).astype('float32')
    out['delta_sin'] = np.sin(dr).astype('float32'); out['delta_cos'] = np.cos(dr).astype('float32')
    out['sky_x'] = (np.cos(dr)*np.cos(ar)).astype('float32')
    out['sky_y'] = (np.cos(dr)*np.sin(ar)).astype('float32')
    out['sky_z'] = np.sin(dr).astype('float32')
    fa = []
    for b in BANDS:
        f = np.power(10.0, -0.4*np.clip(out[b].values.astype(np.float32),-30,30)).astype('float32')
        out[f'flux_{b}'] = f; fa.append(f)
    fv = np.column_stack(fa)
    out['flux_mean'] = fv.mean(axis=1).astype('float32')
    out['flux_std'] = fv.std(axis=1, ddof=1).astype('float32')
    out['flux_min'] = fv.min(axis=1).astype('float32')
    out['flux_max'] = fv.max(axis=1).astype('float32')
    out['flux_range'] = (out['flux_max']-out['flux_min']).astype('float32')
    x=np.arange(5,dtype=np.float32); xc=x-x.mean()
    out['mag_slope'] = ((bv-bv.mean(axis=1,keepdims=True)).dot(xc)/np.sum(xc**2)).astype('float32')
    out['mag_curvature'] = (out['u']-2*out['r']+out['z']).astype('float32')
    out['blue_curvature'] = (out['u']-2*out['g']+out['r']).astype('float32')
    out['red_curvature'] = (out['r']-2*out['i']+out['z']).astype('float32')
    out['redshift_abs'] = out['redshift'].abs().astype('float32')
    out['redshift_log1p_abs'] = np.log1p(out['redshift_abs'].values).astype('float32')
    out['redshift_is_neg'] = (out['redshift']<0).astype('int8')
    out['spectral_type_calc'] = spectral_type(out['g'],out['r'])
    out['galaxy_population_calc'] = galaxy_population(out['u'],out['r'])
    out['spectral_type'] = cat_key(out['spectral_type'])
    out['galaxy_population'] = cat_key(out['galaxy_population'])
    out['spectral_x_pop'] = cat_key(out['spectral_type'])+'__'+cat_key(out['galaxy_population'])
    out['spectral_calc_x_pop_calc'] = cat_key(out['spectral_type_calc'])+'__'+cat_key(out['galaxy_population_calc'])
    return out.replace([np.inf,-np.inf],np.nan)

def add_pairwise_geometry_features(df):
    out = df.copy()
    for c in ['u_g','g_r','r_i','i_z','u_r','g_i','r_z']:
        if c in out.columns:
            out[f'{c}_x_redshift'] = (out[c]*out['redshift']).astype('float32')
            out[f'{c}_abs'] = out[c].abs().astype('float32')
    if 'u_g' in out.columns and 'g_r' in out.columns:
        ug=out['u_g'].values.astype(np.float32); gr=out['g_r'].values.astype(np.float32)
        out['color_plane_radius_ug_gr']=np.sqrt(ug**2+gr**2).astype('float32')
        out['color_plane_angle_ug_gr']=np.arctan2(ug,gr+EPS).astype('float32')
    if 'r_i' in out.columns and 'i_z' in out.columns:
        ri=out['r_i'].values.astype(np.float32); iz=out['i_z'].values.astype(np.float32)
        out['color_plane_radius_ri_iz']=np.sqrt(ri**2+iz**2).astype('float32')
        out['color_plane_angle_ri_iz']=np.arctan2(ri,iz+EPS).astype('float32')
    return out

def add_quantile_bin_features(df, ttm):
    out=df.copy(); qc=[]
    qcols=list(dict.fromkeys(RAW_NUM_COLS+['u_g','g_r','r_i','i_z','u_r','mag_mean','mag_range']))
    for c in qcols:
        if c not in out.columns: continue
        s=pd.to_numeric(out[c],errors='coerce'); ref=s[ttm].dropna()
        if len(ref)<2: continue
        for q in [16,64,256]:
            try: codes=pd.qcut(s,q=q,labels=False,duplicates='drop').fillna(-1).astype(int).astype(str)
            except: codes=pd.Series(pd.cut(s,bins=q,labels=False),index=s.index).fillna(-1).astype(int).astype(str)
            out[f'{c}_qbin{q}']=codes; qc.append(f'{c}_qbin{q}')
    for a,b in [('alpha_qbin64','delta_qbin64'),('u_g_qbin64','g_r_qbin64'),('redshift_qbin64','mag_mean_qbin64')]:
        if a in out.columns and b in out.columns:
            name=f'{a}__x__{b}'; out[name]=cat_key(out[a])+'__'+cat_key(out[b]); qc.append(name)
    return out,qc

def add_frequency_features(df,cols,fm):
    out=df.copy()
    for c in cols:
        if c not in out.columns: continue
        s=cat_key(out[c]); vc=s[fm].value_counts(dropna=False)
        out[f'{c}_freq']=s.map(vc).fillna(0).astype('float32')
        out[f'{c}_freq_log1p']=np.log1p(out[f'{c}_freq'].values).astype('float32')
    return out

def add_original_prior_features(df,cols,om,oy):
    out=df.copy(); om=om.astype(bool)
    pc=np.bincount(oy.values.astype(np.int32),minlength=3).astype(np.float32)
    prior=pc/np.maximum(pc.sum(),1.0)
    for c in cols:
        if c not in out.columns: continue
        key=cat_key(out[c]); ok=key[om].reset_index(drop=True)
        tmp=pd.DataFrame({'key':ok,'y':oy.reset_index(drop=True)})
        vc=ok.value_counts().to_dict()
        out[f'orig_{c}_count']=key.map(vc).fillna(0).astype('float32')
        for ci,cn in INT_TO_CLASS.items():
            rates=tmp.assign(hit=(tmp['y']==ci).astype('float32')).groupby('key')['hit'].mean()
            out[f'orig_{c}_prior_{cn}']=key.map(rates.to_dict()).fillna(float(prior[ci])).astype('float32')
    return out

def select_te_cols(df,cat_cols,max_card=5000):
    cols=[]
    for c in cat_cols:
        if c not in df.columns: continue
        if cat_key(df[c]).nunique(dropna=False)>max_card: continue
        if (c in ['spectral_type','galaxy_population','spectral_type_calc','galaxy_population_calc','spectral_x_pop','spectral_calc_x_pop_calc']
            or '_qbin16' in c or '_qbin64' in c or '_qbin256' in c or '__x__' in c): cols.append(c)
    return cols

def te_sources_needed(tf,ate):
    return [c for c in ate if any(str(f).startswith(f'TE_{c}_') for f in tf)]

def sorted_factorize_three(ts,vs,es):
    vals=pd.concat([cat_key(ts),cat_key(vs),cat_key(es)],ignore_index=True)
    cats=vals.drop_duplicates().sort_values(ignore_index=True)
    mapper={v:i for i,v in enumerate(cats)}
    codes=vals.map(mapper).fillna(-1).astype('int32').reset_index(drop=True)
    nt,nv=len(ts),len(vs)
    return (codes.iloc[:nt].reset_index(drop=True),codes.iloc[nt:nt+nv].reset_index(drop=True),codes.iloc[nt+nv:].reset_index(drop=True))

def add_fold_safe_te(X_tr,y_tr,X_va,X_te,te_cols):
    if not te_cols: return X_tr,X_va,X_te,[]
    X_tr,X_va,X_te=X_tr.copy(),X_va.copy(),X_te.copy(); added=[]
    for c in te_cols:
        if c not in X_tr.columns: continue
        tc,vc,ec=sorted_factorize_three(X_tr[c],X_va[c],X_te[c])
        for ci,cn in INT_TO_CLASS.items():
            yb=(y_tr.values==ci).astype('float32')
            enc=TargetEncoder(cv=TE_INNER_SPLITS,smooth=TE_SMOOTH,target_type='continuous',random_state=SEED+177)
            tvals=enc.fit_transform(tc.values.reshape(-1,1),yb).ravel().astype('float32')
            vvals=enc.transform(vc.values.reshape(-1,1)).ravel().astype('float32')
            evals=enc.transform(ec.values.reshape(-1,1)).ravel().astype('float32')
            name=f'TE_{c}_{cn}'; X_tr[name]=tvals; X_va[name]=vvals; X_te[name]=evals; added.append(name)
    del enc
    return X_tr,X_va,X_te,added

def encode_model_categories(X_tr,X_va,X_te,cat_cols):
    X_tr,X_va,X_te=X_tr.copy(),X_va.copy(),X_te.copy()
    for c in cat_cols:
        if c not in X_tr.columns: continue
        tc,vc,ec=sorted_factorize_three(X_tr[c],X_va[c],X_te[c])
        X_tr[c]=tc.values; X_va[c]=vc.values; X_te[c]=ec.values
    return X_tr,X_va,X_te

def class_weights(ys):
    counts=np.bincount(ys.values.astype(np.int32),minlength=3).astype(np.float32)
    wpk=np.float32(len(ys))/(np.float32(3)*np.maximum(counts,1.0))
    return wpk[ys.values.astype(np.int32)].astype(np.float32)

# ═══════════════════════════ 模型工厂 ═══════════════════════════
def make_xgb(s): return xgb.XGBClassifier(
    objective='multi:softprob',num_class=3,eval_metric='mlogloss',
    tree_method='hist',device='cuda',learning_rate=0.012,n_estimators=7000,
    early_stopping_rounds=180,max_depth=0,max_leaves=72,grow_policy='lossguide',
    max_bin=512,min_child_weight=10,gamma=0.20,reg_alpha=0.30,reg_lambda=4.0,
    subsample=0.82,colsample_bytree=0.74,colsample_bylevel=0.86,
    random_state=s,n_jobs=4,verbosity=0)

def make_lgb(s): return lgb.LGBMClassifier(
    objective='multiclass',num_class=3,boosting_type='gbdt',
    learning_rate=0.02,n_estimators=3000,early_stopping_rounds=100,
    num_leaves=72,max_depth=-1,min_child_samples=10,
    subsample=0.75,colsample_bytree=0.65,reg_alpha=0.30,reg_lambda=4.0,
    random_state=s,n_jobs=-1,verbose=-1,device='gpu')

def make_cat(s): return cb.CatBoostClassifier(
    loss_function='MultiClass',num_boost_round=3000,
    learning_rate=0.02,depth=7,l2_leaf_reg=4.0,
    bootstrap_type='Bernoulli',subsample=0.75,
    random_seed=s,thread_count=-1,verbose=0,early_stopping_rounds=100,
    task_type='GPU',devices='0')

# 轻量 PyTorch MLP — sklearn 兼容，GPU 加速，每折 ~2 分钟
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin

class FastMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=[256,128,64], lr=0.001, epochs=80, batch=512, seed=42):
        self.hidden,self.lr,self.epochs,self.batch,self.seed=hidden,lr,epochs,batch,seed
    def fit(self,X,y):
        X=torch.tensor(X,dtype=torch.float32).cuda()
        y=torch.tensor(y,dtype=torch.long).cuda()
        torch.manual_seed(self.seed)
        layers=[]; ins=X.shape[1]
        for h in self.hidden: layers+=[nn.Linear(ins,h),nn.ReLU(),nn.BatchNorm1d(h),nn.Dropout(0.2)]; ins=h
        layers+=[nn.Linear(ins,3)]
        self.net_=nn.Sequential(*layers).cuda()
        opt=optim.AdamW(self.net_.parameters(),lr=self.lr,weight_decay=1e-4)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=self.epochs)
        loss_fn=nn.CrossEntropyLoss()
        n=X.shape[0]
        for ep in range(self.epochs):
            perm=torch.randperm(n)
            for i in range(0,n,self.batch):
                idx=perm[i:i+self.batch]; opt.zero_grad()
                loss=loss_fn(self.net_(X[idx]),y[idx]); loss.backward(); opt.step()
            sch.step()
        self.classes_=np.array([0,1,2])
        return self
    def predict_proba(self,X):
        self.net_.eval()
        with torch.no_grad():
            Xt=torch.tensor(X,dtype=torch.float32).cuda()
            logits=torch.cat([self.net_(Xt[i:i+self.batch]) for i in range(0,Xt.shape[0],self.batch)])
            return torch.softmax(logits,dim=1).cpu().numpy().astype('float32')
    def predict(self,X): return self.predict_proba(X).argmax(axis=1)

def make_mlp(s): return FastMLP(hidden=[256,128,64],lr=0.001,epochs=80,batch=512,seed=s)

# ═══════════════════════════ Main ═══════════════════════════
if __name__ == '__main__':
    np.random.seed(SEED)

    print("="*60)
    print("  v9 GPU: XGB + LGB + CatBoost + RealMLP Stacking")
    print("="*60)

    # Load
    train=pd.read_csv(DATA_DIR/"train.csv"); test=pd.read_csv(DATA_DIR/"test.csv")
    sample=pd.read_csv(DATA_DIR/"sample_submission.csv")
    le=LabelEncoder()
    y=pd.Series(le.fit_transform(train[TARGET]),name='y').reset_index(drop=True)
    test_ids=test[ID_COL].copy()
    train.drop([ID_COL,TARGET],axis=1,inplace=True); test.drop([ID_COL],axis=1,inplace=True)

    orig=pd.read_csv(ORIG_PATH)
    if 'spectral_type' not in orig.columns: orig['spectral_type']=spectral_type(orig['g'],orig['r'])
    if 'galaxy_population' not in orig.columns: orig['galaxy_population']=galaxy_population(orig['u'],orig['r'])
    orig['spectral_type']=cat_key(orig['spectral_type'])
    orig['galaxy_population']=cat_key(orig['galaxy_population'])
    y_orig=pd.Series(le.transform(orig[TARGET]),name='y_orig').reset_index(drop=True)
    keep=RAW_NUM_COLS+['spectral_type','galaxy_population',TARGET]
    orig=orig[[c for c in keep if c in orig.columns]].copy(); orig.drop([TARGET],axis=1,inplace=True)
    print(f"  Train: {train.shape}  Test: {test.shape}  Orig: {orig.shape}")

    # Features
    print(f"\n{'='*60}")
    print("  Step 1/4: 特征工程")
    print("="*60)
    trb,teb,orb=train.copy(),test.copy(),orig.copy()
    trb['_src']='train'; teb['_src']='test'; orb['_src']='orig'
    all_df=pd.concat([trb,teb,orb],axis=0,ignore_index=True)
    all_df=add_public_features(all_df); all_df=add_pairwise_geometry_features(all_df)
    ttm=all_df['_src'].isin(['train','test'])
    cat_cols=['spectral_type','galaxy_population','spectral_type_calc','galaxy_population_calc','spectral_x_pop','spectral_calc_x_pop_calc']
    all_df,qc=add_quantile_bin_features(all_df,ttm); cat_cols+=qc
    cat_cols=[c for c in dict.fromkeys(cat_cols) if c in all_df.columns]
    fm=all_df['_src'].isin(['train','test','orig'])
    fcols=select_te_cols(all_df,cat_cols,max_card=20000)
    all_df=add_frequency_features(all_df,fcols,fm)
    om=all_df['_src'].eq('orig')
    pcols=select_te_cols(all_df,cat_cols,max_card=10000)
    all_df=add_original_prior_features(all_df,pcols,om,y_orig)
    all_df['is_orig']=all_df['_src'].eq('orig').astype('int8')
    all_df['is_test']=all_df['_src'].eq('test').astype('int8')
    dc=[c for c in [ID_COL,'_src'] if c in all_df.columns]
    all_df=all_df.drop(columns=dc).replace([np.inf,-np.inf],np.nan)
    nt,nte=len(trb),len(teb)
    X=all_df.iloc[:nt].reset_index(drop=True)
    X_test=all_df.iloc[nt:nt+nte].reset_index(drop=True)
    cat_cols=[c for c in cat_cols if c in X.columns]
    print(f"  特征: {X.shape[1]} 列  |  类别列: {len(cat_cols)}")
    ate=select_te_cols(X,cat_cols,max_card=5000)
    TE_COLS=te_sources_needed(TOP_FEATURES,ate)
    MC=[c for c in cat_cols if c in TOP_FEATURES]
    print(f"  TE列: {len(TE_COLS)}  |  模型类别列: {len(MC)}")
    del all_df,trb,teb,orb,train,test,orig; gc.collect()

    # Train
    print(f"\n{'='*60}")
    print("  Step 2/4: 5-Fold CV — GPU 训练 4 模型")
    print("="*60)
    y_np=y.values.astype(np.int32); NC=3
    skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)

    models={
        'xgb':{'mk':make_xgb,'oof':np.zeros((len(X),NC),dtype='float32'),'tst':np.zeros((len(X_test),NC),dtype='float32'),'name':'XGBoost','scale':False},
        'lgb':{'mk':make_lgb,'oof':np.zeros((len(X),NC),dtype='float32'),'tst':np.zeros((len(X_test),NC),dtype='float32'),'name':'LightGBM','scale':False},
        'cat':{'mk':make_cat,'oof':np.zeros((len(X),NC),dtype='float32'),'tst':np.zeros((len(X_test),NC),dtype='float32'),'name':'CatBoost','scale':False},
        'mlp':{'mk':make_mlp,'oof':np.zeros((len(X),NC),dtype='float32'),'tst':np.zeros((len(X_test),NC),dtype='float32'),'name':'MLP','scale':True},
    }

    for fold,(tr_idx,va_idx) in enumerate(skf.split(np.zeros(len(y_np),dtype=np.int8),y_np),start=1):
        fs=SEED+fold*100
        print(f"\n  --- Fold {fold}/5 (seed={fs}) ---")
        X_tr=X.iloc[tr_idx].reset_index(drop=True); y_tr=y.iloc[tr_idx].reset_index(drop=True)
        X_va=X.iloc[va_idx].reset_index(drop=True); y_va=y.iloc[va_idx].reset_index(drop=True)
        X_te=X_test.copy()
        X_tr,X_va,X_te,added_te=add_fold_safe_te(X_tr,y_tr,X_va,X_te,TE_COLS)
        X_tr,X_va,X_te=encode_model_categories(X_tr,X_va,X_te,MC)
        feats=[c for c in TOP_FEATURES if c in X_tr.columns]
        X_tr_a=np.nan_to_num(X_tr[feats].values.astype('float32'),nan=0.0)
        X_va_a=np.nan_to_num(X_va[feats].values.astype('float32'),nan=0.0)
        X_te_a=np.nan_to_num(X_te[feats].values.astype('float32'),nan=0.0)
        scaler=StandardScaler()
        X_tr_s=scaler.fit_transform(X_tr_a); X_va_s=scaler.transform(X_va_a); X_te_s=scaler.transform(X_te_a)
        sw=class_weights(y_tr)

        for key,m in models.items():
            us=m['scale']; xt,xv,xe=(X_tr_s,X_va_s,X_te_s) if us else (X_tr_a,X_va_a,X_te_a)
            clf=m['mk'](fs)
            try:
                if key=='xgb':
                    clf.fit(xt,y_tr.values,sample_weight=sw,eval_set=[(xv,y_va.values)],verbose=250)
                elif key in('lgb','cat'):
                    clf.fit(xt,y_tr.values,sample_weight=sw,eval_set=[(xv,y_va.values)])
                else:  # MLP
                    clf.fit(xt,y_tr.values)
                vp=clf.predict_proba(xv).astype('float32'); tp=clf.predict_proba(xe).astype('float32')
                m['oof'][va_idx]=vp; m['tst']+=tp/N_SPLITS
                print(f"    {m['name']:<10} Fold {fold} BAcc: {balanced_accuracy_score(y_va,vp.argmax(axis=1)):.6f}")
            except Exception as e:
                print(f"    {m['name']:<10} Fold {fold} ERROR: {e}")
        del X_tr,X_va,X_te,X_tr_a,X_va_a,X_te_a,X_tr_s,X_va_s,X_te_s,clf; gc.collect()

    # Stacking
    print(f"\n{'='*60}")
    print("  Step 3/4: Stacking — LogisticRegression 元模型")
    print("="*60)
    for key,m in models.items():
        print(f"  {m['name']:<10} OOF: {balanced_accuracy_score(y_np,m['oof'].argmax(axis=1)):.6f}")
    meta_X=np.column_stack([m['oof'] for m in models.values()])
    meta_X_test=np.column_stack([m['tst'] for m in models.values()])
    print(f"  Meta-features: {meta_X.shape[1]} (4 models × 3 classes)")
    meta=LogisticRegression(max_iter=5000,random_state=SEED,C=1.0)
    meta.fit(meta_X,y_np)
    meta_oof=meta.predict_proba(meta_X)
    mb=balanced_accuracy_score(y_np,meta_oof.argmax(axis=1))
    print(f"  Stacked OOF BAcc: {mb:.6f}")

    # Final
    print(f"\n{'='*60}")
    print("  Step 4/4: 提交")
    print("="*60)
    fp=meta.predict(meta_X_test)
    fc=le.inverse_transform(fp)
    submission=sample.copy(); submission[TARGET]=fc
    submission.to_csv(SUBMISSION_PATH,index=False)
    print(f"  {SUBMISSION_PATH}")
    print(f"  Distribution: {dict(zip(*np.unique(fc,return_counts=True)))}")
    for key,m in models.items():
        np.save(OOF_DIR/f'v9_{key}_oof.npy',m['oof'].astype('float32'))
        np.save(OOF_DIR/f'v9_{key}_test.npy',m['tst'].astype('float32'))
    np.save(OOF_DIR/'v9_meta_oof.npy',meta_oof.astype('float32'))
    np.save(OOF_DIR/'v9_meta_test.npy',meta.predict_proba(meta_X_test).astype('float32'))
    print(f"\n  ✅ v9 GPU Complete! Stacked OOF: {mb:.6f}")
    print(f"{'='*60}")
