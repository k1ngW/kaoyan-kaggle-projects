"""
S6E6 v11 — RealMLP + XGB 软投票 (GPU)
========================================
RealMLP_TD (pytabkit) 64 epoch + v8 XGB OOF → 软投票
目标: 冲 Top 15% (0.97059)

用法: C:/Users/Lenovo/kaggle_env/python -s -u optimize_v11.py
"""
import os, sys, warnings, gc, site
site.ENABLE_USER_SITE = False
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    except: pass

SEED=42; N_SPLITS=5; TARGET,ID_COL='class','id'
CLASSES=['GALAXY','QSO','STAR']; NC=3
INT_TO_CLASS={i:c for c,i in {'GALAXY':0,'QSO':1,'STAR':2}.items()}
EPS=1e-6; RAW=['alpha','delta','u','g','r','i','z','redshift']; BANDS=['u','g','r','i','z']
DATA_DIR=Path(__file__).parent/"data"; SUB=Path(__file__).parent/"submission.csv"
OOF_DIR=Path(__file__).parent/"oof"; OOF_DIR.mkdir(exist_ok=True)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pytabkit import RealMLP_TD_Classifier

# ═══ 特征工程 (简化版 — RealMLP 用原始特征 + 基础衍生) ═══
def cat_key(s): return s.astype(str).fillna('__NA__')

def spectral_type(g,r):
    return pd.cut(r-g,[-np.inf,-1,-0.5,0,np.inf],labels=['M','G/K','A/F','O/B']).astype(str)

def galaxy_population(u,r):
    return pd.cut(u-r,[-np.inf,2.2,np.inf],labels=['Blue_Cloud','Red_Sequence']).astype(str)

def build_features(train, test, orig):
    """RealMLP 适用: 原始特征 + 少量衍生特征 (32个)"""
    trb,teb,orb=train.copy(),test.copy(),orig.copy()
    trb['_s']='train'; teb['_s']='test'; orb['_s']='orig'
    all_df=pd.concat([trb,teb,orb],axis=0,ignore_index=True)

    # 数值特征
    for c in RAW: all_df[c]=pd.to_numeric(all_df[c],errors='coerce').astype('float32')

    # 颜色差
    for a,b in [('u','g'),('g','r'),('r','i'),('i','z'),('u','r')]:
        all_df[f'{a}_{b}']=(all_df[a]-all_df[b]).astype('float32')

    # 天球坐标
    ar=np.deg2rad(all_df['alpha'].values.astype(np.float32))
    dr=np.deg2rad(all_df['delta'].values.astype(np.float32))
    all_df['sin_alpha']=np.sin(ar).astype('float32'); all_df['cos_alpha']=np.cos(ar).astype('float32')
    all_df['sin_delta']=np.sin(dr).astype('float32'); all_df['cos_delta']=np.cos(dr).astype('float32')
    all_df['sky_x']=(np.cos(dr)*np.cos(ar)).astype('float32')
    all_df['sky_y']=(np.cos(dr)*np.sin(ar)).astype('float32')
    all_df['sky_z']=np.sin(dr).astype('float32')

    # 计算的类别 (字符串 → 编码)
    all_df['spectral_type_calc']=spectral_type(all_df['g'],all_df['r'])
    all_df['galaxy_population_calc']=galaxy_population(all_df['u'],all_df['r'])
    all_df['spectral_type']=cat_key(all_df['spectral_type'])
    all_df['galaxy_population']=cat_key(all_df['galaxy_population'])

    # 类别特征编码
    for c in ['spectral_type','galaxy_population','spectral_type_calc','galaxy_population_calc']:
        all_df[c]=all_df[c].astype('category').cat.codes.astype('float32')

    # 清理 + 分离
    all_df=all_df.replace([np.inf,-np.inf],np.nan).fillna(0)
    dc=[c for c in [ID_COL,'_s'] if c in all_df.columns]
    all_df=all_df.drop(columns=dc)
    nt,nte=len(trb),len(teb)
    X=all_df.iloc[:nt].reset_index(drop=True)
    X_test=all_df.iloc[nt:nt+nte].reset_index(drop=True)

    return X, X_test

# ═══ Main ═══
if __name__=='__main__':
    print("="*60)
    print("  v11: RealMLP + v8 XGB 软投票")
    print("="*60)

    # Load data
    train=pd.read_csv(DATA_DIR/"train.csv"); test=pd.read_csv(DATA_DIR/"test.csv")
    sample=pd.read_csv(DATA_DIR/"sample_submission.csv")
    le=LabelEncoder()
    y=pd.Series(le.fit_transform(train[TARGET]),name='y').reset_index(drop=True)
    test_ids=test[ID_COL].copy()
    train.drop([ID_COL,TARGET],axis=1,inplace=True); test.drop([ID_COL],axis=1,inplace=True)

    # Original SDSS
    orig=pd.read_csv(DATA_DIR/"star_classification.csv")
    if 'spectral_type' not in orig.columns: orig['spectral_type']=spectral_type(orig['g'],orig['r'])
    if 'galaxy_population' not in orig.columns: orig['galaxy_population']=galaxy_population(orig['u'],orig['r'])
    orig['spectral_type']=cat_key(orig['spectral_type'])
    orig['galaxy_population']=cat_key(orig['galaxy_population'])
    keep=RAW+['spectral_type','galaxy_population',TARGET]
    orig=orig[[c for c in keep if c in orig.columns]].copy(); orig.drop([TARGET],axis=1,inplace=True)

    # Build features
    print(f"  Train: {train.shape}  Test: {test.shape}  Orig: {orig.shape}")
    X, X_test = build_features(train, test, orig)
    print(f"  Features: {X.shape[1]} columns")

    # Load v8 XGB OOF
    v8_xgb_oof = np.load(OOF_DIR/"v8_oof_preds.npy")
    v8_xgb_test = np.load(OOF_DIR/"v8_test_preds.npy")
    v8_labels = np.load(OOF_DIR/"v8_oof_labels.npy")
    print(f"  Loaded v8 XGB OOF: BAcc {balanced_accuracy_score(v8_labels, v8_xgb_oof.argmax(axis=1)):.6f}")

    # Train RealMLP
    print(f"\n{'='*60}")
    print("  Training RealMLP_TD (GPU, 64 epoch, 5-fold CV)")
    print("="*60)

    y_np = y.values.astype(np.int32)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rmlp_oof = np.zeros((len(X), NC), dtype='float32')
    rmlp_test = np.zeros((len(X_test), NC), dtype='float32')
    X_arr = X.values.astype('float32')
    X_tst = X_test.values.astype('float32')

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_np), dtype=np.int8), y_np), start=1):
        print(f"  Fold {fold}/5...", end='', flush=True)
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        model = RealMLP_TD_Classifier(
            device='cuda', random_state=SEED + fold * 100,
            n_epochs=64, batch_size=512,
        )
        model.fit(X_tr, y_tr)
        rmlp_oof[va_idx] = model.predict_proba(X_va).astype('float32')
        rmlp_test += model.predict_proba(X_tst).astype('float32') / N_SPLITS
        bacc = balanced_accuracy_score(y_va, rmlp_oof[va_idx].argmax(axis=1))
        print(f" BAcc: {bacc:.6f}")

    rmlp_score = balanced_accuracy_score(y_np, rmlp_oof.argmax(axis=1))
    print(f"  RealMLP OOF BAcc: {rmlp_score:.6f}")

    # Soft Vote with v8 XGB
    print(f"\n{'='*60}")
    print("  Soft Vote: v8 XGB + v11 RealMLP")
    print("="*60)

    vote_oof = (v8_xgb_oof + rmlp_oof) / 2
    vote_test = (v8_xgb_test + rmlp_test) / 2
    vote_score = balanced_accuracy_score(y_np, vote_oof.argmax(axis=1))
    print(f"  v8 XGB     OOF: {balanced_accuracy_score(v8_labels, v8_xgb_oof.argmax(axis=1)):.6f}")
    print(f"  RealMLP    OOF: {rmlp_score:.6f}")
    print(f"  XGB+RealMLP OOF: {vote_score:.6f}")

    # Submit
    preds = le.inverse_transform(vote_test.argmax(axis=1))
    submission = sample.copy(); submission[TARGET] = preds
    submission.to_csv(SUB, index=False)
    print(f"\n  Submission: {SUB}")
    print(f"  Distribution: {dict(zip(*np.unique(preds, return_counts=True)))}")

    # Save
    np.save(OOF_DIR/'v11_rmlp_oof.npy', rmlp_oof.astype('float32'))
    np.save(OOF_DIR/'v11_rmlp_test.npy', rmlp_test.astype('float32'))
    np.save(OOF_DIR/'v11_vote_oof.npy', vote_oof.astype('float32'))
    np.save(OOF_DIR/'v11_vote_test.npy', vote_test.astype('float32'))

    print(f"  ✅ v11 Complete! Vote OOF: {vote_score:.6f}")
    print(f"{'='*60}")
