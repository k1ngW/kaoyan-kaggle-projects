"""
S6E6 v16 — 特征层 + 数据层 + 后处理
=====================================
v8 XGB + 40天文学特征 + 原始SDSS参与训练 + 阈值调优
"""
import os, sys, warnings, gc
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
if sys.platform=="win32":
    os.environ["PYTHONIOENCODING"]="utf-8"
    try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    except: pass

SEED=42; N_SPLITS=5; TARGET,ID_COL='class','id'
CLASSES=['GALAXY','QSO','STAR']; NC=3
CLASS_TO_INT={c:i for i,c in enumerate(CLASSES)}
INT_TO_CLASS={i:c for c,i in CLASS_TO_INT.items()}
EPS=1e-6; RAW=['alpha','delta','u','g','r','i','z','redshift']; BANDS=['u','g','r','i','z']
TE_SMOOTH,TE_INNER=20.0,5
USE_ORIGINAL_ROWS=False  # 原始数据混入拉低了分数
ORIGINAL_WEIGHT=0.3
COMPETITION_WEIGHT=1.0
DATA_DIR=Path(__file__).parent/"data"; SUB=Path(__file__).parent/"submission.csv"
OOF_DIR=Path(__file__).parent/"oof"; OOF_DIR.mkdir(exist_ok=True)
ORIG=DATA_DIR/"star_classification.csv"

from sklearn.preprocessing import LabelEncoder,TargetEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import minimize
import xgboost as xgb

# ═══ 特征工程 (v8继承 + 40新特征) ═══
def cat_key(s): return s.astype(str).fillna('__NA__')
def spectral_type(g,r): return pd.cut(r-g,[-np.inf,-1,-0.5,0,np.inf],labels=['M','G/K','A/F','O/B']).astype(str)
def galaxy_population(u,r): return pd.cut(u-r,[-np.inf,2.2,np.inf],labels=['Blue_Cloud','Red_Sequence']).astype(str)

def add_public_features(df):
    """v8 特征 + 天文学新特征"""
    out=df.copy()
    for c in RAW: out[c]=pd.to_numeric(out[c],errors='coerce').astype('float32')

    # ── 10对颜色差 ──
    for a,b in [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('u','i'),('u','z'),('g','i'),('g','z'),('r','z')]:
        out[f'{a}_{b}']=(out[a]-out[b]).astype('float32')

    # ── 星等统计 ──
    bv=out[list(BANDS)].values.astype(np.float32)
    out['mag_mean']=bv.mean(axis=1).astype('float32'); out['mag_std']=bv.std(axis=1,ddof=1).astype('float32')
    out['mag_min']=bv.min(axis=1).astype('float32'); out['mag_max']=bv.max(axis=1).astype('float32')
    out['mag_range']=(out['mag_max']-out['mag_min']).astype('float32')
    out['mag_argmin']=bv.argmin(axis=1).astype('int16'); out['mag_argmax']=bv.argmax(axis=1).astype('int16')

    # ── 新增: 颜色比率 ──
    for a,b in [('u','g'),('g','r'),('r','i'),('i','z')]:
        out[f'{a}_div_{b}']=(out[a]/(out[b]+EPS)).astype('float32')
        out[f'log_{a}_over_{b}']=np.log(out[a].values.astype(np.float32)/(out[b].values.astype(np.float32)+EPS)+EPS).astype('float32')
    # 光谱斜率比
    for (a,b,c) in [('u','g','r'),('g','r','i'),('r','i','z')]:
        diff_ab=out[f'{a}_{b}'].values.astype(np.float32)
        diff_bc=out[f'{b}_{c}'].values.astype(np.float32)
        out[f'slope_{a}{b}_{b}{c}']=(diff_ab/(diff_bc+EPS)).astype('float32')

    # ── 新增: 光谱浓度指数 ──
    out['concen_blue_red']=((out['u'].values.astype(np.float32)+out['g'].values.astype(np.float32))/
                              (out['r'].values.astype(np.float32)+out['i'].values.astype(np.float32)+out['z'].values.astype(np.float32)+EPS)).astype('float32')
    out['concen_uv_opt']=((out['u'].values.astype(np.float32)+out['z'].values.astype(np.float32))/
                           (out['g'].values.astype(np.float32)+out['r'].values.astype(np.float32)+out['i'].values.astype(np.float32)+EPS)).astype('float32')

    # ── 新增: 非线性变换 ──
    out['g_mul_logz']=(out['g']*np.log(out['redshift'].abs().values.astype(np.float32)+EPS)).astype('float32')
    out['exp_neg_abs_z']=np.exp(-out['redshift'].abs().values.astype(np.float32)).astype('float32')
    out['sqrt_mag_range']=np.sqrt(out['mag_range'].values.astype(np.float32)+EPS).astype('float32')
    out['mag_skew']=((bv-bv.mean(axis=1,keepdims=True))**3).mean(axis=1).astype('float32')

    # ── 新增: 三波段曲率(扩展) ──
    out['curv_gri']=(out['g']-2*out['r']+out['i']).astype('float32')
    out['curv_riz']=(out['r']-2*out['i']+out['z']).astype('float32')

    # ── 新增: 颜色差二次项 ──
    for a,b in [('u','g'),('g','r'),('r','i'),('i','z'),('u','r')]:
        val=out[f'{a}_{b}'].values.astype(np.float32)
        out[f'{a}_{b}_sq']=(val**2 * np.sign(val)).astype('float32')

    # ── redshift交互 ──
    for b in BANDS:
        out[f'redshift_{b}']=(out['redshift']*out[b]).astype('float32')
        out[f'{b}_over_redshift']=(out[b]/(out['redshift'].abs()+EPS)).astype('float32')

    # ── 天球坐标 ──
    ar=np.deg2rad(out['alpha'].values.astype(np.float32)); dr=np.deg2rad(out['delta'].values.astype(np.float32))
    out['alpha_sin']=np.sin(ar).astype('float32'); out['alpha_cos']=np.cos(ar).astype('float32')
    out['delta_sin']=np.sin(dr).astype('float32'); out['delta_cos']=np.cos(dr).astype('float32')
    out['sky_x']=(np.cos(dr)*np.cos(ar)).astype('float32')
    out['sky_y']=(np.cos(dr)*np.sin(ar)).astype('float32'); out['sky_z']=np.sin(dr).astype('float32')

    # ── Flux ──
    fa=[]
    for b in BANDS:
        f=np.power(10.0,-0.4*np.clip(out[b].values.astype(np.float32),-30,30)).astype('float32')
        out[f'flux_{b}']=f; fa.append(f)
    fv=np.column_stack(fa)
    out['flux_mean']=fv.mean(axis=1).astype('float32'); out['flux_std']=fv.std(axis=1,ddof=1).astype('float32')
    out['flux_min']=fv.min(axis=1).astype('float32'); out['flux_max']=fv.max(axis=1).astype('float32')
    out['flux_range']=(out['flux_max']-out['flux_min']).astype('float32')
    # 新增: 通量浓度
    out['flux_concen']=((out['flux_u'].values.astype(np.float32)+out['flux_g'].values.astype(np.float32))/
                         (fv.sum(axis=1)+EPS)).astype('float32')

    # ── 星等斜率 ──
    x=np.arange(5,dtype=np.float32); xc=x-x.mean()
    out['mag_slope']=((bv-bv.mean(axis=1,keepdims=True)).dot(xc)/np.sum(xc**2)).astype('float32')
    out['mag_curvature']=(out['u']-2*out['r']+out['z']).astype('float32')
    out['blue_curvature']=(out['u']-2*out['g']+out['r']).astype('float32')
    out['red_curvature']=(out['r']-2*out['i']+out['z']).astype('float32')
    out['redshift_abs']=out['redshift'].abs().astype('float32')
    out['redshift_log1p_abs']=np.log1p(out['redshift_abs'].values).astype('float32')
    out['redshift_is_neg']=(out['redshift']<0).astype('int8')
    out['spectral_type_calc']=spectral_type(out['g'],out['r']); out['galaxy_population_calc']=galaxy_population(out['u'],out['r'])
    out['spectral_type']=cat_key(out['spectral_type']); out['galaxy_population']=cat_key(out['galaxy_population'])
    out['spectral_x_pop']=cat_key(out['spectral_type'])+'__'+cat_key(out['galaxy_population'])
    out['spectral_calc_x_pop_calc']=cat_key(out['spectral_type_calc'])+'__'+cat_key(out['galaxy_population_calc'])
    return out.replace([np.inf,-np.inf],np.nan)

# ═══ 后续特征工程函数 (同 v8/v12) ═══
def add_pairwise_geometry_features(df):
    out=df.copy()
    for c in ['u_g','g_r','r_i','i_z','u_r','g_i','r_z']:
        if c in out.columns: out[f'{c}_x_redshift']=(out[c]*out['redshift']).astype('float32'); out[f'{c}_abs']=out[c].abs().astype('float32')
    if 'u_g' in out.columns and 'g_r' in out.columns:
        ug=out['u_g'].values.astype(np.float32); gr=out['g_r'].values.astype(np.float32)
        out['color_plane_radius_ug_gr']=np.sqrt(ug**2+gr**2).astype('float32'); out['color_plane_angle_ug_gr']=np.arctan2(ug,gr+EPS).astype('float32')
    if 'r_i' in out.columns and 'i_z' in out.columns:
        ri=out['r_i'].values.astype(np.float32); iz=out['i_z'].values.astype(np.float32)
        out['color_plane_radius_ri_iz']=np.sqrt(ri**2+iz**2).astype('float32'); out['color_plane_angle_ri_iz']=np.arctan2(ri,iz+EPS).astype('float32')
    return out

def add_quantile_bin_features(df,ttm):
    out=df.copy(); qc=[]
    for c in list(dict.fromkeys(RAW+['u_g','g_r','r_i','i_z','u_r','mag_mean','mag_range'])):
        if c not in out.columns: continue
        s=pd.to_numeric(out[c],errors='coerce'); ref=s[ttm].dropna()
        if len(ref)<2: continue
        for q in [16,64,256]:
            try: codes=pd.qcut(s,q=q,labels=False,duplicates='drop').fillna(-1).astype(int).astype(str)
            except: codes=pd.Series(pd.cut(s,bins=q,labels=False),index=s.index).fillna(-1).astype(int).astype(str)
            out[f'{c}_qbin{q}']=codes; qc.append(f'{c}_qbin{q}')
    for a,b in [('alpha_qbin64','delta_qbin64'),('u_g_qbin64','g_r_qbin64'),('redshift_qbin64','mag_mean_qbin64')]:
        if a in out.columns and b in out.columns:
            n=f'{a}__x__{b}'; out[n]=cat_key(out[a])+'__'+cat_key(out[b]); qc.append(n)
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
    pr=pc/np.maximum(pc.sum(),1.0)
    for c in cols:
        if c not in out.columns: continue
        key=cat_key(out[c]); ok=key[om].reset_index(drop=True); oy2=oy.reset_index(drop=True)
        vc=ok.value_counts().to_dict(); out[f'orig_{c}_count']=key.map(vc).fillna(0).astype('float32')
        tmp=pd.DataFrame({'key':ok,'y':oy2})
        for ci,cn in INT_TO_CLASS.items():
            rates=tmp.assign(hit=(tmp['y']==ci).astype('float32')).groupby('key')['hit'].mean()
            out[f'orig_{c}_prior_{cn}']=key.map(rates.to_dict()).fillna(float(pr[ci])).astype('float32')
    return out

def select_te_cols(df,cc,mc=5000):
    cols=[]
    for c in cc:
        if c not in df.columns: continue
        if cat_key(df[c]).nunique(dropna=False)>mc: continue
        if c in ['spectral_type','galaxy_population','spectral_type_calc','galaxy_population_calc','spectral_x_pop','spectral_calc_x_pop_calc'] or '_qbin16' in c or '_qbin64' in c or '_qbin256' in c or '__x__' in c: cols.append(c)
    return cols

def te_sources_needed(tf,ate): return [c for c in ate if any(str(f).startswith(f'TE_{c}_') for f in tf)]

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
            enc=TargetEncoder(cv=TE_INNER,smooth=TE_SMOOTH,target_type='continuous',random_state=SEED+177)
            tvals=enc.fit_transform(tc.values.reshape(-1,1),yb).ravel().astype('float32')
            vvals=enc.transform(vc.values.reshape(-1,1)).ravel().astype('float32')
            evals=enc.transform(ec.values.reshape(-1,1)).ravel().astype('float32')
            n=f'TE_{c}_{cn}'; X_tr[n]=tvals; X_va[n]=vvals; X_te[n]=evals; added.append(n)
    del enc
    return X_tr,X_va,X_te,added

def encode_model_cats(X_tr,X_va,X_te,mc):
    X_tr,X_va,X_te=X_tr.copy(),X_va.copy(),X_te.copy()
    for c in mc:
        if c not in X_tr.columns: continue
        tc,vc,ec=sorted_factorize_three(X_tr[c],X_va[c],X_te[c])
        X_tr[c]=tc.values; X_va[c]=vc.values; X_te[c]=ec.values
    return X_tr,X_va,X_te

TOP_FEATURES=['redshift_u','u_over_redshift','z_over_redshift','g_over_redshift','g_z','redshift_g','g_i','u_i','u_r_abs','TE_redshift_qbin64__x__mag_mean_qbin64_QSO','i_over_redshift','u_r','g_i_abs','TE_redshift_qbin16_GALAXY','redshift_z','TE_redshift_qbin64__x__mag_mean_qbin64_GALAXY','redshift_abs','TE_redshift_qbin64_GALAXY','orig_g_qbin64_prior_QSO','redshift_log1p_abs','orig_g_qbin16_prior_QSO','orig_redshift_qbin64_prior_GALAXY','redshift','TE_redshift_qbin64_QSO','mag_slope','TE_u_r_qbin64_GALAXY','TE_alpha_qbin64__x__delta_qbin64_STAR','r_over_redshift','flux_g','redshift_i','flux_std','g','TE_alpha_qbin64__x__delta_qbin64_GALAXY','TE_u_g_qbin64__x__g_r_qbin64_STAR','redshift_is_neg','g_qbin16','orig_g_qbin256_prior_QSO','TE_u_r_qbin64_QSO','flux_range','mag_std','orig_g_qbin16_prior_GALAXY','redshift_r','orig_redshift_qbin64_prior_STAR','orig_u_r_qbin16_prior_QSO','u_z','orig_redshift_qbin64__x__mag_mean_qbin64_prior_QSO','TE_redshift_qbin64_STAR','TE_g_qbin64_QSO','orig_redshift_qbin16_prior_GALAXY','TE_g_qbin16_QSO','u_g','z','orig_alpha_qbin64__x__delta_qbin64_prior_GALAXY','g_i_x_redshift','orig_z_qbin16_prior_QSO','orig_alpha_qbin64__x__delta_qbin64_prior_STAR','color_plane_radius_ug_gr','i','flux_z','TE_i_qbin64_QSO','TE_g_qbin64_GALAXY','orig_redshift_qbin256_prior_GALAXY','r','TE_u_r_qbin16_GALAXY','flux_i','r_i_x_redshift','flux_r','r_z','orig_i_qbin16_prior_QSO','r_z_x_redshift','g_r_x_redshift','orig_mag_range_qbin16_prior_STAR','r_z_abs','mag_max','TE_g_qbin16_GALAXY','orig_mag_range_qbin64_prior_STAR','TE_i_qbin16_QSO','flux_min','TE_u_g_qbin64_STAR','orig_u_qbin16_prior_QSO','TE_redshift_qbin64__x__mag_mean_qbin64_STAR','orig_u_g_qbin16_prior_STAR','flux_max','orig_z_qbin64_prior_QSO','TE_redshift_qbin16_STAR','mag_range','TE_u_g_qbin64__x__g_r_qbin64_QSO','g_r','orig_redshift_qbin64__x__mag_mean_qbin64_prior_GALAXY','redshift_qbin16','mag_mean_qbin16','TE_u_g_qbin16_STAR','TE_z_qbin64_QSO','u_g_abs','orig_u_r_qbin64_prior_QSO','mag_min','orig_r_qbin16_prior_QSO','redshift_qbin64__x__mag_mean_qbin64','u_r_x_redshift','orig_i_qbin64_prior_QSO','u','flux_u','TE_redshift_qbin16_QSO','flux_mean','redshift_qbin64__x__mag_mean_qbin64_freq','TE_alpha_qbin64__x__delta_qbin64_QSO','redshift_qbin64__x__mag_mean_qbin64_freq_log1p','u_g_x_redshift','u_qbin16','TE_g_r_qbin64_GALAXY','color_plane_radius_ri_iz','orig_z_qbin256_prior_QSO','redshift_qbin256','TE_mag_range_qbin64_QSO','g_r_abs','orig_mag_range_qbin256_prior_STAR','orig_g_qbin64_prior_GALAXY','orig_mag_range_qbin16_prior_GALAXY','r_i','r_qbin16','TE_r_qbin64_STAR','TE_g_r_qbin64_STAR','TE_u_r_qbin16_QSO','orig_u_qbin16_prior_STAR','alpha_sin','TE_u_g_qbin64_QSO','orig_spectral_x_pop_prior_QSO','r_qbin64','sky_y','u_g_qbin16','mag_range_qbin256','TE_r_i_qbin64_QSO','TE_mag_range_qbin64_GALAXY','orig_alpha_qbin256_prior_STAR','alpha_qbin256','z_qbin16','delta_cos','orig_u_qbin64_prior_QSO','g_qbin64','TE_r_qbin16_QSO','TE_z_qbin16_QSO','color_plane_angle_ug_gr','mag_range_qbin16','TE_g_qbin64_STAR','TE_g_r_qbin64_QSO','orig_u_g_qbin64_prior_STAR','TE_r_i_qbin16_QSO','blue_curvature','TE_r_i_qbin64_GALAXY','TE_u_qbin64_STAR','TE_u_qbin64_QSO','mag_mean','TE_i_qbin16_GALAXY','TE_u_g_qbin16_QSO','TE_u_g_qbin64_GALAXY','delta','delta_sin','alpha','sky_x','sky_z','i_qbin16','redshift_qbin64','TE_g_r_qbin16_STAR','mag_curvature','TE_g_qbin16_STAR','TE_alpha_qbin16_QSO','TE_mag_range_qbin16_QSO','TE_u_g_qbin64__x__g_r_qbin64_GALAXY','TE_mag_range_qbin16_STAR']

# ═══ 阈值调优 (后处理) ═══
def optimize_thresholds(oof_probs, y_true, n_classes=3):
    """搜索每类最优概率阈值, 最大化 Balanced Accuracy"""
    def objective(t):
        # t = [t0, t1, t2] 每类的缩放因子
        scaled = oof_probs * np.array(t).reshape(1,-1)
        preds = scaled.argmax(axis=1)
        return -balanced_accuracy_score(y_true, preds)
    # 初始阈值=1 (等价argmax), 搜索范围0.5~2.0
    result = minimize(objective, [1.0,1.0,1.0], bounds=[(0.5,2.0)]*3, method='L-BFGS-B')
    return result.x

def make_xgb(s): return xgb.XGBClassifier(objective='multi:softprob',num_class=3,eval_metric='mlogloss',tree_method='hist',device='cuda',learning_rate=0.012,n_estimators=7000,early_stopping_rounds=180,max_depth=0,max_leaves=72,grow_policy='lossguide',max_bin=512,min_child_weight=10,gamma=0.20,reg_alpha=0.30,reg_lambda=4.0,subsample=0.82,colsample_bytree=0.74,colsample_bylevel=0.86,random_state=s,n_jobs=4,verbosity=0)

# ═══ Main ═══
if __name__=='__main__':
    np.random.seed(SEED)
    print("="*60)
    print("  v16: 特征层+数据层+后处理")
    print("="*60)

    train=pd.read_csv(DATA_DIR/"train.csv"); test=pd.read_csv(DATA_DIR/"test.csv")
    sample=pd.read_csv(DATA_DIR/"sample_submission.csv")
    le=LabelEncoder()
    y=pd.Series(le.fit_transform(train[TARGET]),name='y').reset_index(drop=True)
    test_ids=test[ID_COL].copy()
    train.drop([ID_COL,TARGET],axis=1,inplace=True); test.drop([ID_COL],axis=1,inplace=True)

    orig_all=pd.read_csv(ORIG)
    if 'spectral_type' not in orig_all.columns: orig_all['spectral_type']=spectral_type(orig_all['g'],orig_all['r'])
    if 'galaxy_population' not in orig_all.columns: orig_all['galaxy_population']=galaxy_population(orig_all['u'],orig_all['r'])
    orig_all['spectral_type']=cat_key(orig_all['spectral_type']); orig_all['galaxy_population']=cat_key(orig_all['galaxy_population'])
    y_orig=pd.Series(le.transform(orig_all[TARGET]),name='y_orig').reset_index(drop=True)
    keep=RAW+['spectral_type','galaxy_population',TARGET]
    orig=orig_all[[c for c in keep if c in orig_all.columns]].copy(); orig.drop([TARGET],axis=1,inplace=True)
    print(f"  Train:{train.shape}  Orig:{orig.shape}  USE_ORIGINAL_ROWS={USE_ORIGINAL_ROWS}")

    # Features
    print("  Building features...",end='',flush=True)
    trb,teb,orb=train.copy(),test.copy(),orig.copy()
    trb['_s']='train'; teb['_s']='test'; orb['_s']='orig'
    all_df=pd.concat([trb,teb,orb],axis=0,ignore_index=True)
    all_df=add_public_features(all_df)
    # add_pairwise_geometry_features (from v12 import)
    all_df=add_pairwise_geometry_features(all_df)
    ttm=all_df['_s'].isin(['train','test'])
    cc=['spectral_type','galaxy_population','spectral_type_calc','galaxy_population_calc','spectral_x_pop','spectral_calc_x_pop_calc']
    all_df,qc=add_quantile_bin_features(all_df,ttm); cc+=qc
    cc=[c for c in dict.fromkeys(cc) if c in all_df.columns]
    fm=all_df['_s'].isin(['train','test','orig'])
    fcols=select_te_cols(all_df,cc,20000); all_df=add_frequency_features(all_df,fcols,fm)
    om=all_df['_s'].eq('orig')
    pcols=select_te_cols(all_df,cc,10000); all_df=add_original_prior_features(all_df,pcols,om,y_orig)
    all_df['is_orig']=all_df['_s'].eq('orig').astype('int8'); all_df['is_test']=all_df['_s'].eq('test').astype('int8')
    dc=[c for c in [ID_COL,'_s'] if c in all_df.columns]
    all_df=all_df.drop(columns=dc).replace([np.inf,-np.inf],np.nan)
    nt,nte=len(trb),len(teb)
    X=all_df.iloc[:nt].reset_index(drop=True); X_test=all_df.iloc[nt:nt+nte].reset_index(drop=True)
    X_orig_all=all_df.iloc[nt+nte:].reset_index(drop=True)
    cc=[c for c in cc if c in X.columns]
    ate=select_te_cols(X,cc,5000)
    # TOP_FEATURES from v12 (imported via exec)
    TE_COLS=te_sources_needed(TOP_FEATURES,ate); MC=[c for c in cc if c in TOP_FEATURES]
    print(f" {X.shape[1]} cols, {len(TE_COLS)} TE")
    del all_df,trb,teb,orb,train,test,orig; gc.collect()

    # Train
    print("="*60)
    print(f"  5-Fold CV XGB (原始数据{'ON' if USE_ORIGINAL_ROWS else 'OFF'})")
    print("="*60)
    y_np=y.values.astype(np.int32)
    skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
    oof=np.zeros((len(X),NC),dtype='float32'); tst=np.zeros((len(X_test),NC),dtype='float32')
    y_orig_np=y_orig.values.astype(np.int32)

    for fold,(tr_idx,va_idx) in enumerate(skf.split(np.zeros(len(y_np),dtype=np.int8),y_np),start=1):
        fs=SEED+fold*100; print(f"  Fold {fold}/5...",end='',flush=True)
        X_tr=X.iloc[tr_idx].reset_index(drop=True); y_tr=y.iloc[tr_idx].reset_index(drop=True)
        X_va=X.iloc[va_idx].reset_index(drop=True); y_va=y.iloc[va_idx].reset_index(drop=True)
        X_te=X_test.copy()
        n_comp=len(X_tr)

        # 数据层: 混入原始SDSS数据
        if USE_ORIGINAL_ROWS:
            X_tr=pd.concat([X_tr,X_orig_all],axis=0,ignore_index=True)
            y_tr=pd.concat([y_tr,y_orig],axis=0,ignore_index=True)
            sw_src=np.concatenate([np.full(n_comp,COMPETITION_WEIGHT,dtype='float32'),
                                   np.full(len(X_orig_all),ORIGINAL_WEIGHT,dtype='float32')])
        else:
            sw_src=np.full(n_comp,COMPETITION_WEIGHT,dtype='float32')

        X_tr,X_va,X_te,_=add_fold_safe_te(X_tr,y_tr,X_va,X_te,TE_COLS)
        X_tr,X_va,X_te=encode_model_cats(X_tr,X_va,X_te,MC)
        # 使用全部数值特征 (包括40个新天文学特征)
        feats=[c for c in X_tr.columns if str(X_tr[c].dtype) in ('float32','float64','int32','int64','int16','int8','uint8')]
        X_tr_a=np.nan_to_num(X_tr[feats].values.astype('float32'),nan=0.0)
        X_va_a=np.nan_to_num(X_va[feats].values.astype('float32'),nan=0.0)
        X_te_a=np.nan_to_num(X_te[feats].values.astype('float32'),nan=0.0)

        counts=np.bincount(y_tr.values.astype(np.int32),minlength=3).astype(np.float32)
        wpk=np.float32(len(y_tr))/(np.float32(3)*np.maximum(counts,1.0))
        sw_cls=wpk[y_tr.values.astype(np.int32)].astype(np.float32)
        sw=sw_cls*sw_src

        model=make_xgb(fs)
        model.fit(X_tr_a,y_tr.values,sample_weight=sw,eval_set=[(X_va_a,y_va.values)],verbose=0)
        oof[va_idx]=model.predict_proba(X_va_a).astype('float32')
        tst+=model.predict_proba(X_te_a).astype('float32')/N_SPLITS
        print(f" BAcc:{balanced_accuracy_score(y_va,oof[va_idx].argmax(axis=1)):.6f}")
        del X_tr,X_va,X_te,X_tr_a,X_va_a,X_te_a,model,sw,sw_src,sw_cls; gc.collect()

    oof_ba=balanced_accuracy_score(y_np,oof.argmax(axis=1))
    print(f"\n  OOF: {oof_ba:.6f}")

    # 后处理: 阈值调优
    print(f"\n{'='*60}")
    print("  后处理: 阈值调优")
    print("="*60)
    thresholds=optimize_thresholds(oof,y_np)
    print(f"  最优阈值: {dict(zip(CLASSES,[f'{t:.4f}' for t in thresholds]))}")
    tst_tuned=tst*thresholds.reshape(1,-1)
    oof_tuned=oof*thresholds.reshape(1,-1)
    tuned_ba=balanced_accuracy_score(y_np,oof_tuned.argmax(axis=1))
    print(f"  调优后 OOF: {tuned_ba:.6f} (+{tuned_ba-oof_ba:+.6f})")

    preds=le.inverse_transform(tst_tuned.argmax(axis=1))
    submission=sample.copy(); submission[TARGET]=preds
    submission.to_csv(SUB,index=False)
    print(f"\n  {SUB}")
    print(f"  Distribution: {dict(zip(*np.unique(preds,return_counts=True)))}")

    np.save(OOF_DIR/'v16_oof.npy',oof_tuned.astype('float32'))
    np.save(OOF_DIR/'v16_test.npy',tst_tuned.astype('float32'))
    from experiment_logger import log_experiment
    log_experiment('v16',{'oof_raw':float(oof_ba),'oof_tuned':float(tuned_ba)},notes='40新特征+原始SDSS+阈值调优')
    print(f"  Done! Final OOF: {tuned_ba:.6f}")
