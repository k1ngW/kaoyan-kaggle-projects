"""
S6E6 v8 — Chris Deotte Full Pipeline CPU Port
==============================================
Faithful CPU adaptation of cdeotte/gpu-logistic-regression-stacker.
Reference: ref_notebooks/full_notebook_code.py (downloaded from kaggle kernels pull)

集成技术:
  1. 10对颜色差 + 5波段Flux转换 (10^(-0.4*mag))
  2. 天球坐标 (sin/cos/sky_xyz)
  3. 星等统计 (mean/std/min/max/range/argmin/argmax/slope/curvature)
  4. Redshift交互 (5 band × redshift, 5 band / redshift)
  5. Quantile Binning — 16/64/256 分位数分箱
  6. Crossed Bin Features — alpha×delta, u_g×g_r, redshift×mag_mean
  7. Frequency Encoding — 所有类别列
  8. 原始SDSS prior特征 — spectral_type/galaxy_population × class分布
  9. Pairwise Geometry — color×redshift, abs, color_plane radius/angle
  10. Fold-Safe Target Encoding — 每折内5-fold CV防泄露
  11. XGBoost with Chris Deotte精调参数 (max_leaves=72, lossguide, 7000 trees)
  12. Class Weights + Early Stopping
  13. OOF预测保存 (用于后续stacking/blending)

用法: python optimize_v8.py
"""
import os, sys, warnings, gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except:
        pass

# ═══════════════════════════════════════
# 常量 (from Chris' notebook Cell 2)
# ═══════════════════════════════════════
SEED = 42
N_SPLITS = 5
TARGET = 'class'
ID_COL = 'id'
CLASSES = ['GALAXY', 'QSO', 'STAR']
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}
EPS = 1e-6
RAW_NUM_COLS = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
BANDS = ['u', 'g', 'r', 'i', 'z']

# TE parameters
TE_SMOOTH = 20.0
TE_INNER_SPLITS = 5
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_POWER = 1.0

DATA_DIR = Path(__file__).parent / "data"
SUBMISSION_PATH = Path(__file__).parent / "submission.csv"
ORIG_PATH = DATA_DIR / "star_classification.csv"

# ═══════════════════════════════════════
# Chris Deotte 的 240 TOP_FEATURES (Cell 6)
# ═══════════════════════════════════════
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
    'z_qbin16_freq_log1p', 'orig_alpha_qbin256_prior_GALAXY',
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

# ═══════════════════════════════════════
# 特征工程函数 (CPU port from Chris)
# ═══════════════════════════════════════

def cat_key(s):
    """Convert to string, fill NA (from Chris Cell 3)"""
    return s.astype(str).fillna('__NA__')


def spectral_type(g, r):
    """Spectral type from r-g color (from Chris Cell 3)"""
    return pd.cut(
        r - g,
        [-np.inf, -1, -0.5, 0, np.inf],
        labels=['M', 'G/K', 'A/F', 'O/B'],
    ).astype(str)


def galaxy_population(u, r):
    """Galaxy population from u-r color (from Chris Cell 3)"""
    return pd.cut(
        u - r,
        [-np.inf, 2.2, np.inf],
        labels=['Blue_Cloud', 'Red_Sequence'],
    ).astype(str)


def add_public_features(df):
    """
    Core feature engineering (from Chris Cell 4).
    Ported from GPU (cuDF/cuPy) to CPU (pandas/numpy).
    """
    out = df.copy()
    # Ensure numeric types
    for c in RAW_NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors='coerce').astype('float32')

    # ── 10对颜色差 ──
    color_pairs = [
        ('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'),
        ('u', 'r'), ('u', 'i'), ('u', 'z'), ('g', 'i'),
        ('g', 'z'), ('r', 'z'),
    ]
    for a, b in color_pairs:
        out[f'{a}_{b}'] = (out[a] - out[b]).astype('float32')

    # ── 星等统计 ──
    band_values = out[list(BANDS)].values.astype(np.float32)
    out['mag_mean'] = band_values.mean(axis=1).astype('float32')
    out['mag_std'] = band_values.std(axis=1, ddof=1).astype('float32')
    out['mag_min'] = band_values.min(axis=1).astype('float32')
    out['mag_max'] = band_values.max(axis=1).astype('float32')
    out['mag_range'] = (out['mag_max'] - out['mag_min']).astype('float32')
    out['mag_argmin'] = band_values.argmin(axis=1).astype('int16')
    out['mag_argmax'] = band_values.argmax(axis=1).astype('int16')

    # ── Redshift 交互 ──
    for b in BANDS:
        out[f'redshift_{b}'] = (out['redshift'] * out[b]).astype('float32')
        out[f'{b}_over_redshift'] = (out[b] / (out['redshift'].abs() + EPS)).astype('float32')

    # ── 天球坐标 ──
    alpha_rad = np.deg2rad(out['alpha'].values.astype(np.float32))
    delta_rad = np.deg2rad(out['delta'].values.astype(np.float32))
    out['alpha_sin'] = np.sin(alpha_rad).astype('float32')
    out['alpha_cos'] = np.cos(alpha_rad).astype('float32')
    out['delta_sin'] = np.sin(delta_rad).astype('float32')
    out['delta_cos'] = np.cos(delta_rad).astype('float32')
    out['sky_x'] = (np.cos(delta_rad) * np.cos(alpha_rad)).astype('float32')
    out['sky_y'] = (np.cos(delta_rad) * np.sin(alpha_rad)).astype('float32')
    out['sky_z'] = np.sin(delta_rad).astype('float32')

    # ── Flux转换 10^(-0.4*mag) ──
    flux_arrays = []
    for b in BANDS:
        clipped = np.clip(out[b].values.astype(np.float32), -30, 30)
        flux = np.power(10.0, -0.4 * clipped).astype(np.float32)
        out[f'flux_{b}'] = flux
        flux_arrays.append(flux)
    flux_values = np.column_stack(flux_arrays)
    out['flux_mean'] = flux_values.mean(axis=1).astype('float32')
    out['flux_std'] = flux_values.std(axis=1, ddof=1).astype('float32')
    out['flux_min'] = flux_values.min(axis=1).astype('float32')
    out['flux_max'] = flux_values.max(axis=1).astype('float32')
    out['flux_range'] = (out['flux_max'] - out['flux_min']).astype('float32')

    # ── 星等斜率 (线性回归 slope) ──
    x = np.arange(len(BANDS), dtype=np.float32)
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)
    centered_bands = band_values - band_values.mean(axis=1, keepdims=True)
    out['mag_slope'] = (centered_bands.dot(x_centered) / denom).astype('float32')

    # ── 光谱弯曲度 ──
    out['mag_curvature'] = (out['u'] - 2 * out['r'] + out['z']).astype('float32')
    out['blue_curvature'] = (out['u'] - 2 * out['g'] + out['r']).astype('float32')
    out['red_curvature'] = (out['r'] - 2 * out['i'] + out['z']).astype('float32')

    # ── Redshift辅助 ──
    out['redshift_abs'] = out['redshift'].abs().astype('float32')
    out['redshift_log1p_abs'] = np.log1p(out['redshift_abs'].values).astype('float32')
    out['redshift_is_neg'] = (out['redshift'] < 0).astype('int8')

    # ── 计算的类别特征 ──
    out['spectral_type_calc'] = spectral_type(out['g'], out['r'])
    out['galaxy_population_calc'] = galaxy_population(out['u'], out['r'])

    # ── 原始类别特征字符串化 ──
    out['spectral_type'] = cat_key(out['spectral_type'])
    out['galaxy_population'] = cat_key(out['galaxy_population'])

    # ── 交叉类别 ──
    out['spectral_x_pop'] = cat_key(out['spectral_type']) + '__' + cat_key(out['galaxy_population'])
    out['spectral_calc_x_pop_calc'] = cat_key(out['spectral_type_calc']) + '__' + cat_key(out['galaxy_population_calc'])

    return out.replace([np.inf, -np.inf], np.nan)


def add_pairwise_geometry_features(df):
    """
    Color × redshift interactions + abs values + color plane features.
    (from Chris Cell 4, add_pairwise_geometry_features)
    """
    out = df.copy()
    cols = ['u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'g_i', 'r_z']
    for c in cols:
        if c in out.columns:
            out[f'{c}_x_redshift'] = (out[c] * out['redshift']).astype('float32')
            out[f'{c}_abs'] = out[c].abs().astype('float32')

    # Color plane radius and angle
    if 'u_g' in out.columns and 'g_r' in out.columns:
        ug = out['u_g'].values.astype(np.float32)
        gr = out['g_r'].values.astype(np.float32)
        out['color_plane_radius_ug_gr'] = np.sqrt(ug ** 2 + gr ** 2).astype('float32')
        out['color_plane_angle_ug_gr'] = np.arctan2(ug, gr + EPS).astype('float32')

    if 'r_i' in out.columns and 'i_z' in out.columns:
        ri = out['r_i'].values.astype(np.float32)
        iz = out['i_z'].values.astype(np.float32)
        out['color_plane_radius_ri_iz'] = np.sqrt(ri ** 2 + iz ** 2).astype('float32')
        out['color_plane_angle_ri_iz'] = np.arctan2(ri, iz + EPS).astype('float32')

    return out


def add_quantile_bin_features(df, train_test_mask, q_list=[16, 64, 256]):
    """
    CPU port of Chris' qcut_codes_gpu + add_quantile_bin_features (Cell 4).
    Uses pd.qcut with fallback to pd.cut for speed.
    """
    out = df.copy()
    qbin_cols = []

    # Columns to bin (RAW_NUM_COLS + key color diffs + mag stats)
    qcols = RAW_NUM_COLS.copy()
    for extra in ['u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'mag_mean', 'mag_range']:
        if extra in out.columns:
            qcols.append(extra)

    # Deduplicate
    qcols = list(dict.fromkeys(qcols))

    for c in qcols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors='coerce')
        ref = s[train_test_mask].dropna()
        if len(ref) < 2:
            continue

        for q in q_list:
            name = f'{c}_qbin{q}'
            try:
                # Use pd.qcut for equal-frequency bins
                codes = pd.qcut(s, q=q, labels=False, duplicates='drop')
                codes = codes.fillna(-1).astype(int).astype(str)
            except Exception:
                # Fallback: uniform bins
                codes = pd.cut(s, bins=q, labels=False)
                codes = pd.Series(codes, index=s.index).fillna(-1).astype(int).astype(str)
            out[name] = codes
            qbin_cols.append(name)

    # ── Crossed bin features (from Chris :300-304) ──
    crossed_pairs = [
        ('alpha_qbin64', 'delta_qbin64'),
        ('u_g_qbin64', 'g_r_qbin64'),
        ('redshift_qbin64', 'mag_mean_qbin64'),
    ]
    for a, b in crossed_pairs:
        if a in out.columns and b in out.columns:
            name = f'{a}__x__{b}'
            out[name] = cat_key(out[a]) + '__' + cat_key(out[b])
            qbin_cols.append(name)

    return out, qbin_cols


def add_frequency_features(df, cols, fit_mask):
    """Frequency encoding (from Chris Cell 5)."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = cat_key(out[c])
        vc = s[fit_mask].value_counts(dropna=False)
        out[f'{c}_freq'] = s.map(vc).fillna(0).astype('float32')
        out[f'{c}_freq_log1p'] = np.log1p(out[f'{c}_freq'].values).astype('float32')
    return out


def add_original_prior_features(df, cols, orig_mask, orig_y_values):
    """
    Original SDSS prior features (from Chris Cell 5).
    Computes P(class | category) from the original SDSS dataset.
    """
    out = df.copy()
    orig_mask = orig_mask.astype(bool)
    prior_counts = np.bincount(
        orig_y_values.values.astype(np.int32), minlength=len(CLASSES)
    ).astype(np.float32)
    prior = prior_counts / np.maximum(prior_counts.sum(), 1.0)

    for c in cols:
        if c not in out.columns:
            continue
        key = cat_key(out[c])

        # Build crosstab from original data
        orig_keys = key[orig_mask].reset_index(drop=True)
        orig_y = orig_y_values.reset_index(drop=True)

        # Count per key
        vc = orig_keys.value_counts().to_dict()
        out[f'orig_{c}_count'] = key.map(vc).fillna(0).astype('float32')

        # P(class | key) for each class
        tmp = pd.DataFrame({'key': orig_keys, 'y': orig_y})
        for cls_idx, cls_name in INT_TO_CLASS.items():
            hit = (tmp['y'] == cls_idx).astype('float32')
            rates = tmp.assign(hit=hit).groupby('key')['hit'].mean()
            default_val = float(prior[cls_idx])
            out[f'orig_{c}_prior_{cls_name}'] = key.map(rates.to_dict()).fillna(default_val).astype('float32')

    return out


def select_te_cols(df, cat_cols, max_card=5000):
    """
    Filter categorical columns for Target Encoding.
    (from Chris Cell 5, select_te_cols)
    """
    cols = []
    for c in cat_cols:
        if c not in df.columns:
            continue
        card = cat_key(df[c]).nunique(dropna=False)
        if card > max_card:
            continue
        # Keep: core category columns + bin features + crossed features
        keep = (
            c in ['spectral_type', 'galaxy_population', 'spectral_type_calc',
                   'galaxy_population_calc', 'spectral_x_pop', 'spectral_calc_x_pop_calc']
            or '_qbin16' in c or '_qbin64' in c or '_qbin256' in c
            or '__x__' in c
        )
        if keep:
            cols.append(c)
    return cols


def te_sources_needed_for_top_features(top_features, available_te_cols):
    """Determine which columns need TE based on TOP_FEATURES (from Chris Cell 8)."""
    needed = []
    for c in available_te_cols:
        prefix = f'TE_{c}_'
        if any(str(f).startswith(prefix) for f in top_features):
            needed.append(c)
    return needed


def sorted_factorize_three(train_s, valid_s, test_s):
    """
    Consistent label encoding across train/valid/test.
    (from Chris Cell 8, sorted_factorize_three)
    """
    vals = pd.concat([cat_key(train_s), cat_key(valid_s), cat_key(test_s)], ignore_index=True)
    cats = vals.drop_duplicates().sort_values(ignore_index=True)
    mapper = {v: i for i, v in enumerate(cats)}
    codes = vals.map(mapper).fillna(-1).astype('int32').reset_index(drop=True)
    n_tr, n_va = len(train_s), len(valid_s)
    return (
        codes.iloc[:n_tr].reset_index(drop=True),
        codes.iloc[n_tr:n_tr + n_va].reset_index(drop=True),
        codes.iloc[n_tr + n_va:].reset_index(drop=True),
    )


def add_fold_safe_te(X_train, y_train, X_valid, X_test_fold, te_cols):
    """
    CPU port of Chris' add_fold_safe_te_gpu (Cell 8).
    Uses sklearn TargetEncoder with inner 5-fold CV to prevent data leakage.
    """
    if not te_cols:
        return X_train, X_valid, X_test_fold, []
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test_fold = X_test_fold.copy()
    added = []

    # sklearn 1.4 TargetEncoder uses cv=int; fixed random_state ensures
    # consistent fold splits across all TE columns (equivalent to Chris' fold_ids)
    TE_RANDOM_STATE = SEED + 177

    for c in te_cols:
        if c not in X_train.columns:
            continue
        tr_codes, va_codes, te_codes = sorted_factorize_three(X_train[c], X_valid[c], X_test_fold[c])

        for cls_idx, cls_name in INT_TO_CLASS.items():
            y_bin = (y_train.values == cls_idx).astype('float32')
            encoder = TargetEncoder(
                cv=TE_INNER_SPLITS,
                smooth=TE_SMOOTH,
                target_type='continuous',
                random_state=TE_RANDOM_STATE,
            )
            # fit_transform → OOF encodings for training data (prevents leakage)
            tr_vals = encoder.fit_transform(
                tr_codes.values.reshape(-1, 1), y_bin
            ).ravel().astype('float32')
            # transform → global encoding for validation/test (computed on all train)
            va_vals = encoder.transform(
                va_codes.values.reshape(-1, 1)
            ).ravel().astype('float32')
            te_vals = encoder.transform(
                te_codes.values.reshape(-1, 1)
            ).ravel().astype('float32')

            name = f'TE_{c}_{cls_name}'
            X_train[name] = tr_vals
            X_valid[name] = va_vals
            X_test_fold[name] = te_vals
            added.append(name)

    del encoder
    return X_train, X_valid, X_test_fold, added


def encode_model_categories(X_train, X_valid, X_test_fold, model_cat_cols):
    """Ordinal encode categorical features for XGBoost (from Chris Cell 8)."""
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test_fold = X_test_fold.copy()
    for c in model_cat_cols:
        if c not in X_train.columns:
            continue
        tr_codes, va_codes, te_codes = sorted_factorize_three(X_train[c], X_valid[c], X_test_fold[c])
        X_train[c] = tr_codes.values
        X_valid[c] = va_codes.values
        X_test_fold[c] = te_codes.values
    return X_train, X_valid, X_test_fold


def class_weights(y_series):
    """Compute per-sample class weights (from Chris Cell 9)."""
    counts = np.bincount(y_series.values.astype(np.int32), minlength=len(CLASSES)).astype(np.float32)
    weights_per_class = np.float32(len(y_series)) / (np.float32(len(CLASSES)) * np.maximum(counts, np.float32(1.0)))
    weights = weights_per_class[y_series.values.astype(np.int32)]
    if CLASS_WEIGHT_POWER != 1.0:
        weights = np.power(weights, CLASS_WEIGHT_POWER).astype(np.float32)
    return weights.astype(np.float32)


def make_xgb_params(seed):
    """Chris Deotte's tuned XGBoost parameters (from Chris Cell 9)."""
    return {
        'objective': 'multi:softprob',
        'num_class': len(CLASSES),
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cpu',  # CPU mode
        'learning_rate': 0.012,
        'n_estimators': 7000,
        'early_stopping_rounds': 180,
        'max_depth': 0,
        'max_leaves': 72,
        'grow_policy': 'lossguide',
        'max_bin': 512,
        'min_child_weight': 10,
        'gamma': 0.20,
        'reg_alpha': 0.30,
        'reg_lambda': 4.0,
        'subsample': 0.82,
        'colsample_bytree': 0.74,
        'colsample_bylevel': 0.86,
        'random_state': seed,
        'n_jobs': -1,
        'verbosity': 0,
    }


# ═══════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════
if __name__ == '__main__':
    np.random.seed(SEED)

    # ── 1. 数据加载 ──
    print("=" * 60)
    print("  v8: Chris Deotte CPU Port — 数据加载")
    print("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    print(f"  Train: {train.shape}  Test: {test.shape}")

    # Label encode target
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(train[TARGET]), name='y').reset_index(drop=True)
    test_ids = test[ID_COL].copy()
    train.drop([ID_COL, TARGET], axis=1, inplace=True)
    test.drop([ID_COL], axis=1, inplace=True)

    # ── 2. 加载原始SDSS数据 ──
    print(f"\n  加载原始SDSS数据集: {ORIG_PATH}")
    orig = pd.read_csv(ORIG_PATH)
    # Compute spectral_type and galaxy_population if missing
    if 'spectral_type' not in orig.columns:
        orig['spectral_type'] = spectral_type(orig['g'], orig['r'])
    if 'galaxy_population' not in orig.columns:
        orig['galaxy_population'] = galaxy_population(orig['u'], orig['r'])
    orig['spectral_type'] = cat_key(orig['spectral_type'])
    orig['galaxy_population'] = cat_key(orig['galaxy_population'])

    y_orig = pd.Series(le.transform(orig[TARGET]), name='y_orig').reset_index(drop=True)
    keep = RAW_NUM_COLS + ['spectral_type', 'galaxy_population', TARGET]
    orig = orig[[c for c in keep if c in orig.columns]].copy()
    orig.drop([TARGET], axis=1, inplace=True)
    print(f"  Original SDSS: {orig.shape}")

    # ── 3. 构建特征矩阵 ──
    print(f"\n{'=' * 60}")
    print("  Step 1: 特征工程")
    print("=" * 60)

    # 合并所有数据统一处理
    train_base = train.copy()
    test_base = test.copy()
    orig_base = orig.copy()
    train_base['_source'] = 'train'
    test_base['_source'] = 'test'
    orig_base['_source'] = 'orig'

    all_df = pd.concat([train_base, test_base, orig_base], axis=0, ignore_index=True)

    # Add features
    all_df = add_public_features(all_df)
    all_df = add_pairwise_geometry_features(all_df)

    # Quantile binning (only on competition data)
    train_test_mask = all_df['_source'].isin(['train', 'test'])
    cat_cols = [
        'spectral_type', 'galaxy_population', 'spectral_type_calc',
        'galaxy_population_calc', 'spectral_x_pop', 'spectral_calc_x_pop_calc',
    ]
    all_df, qbin_cols = add_quantile_bin_features(all_df, train_test_mask)
    cat_cols += qbin_cols
    cat_cols = [c for c in dict.fromkeys(cat_cols) if c in all_df.columns]

    # Frequency features
    freq_mask = all_df['_source'].isin(['train', 'test', 'orig'])
    freq_cols = select_te_cols(all_df, cat_cols, max_card=20000)  # Wider for freq
    all_df = add_frequency_features(all_df, freq_cols, freq_mask)
    print(f"  量化分箱列: {len(qbin_cols)}  |  频率编码列: {len(freq_cols)}")

    # Original SDSS prior features
    orig_mask = all_df['_source'].eq('orig')
    prior_cols = select_te_cols(all_df, cat_cols, max_card=10000)
    all_df = add_original_prior_features(all_df, prior_cols, orig_mask, y_orig)
    print(f"  原始prior特征列: {len(prior_cols)}")

    # Marker columns
    all_df['is_orig'] = all_df['_source'].eq('orig').astype('int8')
    all_df['is_test'] = all_df['_source'].eq('test').astype('int8')

    # Clean up
    drop_cols = [c for c in [ID_COL, '_source'] if c in all_df.columns]
    all_df = all_df.drop(columns=drop_cols)
    all_df = all_df.replace([np.inf, -np.inf], np.nan)

    # Split back
    n_train = len(train_base)
    n_test = len(test_base)
    X = all_df.iloc[:n_train].reset_index(drop=True)
    X_test = all_df.iloc[n_train:n_train + n_test].reset_index(drop=True)
    X_orig = all_df.iloc[n_train + n_test:].reset_index(drop=True)
    cat_cols = [c for c in cat_cols if c in X.columns]

    print(f"  特征矩阵: X={X.shape}  X_test={X_test.shape}")
    print(f"  总列数: {X.shape[1]}  |  类别列: {len(cat_cols)}")

    # Cleanup
    del all_df, train_base, test_base, orig_base, train, test, orig
    gc.collect()

    # ── 4. 确定需要 TE 的列 ──
    available_te_cols = select_te_cols(X, cat_cols, max_card=5000)
    TE_COLS = te_sources_needed_for_top_features(TOP_FEATURES, available_te_cols)
    MODEL_CAT_COLS = [c for c in cat_cols if c in TOP_FEATURES]
    print(f"  TE列(裁剪后): {len(TE_COLS)}  |  模型类别列: {len(MODEL_CAT_COLS)}")

    # ── 5. 5折 Cross-Validation 训练 ──
    print(f"\n{'=' * 60}")
    print("  Step 2: 5-Fold CV Training with XGBoost")
    print("=" * 60)

    y_np = y.values.astype(np.int32)
    n_classes = len(CLASSES)
    oof = np.zeros((len(X), n_classes), dtype='float32')
    oof_fold = np.full(len(X), -1, dtype='int8')
    test_pred_sum = np.zeros((len(X_test), n_classes), dtype='float32')

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_np), dtype=np.int8), y_np), start=1):
        fold_seed = SEED + fold * 100
        print(f"\n  --- Fold {fold}/{N_SPLITS} (seed={fold_seed}) ---")

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_va = y.iloc[va_idx].reset_index(drop=True)
        X_te = X_test.copy()

        # Fold-safe Target Encoding
        X_tr, X_va, X_te, added_te = add_fold_safe_te(X_tr, y_tr, X_va, X_te, TE_COLS)
        print(f"    TE features added: {len(added_te)}")

        # Encode category columns
        X_tr, X_va, X_te = encode_model_categories(X_tr, X_va, X_te, MODEL_CAT_COLS)

        # Select features: use TOP_FEATURES (skip missing ones)
        missing = [c for c in TOP_FEATURES if c not in X_tr.columns]
        if missing and fold == 1:
            print(f"    Warning: {len(missing)} TOP_FEATURES not found (first 10: {missing[:10]})")
        features = [c for c in TOP_FEATURES if c in X_tr.columns]
        print(f"    Selected features: {len(features)}/{len(TOP_FEATURES)}")

        X_tr_arr = X_tr[features].values.astype('float32')
        X_va_arr = X_va[features].values.astype('float32')
        X_te_arr = X_te[features].values.astype('float32')

        # Fill NaN
        X_tr_arr = np.nan_to_num(X_tr_arr, nan=0.0)
        X_va_arr = np.nan_to_num(X_va_arr, nan=0.0)
        X_te_arr = np.nan_to_num(X_te_arr, nan=0.0)

        # Class weights
        if USE_CLASS_WEIGHTS:
            sample_weight = class_weights(y_tr)
            eval_weights = class_weights(y_va)
        else:
            sample_weight = None
            eval_weights = None

        # Train XGBoost
        model = xgb.XGBClassifier(**make_xgb_params(fold_seed))
        model.fit(
            X_tr_arr, y_tr.values,
            sample_weight=sample_weight,
            eval_set=[(X_va_arr, y_va.values)],
            sample_weight_eval_set=[eval_weights] if eval_weights is not None else None,
            verbose=250,
        )

        # Predict
        va_probs = model.predict_proba(X_va_arr).astype('float32')
        te_probs = model.predict_proba(X_te_arr).astype('float32')

        oof[va_idx] = va_probs
        oof_fold[va_idx] = fold
        test_pred_sum += te_probs / N_SPLITS

        fold_score = balanced_accuracy_score(y_va, va_probs.argmax(axis=1))
        best_iter = getattr(model, 'best_iteration', None)
        print(f"    Fold {fold} Balanced Acc: {fold_score:.6f}  best_iter={best_iter}")

        fold_scores.append({
            'fold': fold,
            'balanced_accuracy': float(fold_score),
            'best_iteration': int(best_iter) if best_iter is not None else None,
            'n_features': len(features),
            'n_te': len(added_te),
        })

        del model, X_tr, X_va, X_te, X_tr_arr, X_va_arr, X_te_arr
        gc.collect()

    # ── 6. CV Summary ──
    print(f"\n{'=' * 60}")
    print("  Step 3: CV Summary")
    print("=" * 60)

    fs_df = pd.DataFrame(fold_scores)
    print(fs_df.to_string())
    cv_score = balanced_accuracy_score(y_np, oof.argmax(axis=1))
    print(f"\n  OOF Balanced Accuracy: {cv_score:.6f}")

    # ── 7. Submission ──
    print(f"\n{'=' * 60}")
    print("  Step 4: Generate Submission")
    print("=" * 60)

    pred_labels = [INT_TO_CLASS[i] for i in np.argmax(test_pred_sum, axis=1)]
    submission = sample.copy()
    submission[TARGET] = pred_labels
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"  Saved: {SUBMISSION_PATH}")
    print(f"  Distribution: {dict(zip(*np.unique(pred_labels, return_counts=True)))}")

    # ── 8. Save OOF ──
    print(f"\n{'=' * 60}")
    print("  Step 5: Save OOF Predictions")
    print("=" * 60)

    oof_dir = Path(__file__).parent / "oof"
    oof_dir.mkdir(exist_ok=True)
    np.save(oof_dir / 'v8_oof_preds.npy', oof.astype('float32'))
    np.save(oof_dir / 'v8_test_preds.npy', test_pred_sum.astype('float32'))
    np.save(oof_dir / 'v8_oof_labels.npy', y_np.astype('int8'))
    fs_df.to_csv(oof_dir / 'v8_fold_scores.csv', index=False)
    print(f"  OOF files saved to: {oof_dir}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ v8 Complete! OOF: {cv_score:.6f}")
    print(f"{'=' * 60}")
