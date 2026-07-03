# Extracted from Chris Deotte GPU Logistic Regression Stacker
# Kaggle: cdeotte/gpu-logistic-regression-stacker
# Also: cdeotte/realmlp-v1-for-s6e6, cdeotte/xgb-v1-for-s6e6 (forks)
################################################################################


# ======================================================================
# # XGBoost for Playground Series S6E6
# 
# This notebook is a fork of Don's ( @donmarch14 ) great notebook [here][1], then I asked Codex GPT5.5 to improve it.
# 
# For fast feature engineering it uses GPU with NVIDIA cuDF and cuML. And it trains XGBoost using GPU.
# 
# [1]: https://www.kaggle.com/code/donmarch14/s6e6-xgboost
# ======================================================================


# --- Cell 1 ---
import os
import gc
import glob
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import cupy as cp
import cudf
from cuml.preprocessing import TargetEncoder
import xgboost as xgb

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 120)

# Kaggle normally exposes one T4 as cuda:0.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

print('cudf version:', cudf.__version__)
print('xgboost version:', xgb.__version__)



# --- Cell 2 ---
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

# EXP2-060: no original rows are appended to training folds.
# The original dataset is still used to create original-prior features.
USE_ORIGINAL_ROWS = False
ORIGINAL_WEIGHT = np.float32(0.0)
COMPETITION_WEIGHT = np.float32(1.00)
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_POWER = 1.0

TE_SOURCE = 'all'
TE_SMOOTH = 20.0
TE_INNER_SPLITS = 5
TE_MAX_CARDINALITY = 5000
TOP_N_FEATURES = 240

random.seed(SEED)
np.random.seed(SEED)
cp.random.seed(SEED)



# --- Cell 3 ---
def find_competition_root():
    candidates = [
        Path('/kaggle/input/competitions/playground-series-s6e6'),
    ]
    candidates += [Path(p).parent for p in glob.glob('/kaggle/input/*/train.csv')]
    for root in candidates:
        if (root / 'train.csv').exists() and (root / 'test.csv').exists():
            return root
    raise FileNotFoundError('Could not find train.csv and test.csv. Add the competition data to the notebook inputs.')


def find_original_path():
    candidates = [
        Path('/kaggle/input/datasets/fedesoriano/stellar-classification-dataset-sdss17/star_classification.csv'),
    ]
    candidates += [Path(p) for p in glob.glob('/kaggle/input/*/star_classification.csv')]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        'Could not find star_classification.csv. Add the original stellar classification dataset '
        'to the Kaggle notebook inputs for this model.'
    )


def clean_num(s):
    return cudf.to_numeric(s, errors='coerce').astype('float32')


def cat_key(s):
    return s.astype('str').fillna('__NA__')


def spectral_type(g, r):
    return cudf.cut(
        r - g,
        [-np.inf, -1, -0.5, 0, np.inf],
        labels=['M', 'G/K', 'A/F', 'O/B'],
    ).astype('str')


def galaxy_population(u, r):
    return cudf.cut(
        u - r,
        [-np.inf, 2.2, np.inf],
        labels=['Blue_Cloud', 'Red_Sequence'],
    ).astype('str')


def read_competition_csv(path):
    df = cudf.read_csv(str(path))
    for c in RAW_NUM_COLS:
        df[c] = clean_num(df[c])
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype('int32')
    df['spectral_type'] = cat_key(df['spectral_type'])
    df['galaxy_population'] = cat_key(df['galaxy_population'])
    return df


def read_original_csv(path):
    orig = cudf.read_csv(str(path))
    for c in RAW_NUM_COLS:
        orig[c] = clean_num(orig[c])
    if 'spectral_type' not in orig.columns:
        orig['spectral_type'] = spectral_type(orig['g'], orig['r'])
    if 'galaxy_population' not in orig.columns:
        orig['galaxy_population'] = galaxy_population(orig['u'], orig['r'])
    orig['spectral_type'] = cat_key(orig['spectral_type'])
    orig['galaxy_population'] = cat_key(orig['galaxy_population'])
    keep = RAW_NUM_COLS + ['spectral_type', 'galaxy_population', TARGET]
    return orig[[c for c in keep if c in orig.columns]].copy()

DATA_ROOT = find_competition_root()
ORIG_PATH = find_original_path()

train = read_competition_csv(DATA_ROOT / 'train.csv')
test = read_competition_csv(DATA_ROOT / 'test.csv')
orig = read_original_csv(ORIG_PATH)
sample_path = DATA_ROOT / 'sample_submission.csv'
sample = pd.read_csv(sample_path) if sample_path.exists() else None

y = train[TARGET].map(CLASS_TO_INT).astype('int8').reset_index(drop=True)
y_orig = orig[TARGET].map(CLASS_TO_INT).astype('int8').reset_index(drop=True)
test_ids = test[ID_COL].copy()

print('competition root:', DATA_ROOT)
print('original dataset:', ORIG_PATH)
print('train/test/original:', train.shape, test.shape, orig.shape)
print(train[TARGET].value_counts(normalize=True).sort_index().to_pandas())



# --- Cell 4 ---
def assign_cupy(df, name, values, dtype='float32'):
    df[name] = cudf.Series(values.astype(dtype), index=df.index)


def add_public_features(df):
    out = df.copy()
    for c in RAW_NUM_COLS:
        out[c] = clean_num(out[c])

    color_pairs = [
        ('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'),
        ('u', 'r'), ('u', 'i'), ('u', 'z'), ('g', 'i'),
        ('g', 'z'), ('r', 'z'),
    ]
    for a, b in color_pairs:
        out[f'{a}_{b}'] = (out[a] - out[b]).astype('float32')

    band_values = out[BANDS].to_cupy().astype(cp.float32)
    assign_cupy(out, 'mag_mean', cp.mean(band_values, axis=1))
    assign_cupy(out, 'mag_std', cp.std(band_values, axis=1, ddof=1))
    assign_cupy(out, 'mag_min', cp.min(band_values, axis=1))
    assign_cupy(out, 'mag_max', cp.max(band_values, axis=1))
    out['mag_range'] = (out['mag_max'] - out['mag_min']).astype('float32')
    out['mag_argmin'] = cudf.Series(cp.argmin(band_values, axis=1).astype(cp.int16), index=out.index)
    out['mag_argmax'] = cudf.Series(cp.argmax(band_values, axis=1).astype(cp.int16), index=out.index)

    for b in BANDS:
        out[f'redshift_{b}'] = (out['redshift'] * out[b]).astype('float32')
        out[f'{b}_over_redshift'] = (out[b] / (out['redshift'].abs() + EPS)).astype('float32')

    alpha_rad = out['alpha'].to_cupy().astype(cp.float32) * np.float32(np.pi / 180.0)
    delta_rad = out['delta'].to_cupy().astype(cp.float32) * np.float32(np.pi / 180.0)
    assign_cupy(out, 'alpha_sin', cp.sin(alpha_rad))
    assign_cupy(out, 'alpha_cos', cp.cos(alpha_rad))
    assign_cupy(out, 'delta_sin', cp.sin(delta_rad))
    assign_cupy(out, 'delta_cos', cp.cos(delta_rad))
    assign_cupy(out, 'sky_x', cp.cos(delta_rad) * cp.cos(alpha_rad))
    assign_cupy(out, 'sky_y', cp.cos(delta_rad) * cp.sin(alpha_rad))
    assign_cupy(out, 'sky_z', cp.sin(delta_rad))

    flux_arrays = []
    for b in BANDS:
        clipped = cp.clip(out[b].to_cupy().astype(cp.float32), -30, 30)
        flux = cp.power(cp.float32(10.0), cp.float32(-0.4) * clipped).astype(cp.float32)
        assign_cupy(out, f'flux_{b}', flux)
        flux_arrays.append(flux)
    flux_values = cp.vstack(flux_arrays).T
    assign_cupy(out, 'flux_mean', cp.mean(flux_values, axis=1))
    assign_cupy(out, 'flux_std', cp.std(flux_values, axis=1, ddof=1))
    assign_cupy(out, 'flux_min', cp.min(flux_values, axis=1))
    assign_cupy(out, 'flux_max', cp.max(flux_values, axis=1))
    out['flux_range'] = (out['flux_max'] - out['flux_min']).astype('float32')

    x = cp.arange(len(BANDS), dtype=cp.float32)
    x_centered = x - x.mean()
    denom = cp.sum(x_centered ** 2)
    centered_bands = band_values - cp.mean(band_values, axis=1, keepdims=True)
    assign_cupy(out, 'mag_slope', centered_bands.dot(x_centered) / denom)
    out['mag_curvature'] = (out['u'] - 2 * out['r'] + out['z']).astype('float32')
    out['blue_curvature'] = (out['u'] - 2 * out['g'] + out['r']).astype('float32')
    out['red_curvature'] = (out['r'] - 2 * out['i'] + out['z']).astype('float32')

    out['redshift_abs'] = out['redshift'].abs().astype('float32')
    out['redshift_log1p_abs'] = cudf.Series(cp.log1p(out['redshift_abs'].to_cupy()).astype(cp.float32), index=out.index)
    out['redshift_is_neg'] = (out['redshift'] < 0).astype('int8')
    out['spectral_type_calc'] = spectral_type(out['g'], out['r'])
    out['galaxy_population_calc'] = galaxy_population(out['u'], out['r'])
    out['spectral_type'] = cat_key(out['spectral_type'])
    out['galaxy_population'] = cat_key(out['galaxy_population'])
    out['spectral_x_pop'] = cat_key(out['spectral_type']) + '__' + cat_key(out['galaxy_population'])
    out['spectral_calc_x_pop_calc'] = cat_key(out['spectral_type_calc']) + '__' + cat_key(out['galaxy_population_calc'])
    return out.replace([np.inf, -np.inf], np.nan)


def add_pairwise_geometry_features(df):
    out = df.copy()
    cols = ['u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'g_i', 'r_z']
    cols = [c for c in cols if c in out.columns]
    for c in cols:
        out[f'{c}_x_redshift'] = (out[c] * out['redshift']).astype('float32')
        out[f'{c}_abs'] = out[c].abs().astype('float32')
    if {'u_g', 'g_r'}.issubset(out.columns):
        ug = out['u_g'].to_cupy().astype(cp.float32)
        gr = out['g_r'].to_cupy().astype(cp.float32)
        assign_cupy(out, 'color_plane_radius_ug_gr', cp.sqrt(ug ** 2 + gr ** 2))
        assign_cupy(out, 'color_plane_angle_ug_gr', cp.arctan2(ug, gr + EPS))
    if {'r_i', 'i_z'}.issubset(out.columns):
        ri = out['r_i'].to_cupy().astype(cp.float32)
        iz = out['i_z'].to_cupy().astype(cp.float32)
        assign_cupy(out, 'color_plane_radius_ri_iz', cp.sqrt(ri ** 2 + iz ** 2))
        assign_cupy(out, 'color_plane_angle_ri_iz', cp.arctan2(ri, iz + EPS))
    return out


def qcut_codes_gpu(values, ref_values, q):
    ref = ref_values.dropna()
    if len(ref) < 2:
        return cp.full(len(values), -1, dtype=cp.int16)
    probs = cp.linspace(0, 1, q + 1, dtype=cp.float32)
    bins = cp.asarray(ref.quantile(probs).values, dtype=cp.float32)
    bins = cp.unique(bins)
    vals = values.to_cupy().astype(cp.float32)
    if len(bins) <= 1:
        return cp.full(len(vals), -1, dtype=cp.int16)
    codes = cp.searchsorted(bins, vals, side='left') - 1
    codes = cp.where(vals == bins[0], 0, codes)
    codes = cp.where((vals < bins[0]) | (vals > bins[-1]) | cp.isnan(vals), -1, codes)
    codes = cp.clip(codes, -1, len(bins) - 2).astype(cp.int16)
    return codes


def add_quantile_bin_features(df, train_test_mask):
    out = df.copy()
    qbin_cols = []
    cols = RAW_NUM_COLS + [c for c in ['u_g', 'g_r', 'r_i', 'i_z', 'u_r', 'mag_mean', 'mag_range'] if c in out.columns]
    for c in cols:
        s = clean_num(out[c])
        ref = s[train_test_mask]
        for q in [16, 64, 256]:
            name = f'{c}_qbin{q}'
            codes = qcut_codes_gpu(s, ref, q)
            out[name] = cudf.Series(codes, index=out.index).astype('int16').astype('str')
            qbin_cols.append(name)
    for a, b in [('alpha_qbin64', 'delta_qbin64'), ('u_g_qbin64', 'g_r_qbin64'), ('redshift_qbin64', 'mag_mean_qbin64')]:
        if a in out.columns and b in out.columns:
            name = f'{a}__x__{b}'
            out[name] = cat_key(out[a]) + '__' + cat_key(out[b])
            qbin_cols.append(name)
    return out, qbin_cols



# --- Cell 5 ---
def select_te_cols(df, cat_cols, source, max_card):
    cols = []
    for c in cat_cols:
        if c not in df.columns:
            continue
        card = int(cat_key(df[c]).nunique(dropna=False))
        if card > max_card:
            continue
        if source == 'core':
            keep = (
                c in ['spectral_type', 'galaxy_population', 'spectral_type_calc', 'galaxy_population_calc', 'spectral_x_pop', 'spectral_calc_x_pop_calc']
                or c.endswith('_floor_cat')
                or c.endswith('_qbin16')
                or c.endswith('_qbin64')
                or '__x__' in c
            )
        else:
            keep = True
        if keep:
            cols.append(c)
    return cols


def add_frequency_features(df, cols, fit_mask):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = cat_key(out[c])
        vc = s[fit_mask].value_counts(dropna=False)
        out[f'{c}_freq'] = s.map(vc).fillna(0).astype('float32')
        out[f'{c}_freq_log1p'] = cudf.Series(cp.log1p(out[f'{c}_freq'].to_cupy()).astype(cp.float32), index=out.index)
    return out


def add_original_prior_features(df, cols, orig_mask, orig_y_values):
    out = df.copy()
    orig_mask = orig_mask.astype('bool')
    prior_counts = cp.bincount(orig_y_values.to_cupy().astype(cp.int32), minlength=len(CLASSES)).astype(cp.float32)
    prior = prior_counts / cp.maximum(prior_counts.sum(), cp.float32(1.0))

    for c in cols:
        if c not in out.columns:
            continue
        key = cat_key(out[c])
        tmp = cudf.DataFrame({
            'key': key[orig_mask].reset_index(drop=True),
            'y': orig_y_values.reset_index(drop=True),
        })
        counts = tmp.groupby('key').size()
        out[f'orig_{c}_count'] = key.map(counts).fillna(0).astype('float32')
        for cls_idx, cls_name in INT_TO_CLASS.items():
            hit = (tmp['y'] == cls_idx).astype('float32')
            rates = tmp.assign(hit=hit).groupby('key')['hit'].mean()
            out[f'orig_{c}_prior_{cls_name}'] = key.map(rates).fillna(float(prior[cls_idx].get())).astype('float32')
    return out



# --- Cell 6 ---
TOP_FEATURES = [
    'redshift_u',
    'u_over_redshift',
    'z_over_redshift',
    'g_over_redshift',
    'g_z',
    'redshift_g',
    'g_i',
    'u_i',
    'u_r_abs',
    'TE_redshift_qbin64__x__mag_mean_qbin64_QSO',
    'i_over_redshift',
    'u_r',
    'g_i_abs',
    'TE_redshift_qbin16_GALAXY',
    'redshift_z',
    'TE_redshift_qbin64__x__mag_mean_qbin64_GALAXY',
    'redshift_abs',
    'TE_redshift_qbin64_GALAXY',
    'orig_g_qbin64_prior_QSO',
    'redshift_log1p_abs',
    'orig_g_qbin16_prior_QSO',
    'orig_redshift_qbin64_prior_GALAXY',
    'redshift',
    'TE_redshift_qbin64_QSO',
    'mag_slope',
    'TE_u_r_qbin64_GALAXY',
    'TE_alpha_qbin64__x__delta_qbin64_STAR',
    'r_over_redshift',
    'flux_g',
    'redshift_i',
    'flux_std',
    'g',
    'TE_alpha_qbin64__x__delta_qbin64_GALAXY',
    'TE_u_g_qbin64__x__g_r_qbin64_STAR',
    'redshift_is_neg',
    'g_qbin16',
    'orig_g_qbin256_prior_QSO',
    'TE_u_r_qbin64_QSO',
    'flux_range',
    'mag_std',
    'orig_g_qbin16_prior_GALAXY',
    'redshift_r',
    'orig_redshift_qbin64_prior_STAR',
    'orig_u_r_qbin16_prior_QSO',
    'u_z',
    'orig_redshift_qbin64__x__mag_mean_qbin64_prior_QSO',
    'TE_redshift_qbin64_STAR',
    'TE_g_qbin64_QSO',
    'orig_redshift_qbin16_prior_GALAXY',
    'TE_g_qbin16_QSO',
    'u_g',
    'z',
    'orig_alpha_qbin64__x__delta_qbin64_prior_GALAXY',
    'g_i_x_redshift',
    'orig_z_qbin16_prior_QSO',
    'orig_alpha_qbin64__x__delta_qbin64_prior_STAR',
    'color_plane_radius_ug_gr',
    'i',
    'flux_z',
    'TE_i_qbin64_QSO',
    'TE_g_qbin64_GALAXY',
    'orig_redshift_qbin256_prior_GALAXY',
    'r',
    'TE_u_r_qbin16_GALAXY',
    'flux_i',
    'r_i_x_redshift',
    'flux_r',
    'r_z',
    'orig_i_qbin16_prior_QSO',
    'r_z_x_redshift',
    'g_r_x_redshift',
    'orig_mag_range_qbin16_prior_STAR',
    'r_z_abs',
    'mag_max',
    'TE_g_qbin16_GALAXY',
    'orig_mag_range_qbin64_prior_STAR',
    'TE_i_qbin16_QSO',
    'flux_min',
    'TE_u_g_qbin64_STAR',
    'orig_u_qbin16_prior_QSO',
    'TE_redshift_qbin64__x__mag_mean_qbin64_STAR',
    'orig_u_g_qbin16_prior_STAR',
    'flux_max',
    'orig_z_qbin64_prior_QSO',
    'TE_redshift_qbin16_STAR',
    'mag_range',
    'TE_u_g_qbin64__x__g_r_qbin64_QSO',
    'orig_redshift_qbin256_count',
    'g_r',
    'orig_redshift_qbin64__x__mag_mean_qbin64_prior_GALAXY',
    'orig_i_qbin256_prior_QSO',
    'redshift_qbin16',
    'mag_mean_qbin16',
    'TE_u_g_qbin16_STAR',
    'TE_z_qbin64_QSO',
    'u_g_abs',
    'orig_u_r_qbin64_prior_QSO',
    'mag_min',
    'orig_r_qbin16_prior_QSO',
    'redshift_qbin64__x__mag_mean_qbin64',
    'u_r_x_redshift',
    'orig_i_qbin64_prior_QSO',
    'u',
    'flux_u',
    'TE_redshift_qbin16_QSO',
    'flux_mean',
    'redshift_qbin64__x__mag_mean_qbin64_freq',
    'TE_alpha_qbin64__x__delta_qbin64_QSO',
    'redshift_qbin64__x__mag_mean_qbin64_freq_log1p',
    'u_g_x_redshift',
    'u_qbin16',
    'TE_g_r_qbin64_GALAXY',
    'color_plane_radius_ri_iz',
    'orig_z_qbin256_prior_QSO',
    'redshift_qbin256',
    'TE_mag_range_qbin64_QSO',
    'g_r_abs',
    'orig_mag_range_qbin256_prior_STAR',
    'orig_g_qbin64_prior_GALAXY',
    'orig_mag_range_qbin16_prior_GALAXY',
    'r_i',
    'r_qbin16',
    'TE_r_qbin64_STAR',
    'TE_g_r_qbin64_STAR',
    'TE_u_r_qbin16_QSO',
    'orig_u_qbin16_prior_STAR',
    'z_qbin16_freq_log1p',
    'orig_alpha_qbin256_prior_GALAXY',
    'alpha_sin',
    'TE_u_g_qbin64_QSO',
    'orig_spectral_x_pop_prior_QSO',
    'r_qbin64',
    'sky_y',
    'u_g_qbin16',
    'mag_range_qbin256',
    'TE_r_i_qbin64_QSO',
    'TE_mag_range_qbin64_GALAXY',
    'orig_alpha_qbin256_prior_STAR',
    'alpha_qbin256',
    'orig_u_qbin16_count',
    'z_qbin16',
    'delta_cos',
    'orig_u_qbin64_prior_QSO',
    'g_qbin64',
    'TE_r_qbin16_QSO',
    'TE_z_qbin16_QSO',
    'color_plane_angle_ug_gr',
    'mag_range_qbin16',
    'u_qbin16_freq_log1p',
    'orig_delta_qbin256_prior_STAR',
    'TE_g_qbin64_STAR',
    'orig_delta_qbin256_prior_GALAXY',
    'z_qbin16_freq',
    'orig_u_g_qbin64_prior_STAR',
    'orig_mag_range_qbin256_prior_GALAXY',
    'u_qbin16_freq',
    'orig_z_qbin16_count',
    'orig_u_r_qbin256_prior_QSO',
    'orig_g_qbin256_prior_GALAXY',
    'TE_r_i_qbin16_QSO',
    'orig_i_qbin16_prior_GALAXY',
    'TE_u_qbin16_QSO',
    'TE_r_i_qbin16_GALAXY',
    'blue_curvature',
    'TE_r_i_qbin64_GALAXY',
    'orig_redshift_qbin64__x__mag_mean_qbin64_prior_STAR',
    'TE_u_qbin64_STAR',
    'TE_g_r_qbin64_QSO',
    'TE_u_qbin64_QSO',
    'mag_mean',
    'TE_i_qbin16_GALAXY',
    'TE_u_g_qbin16_QSO',
    'TE_u_g_qbin64_GALAXY',
    'orig_mag_range_qbin256_prior_QSO',
    'delta',
    'delta_sin',
    'alpha',
    'sky_x',
    'orig_g_qbin64_prior_STAR',
    'sky_z',
    'orig_r_qbin64_prior_QSO',
    'i_qbin16',
    'redshift_qbin64',
    'orig_alpha_qbin64__x__delta_qbin64_prior_QSO',
    'TE_g_r_qbin16_STAR',
    'mag_curvature',
    'orig_r_qbin256_prior_QSO',
    'TE_g_qbin16_STAR',
    'TE_alpha_qbin16_QSO',
    'orig_u_r_qbin16_prior_STAR',
    'TE_mag_range_qbin16_QSO',
    'orig_r_i_qbin16_prior_GALAXY',
    'orig_alpha_qbin16_prior_STAR',
    'orig_delta_qbin16_prior_STAR',
    'orig_u_qbin256_prior_QSO',
    'orig_g_qbin256_prior_STAR',
    'orig_r_i_qbin64_prior_QSO',
    'orig_u_g_qbin16_prior_QSO',
    'orig_mag_range_qbin64_prior_GALAXY',
    'orig_u_g_qbin256_prior_QSO',
    'redshift_qbin64_freq',
    'u_r_qbin16',
    'orig_u_r_qbin256_prior_GALAXY',
    'orig_u_g_qbin64_prior_QSO',
    'orig_alpha_qbin256_prior_QSO',
    'orig_u_g_qbin256_prior_STAR',
    'TE_mag_range_qbin16_STAR',
    'TE_u_g_qbin64__x__g_r_qbin64_GALAXY',
    'orig_g_r_qbin256_prior_GALAXY',
    'orig_alpha_qbin256_count',
    'redshift_qbin64_freq_log1p',
    'orig_delta_qbin256_prior_QSO',
    'orig_u_qbin256_prior_GALAXY',
    'orig_r_i_qbin256_prior_QSO',
    'orig_redshift_qbin64__x__mag_mean_qbin64_count',
    'orig_r_i_qbin16_prior_STAR',
    'u_r_qbin256',
    'orig_r_qbin256_prior_GALAXY',
    'orig_r_i_qbin256_prior_GALAXY',
    'orig_mag_mean_qbin256_prior_QSO',
    'orig_u_g_qbin64__x__g_r_qbin64_prior_GALAXY',
    'orig_u_r_qbin256_prior_STAR',
    'orig_z_qbin256_prior_GALAXY',
    'orig_i_qbin256_prior_GALAXY',
    'orig_u_g_qbin64__x__g_r_qbin64_prior_STAR',
    'orig_redshift_qbin256_prior_QSO',
    'orig_u_g_qbin256_prior_GALAXY',
    'orig_u_qbin256_prior_STAR',
    'orig_mag_mean_qbin256_prior_GALAXY',
    'orig_spectral_calc_x_pop_calc_count',
    'orig_r_i_qbin256_prior_STAR',
    'orig_i_qbin256_prior_STAR',
    'orig_spectral_x_pop_prior_STAR',
    'orig_r_qbin256_prior_STAR',
    'orig_delta_qbin256_count',
    'orig_g_r_qbin256_prior_QSO',
    'orig_z_qbin256_count',
    'orig_g_qbin256_count',
    'orig_r_qbin256_count',
    'orig_mag_mean_qbin256_count',
]
print('selected features:', len(TOP_FEATURES))



# --- Cell 7 ---
def build_feature_matrix(train, test, orig):
    train_base = train.drop(columns=[TARGET]).copy()
    test_base = test.copy()
    orig_base = orig.drop(columns=[TARGET]).copy()
    train_base['_source'] = 'train'
    test_base['_source'] = 'test'
    orig_base['_source'] = 'orig'

    all_df = cudf.concat([train_base, test_base, orig_base], axis=0, ignore_index=True)
    all_df = add_public_features(all_df)
    all_df = add_pairwise_geometry_features(all_df)

    train_test_mask = all_df['_source'].isin(['train', 'test'])
    cat_cols = [
        'spectral_type', 'galaxy_population', 'spectral_type_calc', 'galaxy_population_calc',
        'spectral_x_pop', 'spectral_calc_x_pop_calc',
    ]
    all_df, qbin_cols = add_quantile_bin_features(all_df, train_test_mask)
    cat_cols += qbin_cols
    cat_cols = [c for c in dict.fromkeys(cat_cols) if c in all_df.columns]

    freq_fit_mask = all_df['_source'].isin(['train', 'test', 'orig'])
    freq_cols = select_te_cols(all_df, cat_cols, TE_SOURCE, max_card=TE_MAX_CARDINALITY * 4)
    all_df = add_frequency_features(all_df, freq_cols, freq_fit_mask)

    orig_mask = all_df['_source'].eq('orig')
    prior_cols = select_te_cols(all_df, cat_cols, TE_SOURCE, max_card=TE_MAX_CARDINALITY * 2)
    all_df = add_original_prior_features(all_df, prior_cols, orig_mask, y_orig)

    all_df['is_orig'] = all_df['_source'].eq('orig').astype('int8')
    all_df['is_test'] = all_df['_source'].eq('test').astype('int8')

    all_df = all_df.drop(columns=[c for c in [ID_COL, '_source'] if c in all_df.columns])
    all_df = all_df.replace([np.inf, -np.inf], np.nan)

    n_train = len(train_base)
    n_test = len(test_base)
    X = all_df.iloc[:n_train].reset_index(drop=True)
    X_test = all_df.iloc[n_train:n_train + n_test].reset_index(drop=True)
    X_orig = all_df.iloc[n_train + n_test:].reset_index(drop=True)
    cat_cols = [c for c in cat_cols if c in X.columns]

    del all_df, train_base, test_base, orig_base
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return X, X_test, X_orig, cat_cols

X, X_test, X_orig, cat_cols = build_feature_matrix(train, test, orig)
print('base matrices:', X.shape, X_test.shape, X_orig.shape, 'cat_cols:', len(cat_cols))

# EXP2-060 does not append original rows during fold training, so we can release
# the original feature matrix after its prior features have been created.
if not USE_ORIGINAL_ROWS:
    del X_orig
    X_orig = None

del train, test, orig
gc.collect()
cp.get_default_memory_pool().free_all_blocks()



# --- Cell 8 ---
def sorted_factorize_three(train_s, valid_s, test_s):
    vals = cudf.concat([cat_key(train_s), cat_key(valid_s), cat_key(test_s)], ignore_index=True)
    cats = vals.drop_duplicates().sort_values(ignore_index=True)
    mapper = cudf.Series(cp.arange(len(cats), dtype=cp.int32), index=cats)
    codes = vals.map(mapper).fillna(-1).astype('int32').reset_index(drop=True)
    n_tr, n_va = len(train_s), len(valid_s)
    return (
        codes.iloc[:n_tr].reset_index(drop=True),
        codes.iloc[n_tr:n_tr + n_va].reset_index(drop=True),
        codes.iloc[n_tr + n_va:].reset_index(drop=True),
    )


def make_inner_fold_ids(y_train):
    y_cpu = y_train.to_numpy()
    fold_ids = np.empty(len(y_cpu), dtype=np.int32)
    inner = StratifiedKFold(n_splits=TE_INNER_SPLITS, shuffle=True, random_state=SEED + 177)
    for fold_id, (_, va_idx) in enumerate(inner.split(np.zeros(len(y_cpu), dtype=np.int8), y_cpu)):
        fold_ids[va_idx] = fold_id
    return cp.asarray(fold_ids, dtype=cp.int32)


def add_fold_safe_te_gpu(X_train, y_train, X_valid, X_test_fold, te_cols):
    if not te_cols:
        return X_train, X_valid, X_test_fold, []
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test_fold = X_test_fold.copy()
    fold_ids = make_inner_fold_ids(y_train)
    y_cp = y_train.to_cupy().astype(cp.int32)
    added = []

    for c in te_cols:
        if c not in X_train.columns:
            continue
        tr_codes, va_codes, te_codes = sorted_factorize_three(X_train[c], X_valid[c], X_test_fold[c])
        tr_cp = tr_codes.to_cupy().astype(cp.int32)
        va_cp = va_codes.to_cupy().astype(cp.int32)
        te_cp = te_codes.to_cupy().astype(cp.int32)
        for cls_idx, cls_name in INT_TO_CLASS.items():
            y_bin = (y_cp == cls_idx).astype(cp.float32)
            encoder = TargetEncoder(
                n_folds=TE_INNER_SPLITS,
                smooth=TE_SMOOTH,
                seed=SEED + 177,
                split_method='customize',
                output_type='cupy',
            )
            tr_vals = cp.asarray(encoder.fit_transform(tr_cp, y_bin, fold_ids=fold_ids)).ravel().astype(cp.float32)
            va_vals = cp.asarray(encoder.transform(va_cp)).ravel().astype(cp.float32)
            te_vals = cp.asarray(encoder.transform(te_cp)).ravel().astype(cp.float32)
            name = f'TE_{c}_{cls_name}'
            X_train[name] = cudf.Series(tr_vals)
            X_valid[name] = cudf.Series(va_vals)
            X_test_fold[name] = cudf.Series(te_vals)
            added.append(name)
            del encoder, tr_vals, va_vals, te_vals, y_bin
        del tr_codes, va_codes, te_codes, tr_cp, va_cp, te_cp
    cp.get_default_memory_pool().free_all_blocks()
    return X_train, X_valid, X_test_fold, added


def encode_model_categories_gpu(X_train, X_valid, X_test_fold, model_cat_cols):
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test_fold = X_test_fold.copy()
    for c in model_cat_cols:
        if c not in X_train.columns:
            continue
        tr_codes, va_codes, te_codes = sorted_factorize_three(X_train[c], X_valid[c], X_test_fold[c])
        X_train[c] = tr_codes
        X_valid[c] = va_codes
        X_test_fold[c] = te_codes
    return X_train, X_valid, X_test_fold


def te_sources_needed_for_top_features(top_features, available_te_cols):
    needed = []
    for c in available_te_cols:
        prefix = f'TE_{c}_'
        if any(str(f).startswith(prefix) for f in top_features):
            needed.append(c)
    return needed

available_te_cols = select_te_cols(X, cat_cols, TE_SOURCE, TE_MAX_CARDINALITY)
TE_COLS = te_sources_needed_for_top_features(TOP_FEATURES, available_te_cols)
MODEL_CAT_COLS = [c for c in cat_cols if c in TOP_FEATURES]
print(f'target-encoding sources pruned: {len(available_te_cols)} -> {len(TE_COLS)}')
print(f'raw categorical features selected for model: {len(MODEL_CAT_COLS)}')
print(TE_COLS[:25])



# --- Cell 9 ---
def to_numpy(a):
    if isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def balanced_error_metric(y_true, y_pred, sample_weight=None):
    y_true_np = to_numpy(y_true).astype(int)
    y_pred_np = to_numpy(y_pred)
    if y_pred_np.ndim == 1:
        y_pred_np = y_pred_np.reshape(-1, len(CLASSES))
    return 1.0 - balanced_accuracy_score(y_true_np, np.argmax(y_pred_np, axis=1))


def make_xgb_params(seed):
    return {
        'objective': 'multi:softprob',
        'num_class': len(CLASSES),
        'eval_metric': balanced_error_metric,
        'tree_method': 'hist',
        'device': 'cuda',
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
        'n_jobs': 4,
    }


def source_weights_gpu(n_comp_train, n_orig):
    if n_orig == 0:
        return cp.ones(n_comp_train, dtype=cp.float32) * COMPETITION_WEIGHT
    return cp.concatenate([
        cp.ones(n_comp_train, dtype=cp.float32) * COMPETITION_WEIGHT,
        cp.ones(n_orig, dtype=cp.float32) * ORIGINAL_WEIGHT,
    ])


def class_weights_gpu(y_series):
    y_cp = y_series.to_cupy().astype(cp.int32)
    counts = cp.bincount(y_cp, minlength=len(CLASSES)).astype(cp.float32)
    weights_per_class = cp.float32(len(y_cp)) / (cp.float32(len(CLASSES)) * cp.maximum(counts, cp.float32(1.0)))
    weights = weights_per_class[y_cp]
    if CLASS_WEIGHT_POWER != 1.0:
        weights = cp.power(weights, CLASS_WEIGHT_POWER).astype(cp.float32)
    return weights.astype(cp.float32)


def prepare_selected_cupy(X_train, X_valid, X_test_fold):
    X_train, X_valid, X_test_fold = encode_model_categories_gpu(X_train, X_valid, X_test_fold, MODEL_CAT_COLS)
    missing = [c for c in TOP_FEATURES if c not in X_train.columns]
    if missing:
        print(f'Missing {len(missing)} embedded top features; first missing:', missing[:10])
    features = [c for c in TOP_FEATURES if c in X_train.columns]
    X_train_cp = X_train[features].astype('float32').to_cupy().astype(cp.float32)
    X_valid_cp = X_valid[features].astype('float32').to_cupy().astype(cp.float32)
    X_test_cp = X_test_fold[features].astype('float32').to_cupy().astype(cp.float32)
    return X_train_cp, X_valid_cp, X_test_cp, features



# --- Cell 10 ---
y_cpu = y.to_numpy()
oof = np.zeros((len(X), len(CLASSES)), dtype='float32')
oof_fold = np.full(len(X), -1, dtype='int8')
test_pred_sum = np.zeros((len(X_test), len(CLASSES)), dtype='float32')
fold_rows = []
importance_rows = []

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_cpu), dtype=np.int8), y_cpu), start=1):
    fold_seed = SEED + fold * 100
    print(f'\n===== Fold {fold}/{N_SPLITS} | seed={fold_seed} =====')

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    y_tr = y.iloc[tr_idx].reset_index(drop=True)
    X_va = X.iloc[va_idx].reset_index(drop=True)
    y_va = y.iloc[va_idx].reset_index(drop=True)
    X_te = X_test.copy(deep=True)

    n_comp_train = len(X_tr)
    if USE_ORIGINAL_ROWS:
        X_tr = cudf.concat([X_tr, X_orig], axis=0, ignore_index=True)
        y_tr = cudf.concat([y_tr, y_orig], axis=0, ignore_index=True)
        sw_source = source_weights_gpu(n_comp_train, len(X_orig))
    else:
        sw_source = source_weights_gpu(n_comp_train, 0)

    X_tr, X_va, X_te, added_te = add_fold_safe_te_gpu(X_tr, y_tr, X_va, X_te, TE_COLS)
    X_tr_cp, X_va_cp, X_te_cp, features = prepare_selected_cupy(X_tr, X_va, X_te)
    y_tr_cp = y_tr.to_cupy().astype(cp.int32)
    y_va_cp = y_va.to_cupy().astype(cp.int32)
    print('training shape:', X_tr_cp.shape, 'validation shape:', X_va_cp.shape, 'test shape:', X_te_cp.shape)
    print('TE features added before top selection:', len(added_te))

    if USE_CLASS_WEIGHTS:
        sample_weight = class_weights_gpu(y_tr) * sw_source
        valid_weight = class_weights_gpu(y_va)
    else:
        sample_weight = sw_source.astype(cp.float32)
        valid_weight = None

    model = xgb.XGBClassifier(**make_xgb_params(fold_seed))
    model.fit(
        X_tr_cp,
        y_tr_cp,
        sample_weight=sample_weight,
        eval_set=[(X_va_cp, y_va_cp)],
        sample_weight_eval_set=[valid_weight] if valid_weight is not None else None,
        verbose=250,
    )

    va_probs = cp.asarray(model.predict_proba(X_va_cp)).astype(cp.float32)
    te_probs = cp.asarray(model.predict_proba(X_te_cp)).astype(cp.float32)
    oof[va_idx] = cp.asnumpy(va_probs)
    oof_fold[va_idx] = fold
    test_pred_sum += cp.asnumpy(te_probs) / N_SPLITS

    fold_score = balanced_accuracy_score(y_cpu[va_idx], np.argmax(oof[va_idx], axis=1))
    best_iter = getattr(model, 'best_iteration', None)
    print(f'fold {fold} balanced accuracy: {fold_score:.8f} | best_iteration={best_iter}')
    fold_rows.append({
        'fold': fold,
        'balanced_accuracy': float(fold_score),
        'best_iteration': int(best_iter) if best_iter is not None else None,
        'n_train': int(X_tr_cp.shape[0]),
        'n_valid': int(X_va_cp.shape[0]),
        'n_features': int(len(features)),
        'n_te_features': int(len(added_te)),
    })

    gain = model.get_booster().get_score(importance_type='gain')
    for i, f in enumerate(features):
        importance_rows.append({'fold': fold, 'feature': f, 'gain': float(gain.get(f'f{i}', 0.0))})

    del model, X_tr, X_va, X_te, X_tr_cp, X_va_cp, X_te_cp, y_tr, y_va, y_tr_cp, y_va_cp
    del sw_source, sample_weight, valid_weight, va_probs, te_probs
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

fold_scores = pd.DataFrame(fold_rows)
feature_importance = pd.DataFrame(importance_rows)
cv_score = balanced_accuracy_score(y_cpu, np.argmax(oof, axis=1))
print('\nFold scores:')
display(fold_scores)
print(f'OOF balanced accuracy: {cv_score:.8f}')



# --- Cell 11 ---
test_preds = test_pred_sum.astype('float32')
oof_preds = oof.astype('float32')

np.save('xgb2_oof_preds.npy', oof_preds)
np.save('xgb2_test_preds.npy', test_preds)
np.save('xgb2_oof_labels.npy', y_cpu.astype('int8'))
np.save('xgb2_oof_fold.npy', oof_fold.astype('int8'))

# Convenience aliases for downstream blend/analysis notebooks.
np.save('oof_preds.npy', oof_preds)
np.save('test_preds.npy', test_preds)

fold_scores.to_csv('xgb2_fold_scores.csv', index=False)
feature_importance.to_csv('xgb2_feature_importance.csv', index=False)

top_importance = feature_importance.groupby('feature')['gain'].mean().sort_values(ascending=False).reset_index()
display(top_importance.head(30))

pred_labels = [INT_TO_CLASS[i] for i in np.argmax(test_preds, axis=1)]
if sample is not None and ID_COL in sample.columns:
    submission = sample.copy()
    submission[TARGET] = pred_labels
else:
    submission = pd.DataFrame({ID_COL: test_ids.to_pandas().to_numpy(), TARGET: pred_labels})
submission.to_csv('submission.csv', index=False)

print('Saved:')
for path in [
    'xgb2_oof_preds.npy', 'xgb2_test_preds.npy', 'xgb2_oof_labels.npy', 'xgb2_oof_fold.npy',
    'xgb2_fold_scores.csv', 'xgb2_feature_importance.csv', 'submission.csv',
]:
    print(' ', path)

display(submission.head())


