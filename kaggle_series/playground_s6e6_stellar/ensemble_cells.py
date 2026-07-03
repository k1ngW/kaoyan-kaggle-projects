"""
跨架构集成 Cell — Kaggle Notebook 用
======================================
用法: RealMLP notebook 跑完后, 在末尾添加这些 Cell,
先 Add Input 挂载其他模型的 OOF 数据集, 然后依次运行。

前提: 当前 notebook 里已有 oof, test_preds, y_cpu, sample, cv_score 等变量。
"""

# ═══════════════════════════════════════════════════════
# Cell A: 加载外部 OOF + 搜索最优权重
# ═══════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import os, glob
from sklearn.metrics import balanced_accuracy_score

# 自动找外部 OOF (排除当前自己的 oof)
def find_external_oofs():
    external = {}
    for root, dirs, files in os.walk('/kaggle/input/'):
        if 'oof.npy' in files and 'test_preds.npy' in files:
            # 跳过本 notebook 自己的产物
            if 'cat-3' in root or 'realmlp' in root.lower():
                continue
            name = root.split('/')[-1]
            external[name] = root
    return external

external = find_external_oofs()
print(f'Found {len(external)} external OOF datasets:')
for name, path in external.items():
    print(f'  {name}: {path}')

# 加载外部 OOF
ext_oofs = {}
ext_tests = {}
for name, path in external.items():
    ext_oofs[name] = np.load(os.path.join(path, 'oof.npy')).astype('float32')
    ext_tests[name] = np.load(os.path.join(path, 'test_preds.npy')).astype('float32')
    print(f'{name} oof: {ext_oofs[name].shape}')

# 搜索最优权重 — 每个外部模型独立搜索
models = {'current': oof}
models.update(ext_oofs)
test_models = {'current': test_preds}
test_models.update(ext_tests)

print(f'\n当前模型 OOF: {cv_score:.6f}')
best_blend_score = 0
best_weights = {}

# Grid search: 权重在 0.30-0.80 范围, 步长 0.01
for w_current in np.linspace(0.30, 0.80, 51):
    remaining = 1.0 - w_current
    if len(ext_oofs) == 0:
        break
    elif len(ext_oofs) == 1:
        # 两个模型: 当前 vs 外部
        ext_name = list(ext_oofs.keys())[0]
        blend = w_current * oof + remaining * ext_oofs[ext_name]
        score = balanced_accuracy_score(y_cpu, blend.argmax(axis=1))
        if score > best_blend_score:
            best_blend_score = score
            best_weights = {'current': w_current, ext_name: remaining}
    else:
        # 多个外部模型: 等分剩余权重
        w_ext = remaining / len(ext_oofs)
        for _ in range(5):  # 简单平均, 不做 full grid
            blend = w_current * oof
            for name in ext_oofs:
                blend += w_ext * ext_oofs[name]
            score = balanced_accuracy_score(y_cpu, blend.argmax(axis=1))
            if score > best_blend_score:
                best_blend_score = score
                best_weights = {'current': w_current}
                for name in ext_oofs:
                    best_weights[name] = w_ext
            w_ext += 0.02  # 微调

print(f'\nBest Blend OOF:  {best_blend_score:.6f}')
print(f'Current solo OOF: {cv_score:.6f}  →  Δ: {best_blend_score - cv_score:+.6f}')
print('Weights:')
for name, w in best_weights.items():
    print(f'  {name}: {w:.3f}')


# ═══════════════════════════════════════════════════════
# Cell B: 生成融合提交
# ═══════════════════════════════════════════════════════

blend_test = np.zeros_like(test_preds, dtype='float32')
if 'current' in best_weights:
    blend_test += best_weights['current'] * test_models['current']
for name in ext_tests:
    if name in best_weights:
        blend_test += best_weights[name] * ext_tests[name]

pred_labels = [INT_TO_CLASS[i] for i in np.argmax(blend_test, axis=1)]
submission = sample.copy()
submission[TARGET] = pred_labels

# 生成有意义的名字
version_tag = '+'.join([f'{n[:6]}×{w:.2f}' for n, w in best_weights.items()])
fname = f'subs/ensemble_{version_tag.replace(" ", "")}.csv'
submission.to_csv(fname, index=False)

print(f'Blend: {version_tag}')
print('Pred distribution:')
print(submission[TARGET].value_counts(normalize=True).sort_index().to_string())


# ═══════════════════════════════════════════════════════
# Cell C: 提交
# ═══════════════════════════════════════════════════════

import subprocess, json

os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump({"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}, f)
os.chmod('/root/.kaggle/kaggle.json', 0o600)

subprocess.run([
    'kaggle', 'competitions', 'submit',
    'playground-series-s6e6', '-f', fname,
    '-m', f'Ensemble: {version_tag} (OOF {best_blend_score:.6f})'
], check=True)
print('Submitted!')
