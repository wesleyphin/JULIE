"""Phase A.3 smoke test for Kronos with RAM monitoring.

Hard constraints:
- Kronos-mini (4.1M params)
- sample_count=5
- device=cpu, OMP_NUM_THREADS=1
- Abort if free RAM drops below 500MB
"""
import os
import sys
import gc
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ROOT = '/Users/wes/Downloads/JULIE001'
sys.path.insert(0, os.path.join(ROOT, 'kronos_external'))

import psutil


def ram_mb():
    return psutil.virtual_memory().available // 1024 // 1024


def abort_if_low(label, floor=500):
    free = ram_mb()
    print(f"[{label}] free RAM: {free} MB")
    if free < floor:
        print(f"ABORT: free RAM {free} MB below floor {floor}")
        sys.exit(2)
    return free


print(f"Pre-import free RAM: {ram_mb()} MB")

import torch
torch.set_num_threads(1)
abort_if_low("post-torch-import")

from model import Kronos, KronosTokenizer, KronosPredictor

print("Loading tokenizer NeoQuasar/Kronos-Tokenizer-base ...")
t0 = time.time()
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
print(f"  tokenizer loaded in {time.time()-t0:.1f}s")
abort_if_low("post-tokenizer")

print("Loading model NeoQuasar/Kronos-mini ...")
t0 = time.time()
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
model.eval()
print(f"  model loaded in {time.time()-t0:.1f}s")
abort_if_low("post-model-load")

# Param count
n_params = sum(p.numel() for p in model.parameters())
print(f"  model param count: {n_params:,}")

predictor = KronosPredictor(model, tokenizer, max_context=512, device='cpu')
abort_if_low("post-predictor-init")

import pandas as pd
import numpy as np

ts = pd.date_range('2025-01-01 09:00', periods=600, freq='1min')
np.random.seed(0)
close = 6000 + np.cumsum(np.random.randn(600) * 0.5)
df = pd.DataFrame({
    'open':   close + np.random.randn(600) * 0.1,
    'high':   close + np.abs(np.random.randn(600)) * 0.3,
    'low':    close - np.abs(np.random.randn(600)) * 0.3,
    'close':  close,
    'volume': np.random.randint(100, 1000, 600).astype(float),
    'amount': np.random.randint(1000, 10000, 600).astype(float),
}, index=ts)

print("Running single inference (sample_count=5, pred_len=30) ...")
t0 = time.time()
with torch.no_grad():
    pred = predictor.predict(
        df=df.iloc[:512],
        x_timestamp=pd.Series(df.index[:512]),
        y_timestamp=pd.Series(df.index[512:512+30]),
        pred_len=30, T=1.0, top_p=0.9, sample_count=5,
    )
elapsed = time.time() - t0
print(f"Single inference: {elapsed:.2f}s")
abort_if_low("post-inference")

print(f"Forecast shape: {pred.shape}, columns: {list(pred.columns)}")
print(f"Forecast head:\n{pred.head()}")
print(f"Forecast describe:\n{pred.describe()}")

del pred
gc.collect()
print(f"Final free RAM: {ram_mb()} MB")
print("SMOKE OK")
