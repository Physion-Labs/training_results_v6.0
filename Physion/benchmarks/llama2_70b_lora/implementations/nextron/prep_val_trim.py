"""The validation set's encode cache with each sample cut right after its last label token (under causal attention the
loss is the same; 30% fewer tokens), for packed evaluation rows.

  python prep_val_trim.py ROOT    # ROOT/data_raw/data/validation-*.parquet -> ROOT/cache/validation_trim, ROOT/validation_trim.jsonl"""
import json
import os
import sys

import pandas as pd
import torch

root = sys.argv[1]
df = pd.read_parquet(f"{root}/data_raw/data/validation-00000-of-00001.parquet")
cdir = f"{root}/cache/validation_trim"
os.makedirs(cdir, exist_ok=True)
lens = []
n_lab = 0
with open(f"{root}/validation_trim.jsonl", "w") as f:
    for i, r in enumerate(df.itertuples()):
        ids = [int(x) for x in r.input_ids]
        lab = [int(x) for x in r.labels]
        end = max((j for j, x in enumerate(lab) if x != -100), default=0) + 1
        ids, lab = ids[:end], lab[:end]
        lens.append(end)
        n_lab += sum(1 for x in lab if x != -100)
        torch.save({"input_ids": ids, "labels": lab, "length": end}, f"{cdir}/{i}.pt")
        row = {"messages": [{"role": "user", "content": "hello " * max(1, end - 16)}, {"role": "assistant", "content": "ok"}]}
        f.write(json.dumps(row) + "\n")
torch.save(lens, f"{cdir}/lengths.pt")
print(f"validation_trim: {len(lens)} rows, total_tok={sum(lens)} ({sum(lens)/8192:.1f} x 8k), mean={sum(lens)/len(lens):.1f}, max={max(lens)}, label_tok={n_lab}")
