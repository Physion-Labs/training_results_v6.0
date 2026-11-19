"""MLPerf llama2_70b_lora data (regisss/scrolls_gov_report_preprocessed_mlperf_2 @ 21ff123) -> Nextron's full-encode
cache (engine.vision_cache: <i>.pt = {input_ids, labels, length}) plus a placeholder jsonl whose rows only serve
the length estimate (every row longer than max_seq_len=8192, so the estimate clamps to 8192, the real length)."""
import json
import os
import sys

import pandas as pd
import torch

root = sys.argv[1]
for split in ("train", "validation"):
    df = pd.read_parquet(f"{root}/data_raw/data/{split}-00000-of-00001.parquet")
    cdir = f"{root}/cache/{split}"
    os.makedirs(cdir, exist_ok=True)
    n_lab, n_tok, bad = 0, 0, 0
    with open(f"{root}/{split}.jsonl", "w") as f:
        for i, r in enumerate(df.itertuples()):
            ids = [int(x) for x in r.input_ids]
            lab = [int(x) for x in r.labels]
            assert len(ids) == len(lab) == 8192, (i, len(ids), len(lab))
            bad += sum(1 for a, b in zip(ids, lab) if b != -100 and a != b)
            n_lab += sum(1 for b in lab if b != -100)
            n_tok += len(ids)
            torch.save({"input_ids": ids, "labels": lab, "length": len(ids)}, f"{cdir}/{i}.pt")
            row = {"messages": [{"role": "user", "content": "hello " * 9000}, {"role": "assistant", "content": "ok"}]}
            f.write(json.dumps(row) + "\n")
    print(split, len(df), "rows,", n_tok, "tokens,", n_lab, "label tokens", f"({n_lab / n_tok:.3f}),",
          "label != input_id at label positions:", bad, "| columns:", list(df.columns), flush=True)
