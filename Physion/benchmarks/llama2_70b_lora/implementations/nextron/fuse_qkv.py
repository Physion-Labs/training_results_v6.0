"""Llama-2-70b-hf -> Llama-2-70b-hf-fused-qkv: each layer's q/k/v_proj weights concatenated into qkv_proj ([q; k; v],
in the shard that held q_proj), every other tensor unchanged; config.json gets auto_map to modeling_llama_fused.

  python fuse_qkv.py SRC DST MODELING_PY"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst, modeling = sys.argv[1:4]
os.makedirs(dst, exist_ok=True)
index = json.load(open(f"{src}/model.safetensors.index.json"))
wmap = index["weight_map"]
new_map = {}
for shard in sorted(set(wmap.values())):
    out = {}
    with safe_open(f"{src}/{shard}", "pt") as f:
        for name in f.keys():
            if ".self_attn.k_proj." in name or ".self_attn.v_proj." in name:
                continue
            if ".self_attn.q_proj." in name:
                parts = []
                for p in ("q_proj", "k_proj", "v_proj"):
                    other = name.replace("q_proj", p)
                    with safe_open(f"{src}/{wmap[other]}", "pt") as g:
                        parts.append(g.get_tensor(other))
                name = name.replace("q_proj", "qkv_proj")
                out[name] = torch.cat(parts, 0).contiguous()
            else:
                out[name] = f.get_tensor(name)
            new_map[name] = shard
    save_file(out, f"{dst}/{shard}", metadata={"format": "pt"})
    print(shard, len(out), "tensors", flush=True)

assert len(new_map) == len(wmap) - 2 * sum(1 for k in wmap if ".self_attn.q_proj." in k)
index["weight_map"] = new_map
json.dump(index, open(f"{dst}/model.safetensors.index.json", "w"), indent=2)
cfg = json.load(open(f"{src}/config.json"))
cfg["architectures"] = ["LlamaFusedQKVForCausalLM"]
cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_llama_fused.LlamaFusedQKVForCausalLM"}
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
for fn in ("generation_config.json", "special_tokens_map.json", "tokenizer.model", "tokenizer_config.json"):
    shutil.copy(f"{src}/{fn}", f"{dst}/{fn}")
shutil.copy(modeling, f"{dst}/modeling_llama_fused.py")
print("done", len(new_map), "tensors")
