"""Llama-2-70b-hf-fused-qkv -> ...-fused-qkv-gu: each layer's mlp.gate_proj and mlp.up_proj weights concatenated into
mlp.gate_up_proj ([gate; up], in the shard that held gate_proj); every other tensor and file unchanged except the
model code (MODELING_PY, which has the fused MLP).

  python fuse_gate_up.py SRC DST MODELING_PY"""
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
            if ".mlp.up_proj." in name:
                continue
            if ".mlp.gate_proj." in name:
                up = name.replace("gate_proj", "up_proj")
                with safe_open(f"{src}/{wmap[up]}", "pt") as g:
                    u = g.get_tensor(up)
                name_gu = name.replace("gate_proj", "gate_up_proj")
                out[name_gu] = torch.cat([f.get_tensor(name), u], 0).contiguous()
                new_map[name_gu] = shard
                continue
            out[name] = f.get_tensor(name)
            new_map[name] = shard
    save_file(out, f"{dst}/{shard}", metadata={"format": "pt"})
    print(shard, len(out), "tensors", flush=True)

assert len(new_map) == len(wmap) - sum(1 for k in wmap if ".mlp.up_proj." in k)
index["weight_map"] = new_map
json.dump(index, open(f"{dst}/model.safetensors.index.json", "w"), indent=2)
for fn in os.listdir(src):
    if fn.endswith((".json", ".model")) and fn != "model.safetensors.index.json":
        shutil.copy(f"{src}/{fn}", f"{dst}/{fn}")
shutil.copy(modeling, f"{dst}/modeling_llama_fused.py")
cfg = json.load(open(f"{dst}/config.json"))
cfg["nx_fused_gate_up"] = True
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
print("done", len(new_map), "tensors")
