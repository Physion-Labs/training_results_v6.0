# llama2_70b_lora on one 8x H100 node with NexTron

Closed division. The reference's model (Llama-2-70B with fused qkv), data (GovReport, 3,901 train / 173 validation
samples of 8,192 tokens), LoRA (r 16, alpha 32, dropout 0.1 on `qkv_proj` and `o_proj`), AdamW (weight decay 1e-4),
gradient clip 0.3, batch 8 and target (eval loss <= 0.925, every 384 samples, the first 3 evaluations skipped).
Tuned: learning rate 6e-4, cosine schedule over 550 steps, no warmup.

## Setup

```bash
python prep_data.py ./data          # parquet from regisss/scrolls_gov_report_preprocessed_mlperf_2 in ./data/data_raw
python prep_val_trim.py ./data      # validation samples cut after their last label token (same loss, fewer tokens)
python fuse_qkv.py      ./models/Llama-2-70b-hf            ./models/Llama-2-70b-hf-fused-qkv    <model code>
python fuse_gate_up.py  ./models/Llama-2-70b-hf-fused-qkv  ./models/Llama-2-70b-hf-fused-qkv-gu <model code>
```

`fuse_gate_up.py` concatenates each layer's gate and up projections into one matrix (the same function). The model
code passed to both scripts ships with NexTron.

## Run

`run_and_time.sh` (config `config_1x8_h100.json`). Per run: page cache dropped, seed
from `/dev/urandom`, initialization and 2 synthetic warm-up steps before `run_start`, nothing trained in the warm-up.

## Notes on the method

- **Eval loss** is the mean over the 173 samples of each sample's mean token loss, as the reference's HF Trainer
  reports it with an evaluation batch of one sample. Validation samples are packed several to a row (block-diagonal
  attention); a packed sample's first token is not scored, as in the reference.
- **Precision**: frozen base weights in FP8 (e4m3, one scale per matrix), GEMM inputs e4m3, gradients e5m2,
  attention in BF16 (FlashAttention 3), LoRA adapters and optimizer state in FP32.
- **NexTron**: the frozen weights live in host memory and stream to the GPUs layer by layer (each GPU copies 1/8 of a
  layer over PCIe, NVLink all-gather for the rest); each GPU trains on one full 8,192-token sample, no tensor
  parallelism; selective activation checkpointing keeps the projection outputs (FP8) so the backward recomputes no
  GEMM; the optimizer step runs inside the next step's forward.
