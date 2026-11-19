# gpt_oss_20b on one 8x H100 node with NexTron

Closed division. From the reference (mlcommons/training `small_llm_moe_pretraining`):

- **Model:** gpt-oss-20B's shape as Primus/Megatron builds it, at Megatron's random initialization (std 0.008; the
  attention output and expert down projections 0.008 / sqrt(48)). It has 24 layers, hidden size 2,880, 64 query heads
  of 64 and 8 KV heads, and 32 experts with top-4 softmax routing and no auxiliary loss. Each expert is a plain SwiGLU
  of width 2,880. It uses QK RMSNorm, RoPE base 150,000, a window of 128 earlier keys on every other layer, no biases
  or attention sinks, and the Llama-3.1 tokenizer (128,256 tokens).
- **Data:** C4 en 3.0.1 in the reference's Megatron indexed files, 8,192-token samples.
- **Evaluation:** loss on the first 1,024 validation samples every 12,288 training samples; target eval loss <= 3.34.
- **Optimizer:** AdamW (beta1 0.9, beta2 0.95, eps 1e-5, weight decay 0.1, none on norms), gradient clipping 1.0.

Tuned, as the closed division allows: global batch 64 and learning rate 5.2e-4 with 32 warmup steps, cosine to
5.2e-5 over the reference's 1.2M steps (so flat within the run).

## Setup

```bash
# data: the reference's C4 files in ./data
#   c4-train.en_6_text_document.{bin,idx}, c4-validation-91205-samples.en_text_document.{bin,idx}
python -m nextron.tools.mlperf_gpt_oss_init ./models/Llama-3.1-8B ./models/gpt-oss-20b-mlperf-init-s1234 --seed 1234
```

The initialization tool ships with NexTron. It writes the reference's model at random initialization as a checkpoint
the engine trains from; `./models/Llama-3.1-8B` holds the tokenizer files.

## Run

`run_and_time.sh` (config `config_1x8_h100.yaml`) trains until the eval loss reaches
3.34 and stops.

## Notes on the method

- **Eval loss** is the mean token loss over the first 1,024 validation samples in Megatron's GPTDataset order
  (documents shuffled with the seed, then cut into samples).
- **Precision:** the experts' GEMMs in FP8 (e4m3 forward, e5m2 gradients, weight gradients in BF16), router logits
  and probabilities in FP32, the rest in BF16. Master weights and AdamW state in FP32.
- **NexTron:** the weights live in host memory and stream to the GPUs layer by layer; each GPU trains on 8 of the
  step's 64 samples (data parallel, no tensor or expert parallelism); selective activation checkpointing; each
  layer's AdamW step runs inside the next step's forward.
