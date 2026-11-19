# MLPerf Training llama2_70b_lora: 8x H100 with NexTron (unofficial)

**Score: 22.35 minutes**: the olympic mean of 10 runs (fastest and slowest dropped, mean of the other 8) is 21.60
minutes, normalized by the RCP checker's factor 1.0347.

Internal measurement following the MLPerf Training v6.0 rules for llama2_70b_lora, closed division. Not submitted to
or reviewed by MLCommons.

| | |
|---|---|
| System | 1 node, Supermicro SYS-821GE-TNHR, 8x NVIDIA H100-SXM5-80GB, 2x Xeon Platinum 8468, 2 TB host memory |
| Framework | NexTron (closed source) on PyTorch 2.12, CUDA 13.0 |
| Hyperparameters | batch 8, lr 6e-4, cosine over 550 steps, no warmup; the rest fixed by the closed division |
| Step time | 3.08 s (8 samples of 8,192 tokens), 2,660 tokens/s per GPU |

## Runs

The 10 runs below are the whole result set: one configuration, seeds from `/dev/urandom`, all runs listed. Times are
`run_start` to `run_stop` in each MLPerf log.

| Log | Seed | Converged at (samples) | Last eval loss | Time (min) | Status |
|---|---|---|---|---|---|
| result_0.txt | 1262705762 | 3,072 | 0.9246 | 21.39 | success (fastest, dropped) |
| result_8.txt | 150146891 | 3,072 | 0.9239 | 21.39 | success |
| result_5.txt | 1099210273 | 3,072 | 0.9249 | 21.50 | success |
| result_1.txt | 2112819255 | 3,072 | 0.9233 | 21.59 | success |
| result_9.txt | 87762478 | 3,072 | 0.9233 | 21.62 | success |
| result_2.txt | 1648916111 | 3,072 | 0.9226 | 21.64 | success |
| result_3.txt | 717879800 | 3,072 | 0.9241 | 21.64 | success |
| result_6.txt | 1081290250 | 3,072 | 0.9221 | 21.66 | success |
| result_4.txt | 1073582483 | 3,072 | 0.9219 | 21.76 | success |
| result_7.txt | 200489859 | 3,456 | 0.9228 | 24.23 | success (slowest, dropped) |

Initialization (model load, FP8 weight cache, 2 synthetic warm-up steps and one validation pass, nothing trained)
took 4.1-4.3 min per run, outside the scored time (limit 30 min).

## Checks (mlperf_logging 4.1.69, `checks/`)

- **Compliance checker** (`--ruleset 6.0.0`): all 10 logs pass.
- **RCP checker**: passed. The 8 scored runs converged at 3,072 samples on average, faster than the batch-8 reference
  mean of 3,178.7 but within the allowed speedup (minimum 2,960.6), so the score is multiplied by
  3,178.7 / 3,072 = 1.0347: 21.60 -> 22.35 minutes.

## Notes

- The 10 runs ran in parallel on 10 identical nodes ("Runs may go in parallel on the submitter's compute
  resources").
- The page cache was dropped before each run (`cache_clear: true`).
- Eval loss follows the reference's HF Trainer at evaluation batch 1: the mean over samples of each sample's token
  mean.

## Layout

```
systems/1xSYS-821GE-TNHR_8xH100-SXM5-80GB_NexTron.json
benchmarks/llama2_70b_lora/implementations/nextron/     config, run script, data and model prep
results/1xSYS-821GE-TNHR_8xH100-SXM5-80GB_NexTron/llama2_70b_lora/result_{0..9}.txt   MLPerf logs
checks/compliance.txt, checks/rcp.txt                   checker output
```

## gpt_oss_20b

**5 h 38 min 53 s** to eval loss 3.3355 <= 3.34 at 294,912 samples, on the same node type (global batch 64, lr
5.2e-4, AdamW). One run, not instrumented with the MLPerf logger (the time is from the run's TensorBoard timestamps),
so it is not a scored result set. Details in `results/1xSYS-821GE-TNHR_8xH100-SXM5-80GB_NexTron_open/gpt_oss_20b/`,
the implementation in `benchmarks/gpt_oss_20b/implementations/nextron/`, the system in
`systems/1xSYS-821GE-TNHR_8xH100-SXM5-80GB_NexTron_open.json`.
