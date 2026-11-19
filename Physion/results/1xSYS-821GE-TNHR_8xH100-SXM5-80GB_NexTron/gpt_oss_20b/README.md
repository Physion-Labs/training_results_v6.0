# gpt_oss_20b: one 8x H100 node with NexTron (unofficial)

**5 h 38 min 53 s** (20,333 s) to eval loss 3.3355 <= 3.34, at 294,912 samples (2.42B tokens, step 4,608 of global
batch 64). One run. This is an internal measurement following the MLPerf Training v6.0 rules for gpt_oss_20b, closed
division. It is not a scored result set, and it has not been submitted to or reviewed by MLCommons.

| | |
|---|---|
| System | 1 node, Supermicro SYS-821GE-TNHR, 8x NVIDIA H100-SXM5-80GB, 2x Xeon Platinum 8468, 2 TB host memory |
| Framework | NexTron (closed source) on PyTorch 2.12, CUDA 13.0 |
| Hyperparameters | global batch 64, lr 5.2e-4, 32 warmup steps, cosine to 5.2e-5 over 1.2M steps, AdamW (beta2 0.95, eps 1e-5, weight decay 0.1), gradient clipping 1.0, seed 1234 |
| Step time | 4.41 s on average including the evaluations (4.22 s of compute) for 64 samples of 8,192 tokens: 118.8K tokens/s over the run, 14.9K per GPU |

## How the time was measured

The run was not instrumented with the MLPerf logger, so there are no MLLOG lines, and the compliance and RCP checkers
were not run. The time comes from the run's TensorBoard timestamps. It starts at the start of the first training
step (the log time of step 1 minus its duration) and ends when the evaluation that reached the target finished. It
includes all 24 evaluations. Initialization (loading the initialized model, before step 1) is not
included; from the job's start the run took 5 h 41 min.

The job was interrupted once, at step ~1,210, when an unrelated job cancellation stopped it. It restarted from step 0
(no checkpoint), and this result is that restarted run.

## Evaluations

| Step | Samples | Tokens | Elapsed (h:mm:ss) | Eval loss |
|---|---|---|---|---|
| 192 | 12,288 | 0.10B | 0:14:17 | 6.2188 |
| 384 | 24,576 | 0.20B | 0:28:22 | 5.3375 |
| 576 | 36,864 | 0.30B | 0:42:28 | 4.8222 |
| 768 | 49,152 | 0.40B | 0:56:35 | 4.4823 |
| 960 | 61,440 | 0.50B | 1:10:43 | 4.3364 |
| 1,152 | 73,728 | 0.60B | 1:24:51 | 4.2640 |
| 1,344 | 86,016 | 0.70B | 1:39:03 | 4.0696 |
| 1,536 | 98,304 | 0.81B | 1:53:10 | 3.9716 |
| 1,728 | 110,592 | 0.91B | 2:07:15 | 3.8745 |
| 1,920 | 122,880 | 1.01B | 2:21:22 | 3.7946 |
| 2,112 | 135,168 | 1.11B | 2:35:30 | 3.7275 |
| 2,304 | 147,456 | 1.21B | 2:49:36 | 3.6496 |
| 2,496 | 159,744 | 1.31B | 3:03:43 | 3.6221 |
| 2,688 | 172,032 | 1.41B | 3:17:48 | 3.5775 |
| 2,880 | 184,320 | 1.51B | 3:31:56 | 3.5360 |
| 3,072 | 196,608 | 1.61B | 3:46:04 | 3.5093 |
| 3,264 | 208,896 | 1.71B | 4:00:10 | 3.4768 |
| 3,456 | 221,184 | 1.81B | 4:14:17 | 3.4543 |
| 3,648 | 233,472 | 1.91B | 4:28:23 | 3.4263 |
| 3,840 | 245,760 | 2.01B | 4:42:30 | 3.4053 |
| 4,032 | 258,048 | 2.11B | 4:56:36 | 3.3927 |
| 4,224 | 270,336 | 2.21B | 5:10:41 | 3.3672 |
| 4,416 | 282,624 | 2.32B | 5:24:48 | 3.3497 |
| 4,608 | 294,912 | 2.42B | 5:38:53 | 3.3355 |

## Files

- `run_log.txt`: every training step and evaluation line of the run's log, unedited.
- `eval_times.csv`: each evaluation's step, samples, tokens, seconds since the start of step 1 (TensorBoard), and
  eval loss.

The implementation (config, run script) is in `benchmarks/gpt_oss_20b/implementations/nextron/`.
