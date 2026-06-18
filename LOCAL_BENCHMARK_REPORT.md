# Local Benchmark Report

This file is stored outside `outputs/` so it can be committed to GitHub.

## Goal

The lab compares a standard `ReAct` agent with a `Reflexion` agent to see whether adding self-reflection improves answer quality on multi-hop QA, and what cost it adds in tokens and latency.

## Runtime Setup

- Runtime mode: `local`
- Local model: `google/flan-t5-small`
- Backend: `transformers` + `torch`
- Metrics measured from real local model calls:
  - `EM` (Exact Match)
  - `avg_attempts`
  - `avg_token_estimate`
  - `avg_latency_ms`

## Experiment 1: Hotpot Dev Distractor 100

- Dataset: `data/hotpot_dev_distractor_100.json`
- Output directory used during the run: `outputs/local_flan100`
- Number of questions: 100

| Metric | ReAct | Reflexion | Delta (Reflexion - ReAct) |
|---|---:|---:|---:|
| EM | 0.13 | 0.13 | 0.00 |
| Avg attempts | 1.00 | 1.89 | +0.89 |
| Avg token estimate | 1037.20 | 2423.69 | +1386.49 |
| Avg latency (ms) | 1769.84 | 3896.53 | +2126.69 |

### Interpretation

- Reflexion did not improve final exact-match accuracy on this 100-question benchmark.
- Reflexion nearly doubled the number of attempts.
- Reflexion consumed far more tokens and latency than ReAct.
- Main takeaway: with this small local model, reflection increased cost but did not improve final accuracy.

## Experiment 2: Hotpot Golden

- Dataset: `hotpot_golden.json`
- Output directory used during the run: `outputs/local_hotpot_golden`
- Number of questions: 20

| Metric | ReAct | Reflexion | Delta (Reflexion - ReAct) |
|---|---:|---:|---:|
| EM | 0.60 | 0.60 | 0.00 |
| Avg attempts | 1.00 | 1.40 | +0.40 |
| Avg token estimate | 687.20 | 1143.65 | +456.45 |
| Avg latency (ms) | 759.50 | 1157.40 | +397.90 |

### Interpretation

- Both agents answered 12 out of 20 questions correctly, so `EM = 0.60`.
- Reflexion again increased cost and latency.
- On this smaller and easier set, both agents performed better than on the 100-question distractor benchmark, but Reflexion still did not outperform ReAct.

## Failure Pattern Summary

- Most failures were `wrong_final_answer`.
- Common issues:
  - choosing a related but incorrect entity
  - returning an intermediate hop instead of the final answer
  - returning a partial answer span instead of the exact target

## Final Conclusion

In this lab implementation, the Reflexion pipeline works correctly and can retry with reflection memory, but with the chosen local model (`google/flan-t5-small`) it did not improve EM over ReAct on either benchmark. The extra reflection loop mainly increased token cost and latency.

## Reference Files

- `src/reflexion_lab/agents.py`
- `src/reflexion_lab/mock_runtime.py`
- `src/reflexion_lab/local_runtime.py`
- `outputs/local_flan100/report.json`
- `outputs/local_hotpot_golden/report.json`
