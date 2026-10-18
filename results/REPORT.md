# Real LivePipe reconfiguration search benchmark

## Setup

- Template config: `experiments/correctness/configs/gpt_1_3B_12layers_pp4_dp2_8gpu.json`
- Profile database: `profiled-time-eurosys`
- Model: `gpt` / `1_3B`
- Surviving pipeline stages N: 2, 4, 8, 16, 32
- Layer-allocation values L: 8, 16, 32, 64, 128, 256, 512
- Initial stages: N+1; the middle one-node stage fails completely
- Lost layers: 4
- Stage resources: 4 GPUs; real TP/DP/recompute candidates
- Memory limit: 32000 MB
- Repeats: 3, warmups: 1
- Operator-profile loading, one-time layer-cost aggregation, and result serialization are outside the timed region.

## Median execution time (ms)

| N | L | P | cost-model precompute | DP | apply plan | total | p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 6 | 0.491 | 0.029 | 1.498 | 2.006 | 2.033 |
| 2 | 16 | 6 | 0.914 | 0.073 | 1.479 | 2.466 | 2.484 |
| 2 | 32 | 6 | 1.734 | 0.245 | 1.475 | 3.475 | 3.476 |
| 2 | 64 | 6 | 3.487 | 0.894 | 1.473 | 5.848 | 5.894 |
| 2 | 128 | 6 | 6.738 | 3.378 | 1.484 | 11.595 | 11.714 |
| 2 | 256 | 6 | 13.499 | 13.434 | 1.477 | 28.409 | 28.661 |
| 2 | 512 | 6 | 27.054 | 53.660 | 1.499 | 82.181 | 82.629 |
| 4 | 8 | 6 | 0.928 | 0.049 | 2.528 | 3.500 | 3.548 |
| 4 | 16 | 6 | 1.803 | 0.140 | 2.542 | 4.483 | 4.494 |
| 4 | 32 | 6 | 3.527 | 0.489 | 2.517 | 6.533 | 6.635 |
| 4 | 64 | 6 | 7.055 | 1.792 | 2.523 | 11.367 | 11.412 |
| 4 | 128 | 6 | 13.937 | 6.840 | 2.529 | 23.289 | 23.313 |
| 4 | 256 | 6 | 27.777 | 27.011 | 2.520 | 57.398 | 57.483 |
| 4 | 512 | 6 | 55.076 | 108.729 | 2.597 | 166.527 | 168.003 |
| 8 | 8 | 6 | 1.865 | 0.089 | 4.619 | 6.574 | 6.592 |
| 8 | 16 | 6 | 3.622 | 0.271 | 4.614 | 8.480 | 8.555 |
| 8 | 32 | 6 | 7.161 | 0.951 | 4.614 | 12.722 | 12.769 |
| 8 | 64 | 6 | 14.102 | 3.527 | 4.596 | 22.247 | 22.320 |
| 8 | 128 | 6 | 27.917 | 13.700 | 4.639 | 46.255 | 48.866 |
| 8 | 256 | 6 | 55.984 | 53.530 | 4.637 | 114.053 | 114.341 |
| 8 | 512 | 6 | 111.692 | 217.880 | 4.683 | 334.255 | 337.002 |
| 16 | 8 | 6 | 3.672 | 0.167 | 8.737 | 12.580 | 12.617 |
| 16 | 16 | 6 | 7.189 | 0.527 | 8.797 | 16.512 | 16.617 |
| 16 | 32 | 6 | 14.234 | 1.846 | 8.832 | 24.896 | 25.025 |
| 16 | 64 | 6 | 28.405 | 6.990 | 8.781 | 44.210 | 44.235 |
| 16 | 128 | 6 | 56.172 | 27.457 | 8.842 | 92.937 | 96.333 |
| 16 | 256 | 6 | 112.335 | 107.174 | 8.822 | 228.199 | 228.342 |
| 16 | 512 | 6 | 226.073 | 434.020 | 8.825 | 668.587 | 669.136 |
| 32 | 8 | 6 | 7.208 | 0.322 | 17.012 | 24.542 | 24.577 |
| 32 | 16 | 6 | 14.396 | 1.030 | 17.027 | 32.453 | 32.615 |
| 32 | 32 | 6 | 28.415 | 3.701 | 17.005 | 49.159 | 49.428 |
| 32 | 64 | 6 | 56.627 | 14.017 | 16.985 | 87.591 | 87.801 |
| 32 | 128 | 6 | 112.634 | 54.753 | 17.035 | 184.578 | 184.969 |
| 32 | 256 | 6 | 227.783 | 213.960 | 17.266 | 459.089 | 461.893 |
| 32 | 512 | 6 | 449.967 | 872.301 | 17.230 | 1339.424 | 1360.926 |

## Scaling checks

- At L=512, total planner time versus N: R^2=1.0000.
- At L=512, layer-level candidate precompute versus N: R^2=1.0000.
- At N=32, DP time versus L^2: R^2=1.0000.
- At N=32, layer-level candidate precompute versus L: R^2=1.0000; versus L^2: R^2=0.9318.
- Doubling N from 16 to 32 at L=512 changes candidate precompute by 1.99x.
- Doubling L from 256 to 512 at N=32 changes candidate precompute by 1.98x and DP by 4.08x.
- Real operator profiles are aggregated once into fixed and per-layer coefficients for every TP/DP/recompute strategy. Candidate evaluation is then O(1), so the measured table construction targets O(NLP).
- The complete search targets the paper's O(NLP + NL^2) complexity; the second term is the minimax allocation DP.
- Every case performs real profile-backed memory/time predictions, solves a complete plan, and applies it to an AcesoConfig.
- Distributed rendezvous and model-state movement are intentionally not included.
