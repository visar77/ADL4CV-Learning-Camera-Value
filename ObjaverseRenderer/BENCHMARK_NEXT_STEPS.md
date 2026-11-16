# Base-cloud benchmark run

Run the matched fresh-versus-adapted comparison with the single benchmark
entry point and its same-named config:

```bash
python benchmarks/comparison_base_cloud_and_fresh_candidates.py --dry-run
python benchmarks/comparison_base_cloud_and_fresh_candidates.py
```

Use the existing FastGS CUDA environment and keep the selected rendered views
corpus unchanged for the full run. The default comparison trains a 30k shared
base, compares fresh 30k candidate training against 3k adaptation from that
exact base PLY, and evaluates both on identical test views. The default
candidate actions are blindspot, next, and random. See the [README](README.md#matched-benchmarking)
for input selection, config fields, output layout, and report interpretation.

Review `benchmarks/comparison_base_cloud_and_fresh_candidates/report/` for
per-candidate PSNR/SSIM/LPIPS, gain deltas, runtime, and selection-fidelity
statistics. Inspect rendered images before deciding whether the faster adapted
procedure is acceptable. A speedup does not establish equivalent quality or
fresh-training view-selection fidelity. Rerunning the same command resumes
matching outputs; use a different benchmark script/config name for a separate
protocol or timing repeat.
