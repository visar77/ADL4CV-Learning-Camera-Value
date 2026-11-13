# Reviewed object subsets

`unflagged20_20261113.json` lists the exact 20 G-buffer IDs left after removing
all 30 visually flagged objects from the seed-77, 50-object Objaverse++-filtered
sample prepared on 2026-11-13. Original comparison numbers are:
01, 03, 04, 05, 09, 10, 13, 14, 19, 20, 24, 31, 33, 34, 39, 40, 43, 46, 48, 50.

Download exactly this set from `ObjaverseRenderer/`:

```bash
python3 download_gbuffer_objaverse.py --count 20 --seed 77 \
  --index object_subsets/unflagged20_20261113.json --output-dir objects
```

The index contains 20 distinct IDs, so selecting all 20 preserves the exact set
regardless of shuffle order. The existing downloader fetches their original GLBs
and names. Other files already in `objects/` remain there.

The subset means **no recorded difficulty flag in the earlier single-view
review**, not validated reconstruction quality. It retains the earlier quality
and attribute filtering and has no new reconstruction measurements. Local
comparison/provenance artifacts live under `views_unflagged20_20261113/` (ignored
generated output). Only this index and the existing downloader are needed to
download the same set on another machine.
