# Gain metrics

Plots the PSNR, relative PSNR, SSIM and LPIPS gain of adding one candidate camera to a base reconstruction. Positive gain means improvement. No GPU needed.

```bash
python3 ObjaverseRenderer/experiments/gain_metrics.py/plot_gains.py
```

Settings are in `config.json` (override with `--config`, or per setting with `--objects`, `--output`, `--output-name`, `--formats`):

| Key | Meaning |
| --- | --- |
| `inputs` | Result roots containing `results/<object>/<experiment>/results.json`, relative to the config |
| `objects` | Objects to include, or `null` for all |
| `output`, `output_name` | Figures are written to `<output>/<output_name>/` |
| `formats` | `png`, `pdf`, `svg` (PNG is always written) |

Figures: histograms and box plots per metric (by object and by base experiment), `overview`, `overview_by_experiment`, and `correlations`.
