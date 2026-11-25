#!/usr/bin/env bash
#SBATCH --job-name=lvsm-home
#SBATCH --partition=rtx_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

cluster_env="$HOME/miniforge3/envs/adl4cv-learning"
python="$cluster_env/bin/python"
export LD_LIBRARY_PATH="$cluster_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

run="$PWD"
predictor="$run/Reconstruction-Gain-Predictor"
output="$predictor/data/cluster_34"
manifest="$output/complete-34-psnr.jsonl"
weights="$predictor/lvsm_scene_encoder_decoder_res512.pt"
if [[ ! -s "$weights" ]]; then
  # cache_features.py downloads the checkpoint here on first use.
  weights="$predictor/data/weights/lvsm_scene_encoder_decoder_res512.pt"
fi

# Objects with all ten experiments scored in the audited cluster logs.
objects=(
  'American Bison Voxel'
  'AncientHammer (denisss456)'
  'Cabina De Grabacion'
  'Calimara din ceramica'
  'Chair'
  'Coke'
  'Crate'
  'Dice'
  'Emo dream pony'
  'Fire Hydrant'
  'Fridge'
  'Gearbox (Planetary)'
  'Goldman BEAR'
  'HCR2 superbike (rudolf)'
  'HW 9'
  'Kyle Broflovski'
  'lathe'
  'Military Crate'
  'My (Hopefully) Future Fightstick'
  'PC Computer - Garfield 2 Tale of Two Kitties - G'
  'Peanut alien gun'
  'personaje_vr'
  'phenom cap'
  'Po-Tato'
  'Porsche 911 993 RWB GT2 (RAUH_Welt)'
  'radioactive barrel'
  'Scarab'
  'Sci-fi _ Retro Future Car'
  'Segozancia Sp Male Crab - RapidCompact'
  'Sentient Sci-Fi Ninja'
  'simple doll'
  'Sony PlayStation 5'
  'vespa_HP'
  'Wood Box'
)

results=()
for object in "${objects[@]}"; do
  results+=("$run/results/$object")
done

"$python" -B - "${results[@]}" <<'PY'
import json
import math
from pathlib import Path
import sys

experiments = (
    "furthest_14_seed_77", "furthest_4_seed_77", "furthest_8_seed_77",
    "random_10_seed_77", "random_15_seed_77", "random_3_seed_77", "random_5_seed_77",
    "trajectory_12_seed_77", "trajectory_2_seed_77", "trajectory_6_seed_77",
)
budgets = {
    "base": 30_000,
    "blindspot-1": 3_000,
    **{f"random-{number}": 3_000 for number in range(1, 7)},
    **{f"next-{number}": 3_000 for number in range(1, 4)},
}
for object_dir in map(Path, sys.argv[1:]):
    expected = {object_dir / experiment / "results.json" for experiment in experiments}
    found = set(object_dir.rglob("results.json"))
    if found != expected:
        raise SystemExit(
            f"{object_dir}: missing {sorted(map(str, expected - found))}, "
            f"extra {sorted(map(str, found - expected))}"
        )
    for path in sorted(expected):
        result = json.loads(path.read_text())
        if (result["object_id"], result["experiment_id"]) != (object_dir.name, path.parent.name):
            raise SystemExit(f"{path}: object or experiment ID differs from its directory")
        for name, budget in budgets.items():
            record = result["reconstructions"].get(name, {})
            score = result.get("metrics", {}).get(name, {})
            psnr = score.get("averages", {}).get("PSNR", math.nan)
            if (record.get("status") != "complete" or record.get("iterations") != budget
                    or len(score.get("per_view", {})) != 8 or not math.isfinite(psnr)):
                raise SystemExit(f"{path}: incomplete {budget}-step PSNR receipt for {name}")
        for name in ("redundancy-1", "redundancy-2"):
            if result["reconstructions"].get(name, {}).get("status") != "redundant":
                raise SystemExit(f"{path}: missing {name} redundancy record")
print(f"Checked {len(sys.argv) - 1} objects and their ten PSNR experiments.")
PY

cd "$predictor"
"$python" -B -u prepare_data.py \
  --results-root "${results[@]}" --views-root "$run/views" \
  --object-ids "${objects[@]}" --metric PSNR --output "$manifest"

exec "$python" -B -u cache_features.py \
  --dataset "$manifest" --output "$output/lvsm" --lvsm-checkpoint "$weights" \
  --backbone lvsm --image-size 512 --lvsm-latent-tokens 3072 \
  --device cuda --precision bf16 --candidate-batch-size 1
