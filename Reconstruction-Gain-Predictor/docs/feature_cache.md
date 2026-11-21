# Frozen feature caching

Run commands from the project directory.

For cached-backbone training, encodes each distinct context once, decodes its candidate cameras, and saves embeddings. Repeating extraction reuses completed context/candidate files. **This step accelerates training; direct inference performs the backbone computation internally.** Baselines and the CNN skip it.

Production LVSM extraction on a CUDA machine, followed by head training:

```bash
python cache_features.py --dataset data/processed/table.jsonl \
  --output data/features/lvsm512 --image-size 512 --lvsm-latent-tokens 3072 \
  --device cuda --precision bf16

python train.py --dataset data/processed/table.jsonl --model cached \
  --features data/features/lvsm512/features.pt --mode context --epochs 100
```

The **frozen backbone** turns context RGB/cameras and candidate cameras into a 768-dimensional candidate-conditioned embedding. The small trainable head maps that embedding to reconstruction gain; the backbone is not optimized during head training.

| Phase | Does LVSM run? |
|---|---|
| `prepare_data.py` | No; assembles metadata/labels only. |
| First `cache_features.py` run | Yes: one encoding per unique context, then candidate decoding. |
| Repeat extraction with complete cache | No; reuses cached embeddings, even if backbone weights are absent. |
| Cached-head training, including rolling validation | No; uses embeddings already loaded in memory. |
| Direct inference on new images/cameras | Yes; loads weights once and shares context encoding across candidates. |
| Offline evaluation/inference | Uses a valid cache when available; otherwise computes features online. |

The first cache build is the long step: it may download roughly 2 GB of weights, hash source files, then run encoding/decoding. At 512 resolution each image produces 4,096 tokens; the encoder processes `3072 + 4096 × context_views` tokens through full attention. Many views and CPU execution are especially expensive. Phase messages print **before** this work starts, and the timing JSON separates hashing, loading, encoding, and decoding. Head-training epochs do not rebuild this cache.

The official `lvsm_scene_encoder_decoder_res512.pt` weights download if missing. `--lvsm-checkpoint` selects another location/checkpoint. The 3,072 latent tokens are the pretrained model's learned scene-representation capacity; each token has 768 features. `--candidate-batch-size` controls decoder memory.

For a **CPU integration/overfitting smoke test**:

```bash
python cache_features.py --dataset data/processed/table.jsonl --num-samples 8 \
  --output data/features/lvsm_smoke --image-size 16 --lvsm-latent-tokens 4 --device cpu

python train.py --dataset data/processed/table.jsonl --num-samples 8 \
  --model cached --features data/features/lvsm_smoke/features.pt --mode overfit \
  --epochs 200 --scheduler none --early-stopping-patience 0
```

Reduced resolution/latent counts change pretrained behavior; this smoke test does not validate production feature quality. Full-size CPU extraction is expensive. FP32 is the default portable precision. Optional BF16 also works on CPU; on a GPU without BF16 support, execution falls back to FP32 with a notice and possible small numerical differences.

The cache contains `features.pt`, phase timings, and resumable context/candidate files. Content/configuration hashes reject incompatible entries. Partial extraction covers only selected samples; match training selection flags or extract the whole manifest.

## Adding a backbone

Implement `FeatureBackbone` from `gain_predictor.models.backbone`, with
`encode_context()` and `decode_candidates()`. Context state is a dictionary of
tensors; candidate embeddings have shape `[B,Q,D]`. Pass its factory to
`gain_predictor.feature_cache.extract_features()`, or register it with
`gain_predictor.models.register_backbone()`. The built-in CLI and checkpoint
loader use the LVSM constructor arguments; another adapter must accept those
arguments or provide its own extraction entry point. Head training accepts any
embedding width and needs no backbone-specific changes.
