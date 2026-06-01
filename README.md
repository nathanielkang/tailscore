# TailScore (TCSE) — reference implementation

**Tail Conditional Score Energy (TCSE)** is a feature-aware importance reweighting method for imbalanced tabular regression. A mixed-type denoising score model supplies a per-row tail energy; energies are calibrated within feature neighbourhoods and mapped to clipped sample weights while the regressor trains on real rows only (no synthetic augmentation).

This repository is the **reference implementation** accompanying the TailScore manuscript. It contains **source code only**. Datasets are loaded at runtime via OpenML and scikit-learn; running the scripts writes JSON summaries under `results/` on your machine (not tracked in git).

**Repository:** https://github.com/nathanielkang/tailscore

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick verification (smoke test)

Run a short end-to-end check on synthetic data before any full benchmark:

```bash
python scripts/smoke_test.py
```

A successful run writes `results/pilot_smoke.json` locally (gitignored) and exits with code 0.

## Reproducing experiments

Public tabular benchmarks are loaded at runtime via OpenML and scikit-learn. Hyperparameters for the paper runs are frozen in `configs/`. Full runs write JSON summaries under `results/` on your machine only.

```bash
# Optional: export offline CSV cache for reproducible loaders (local data/ only)
python scripts/build_dataset_cache.py

# R6 synthetic suite + real-data pilots (see configs/pilot_r6.yaml, flagship_9.yaml)
python scripts/run_r6_synthetic_suite.py --quick
python scripts/run_pilot.py --config configs/pilot_r6.yaml
python scripts/run_flagship_9.py
```

Schema definitions for machine-readable outputs live in `results/schemas/`.

## Package layout

| Path | Role |
|------|------|
| `src/tailscore/methods/tcse.py` | TCSE energy, calibration, and weight mapping |
| `src/tailscore/corruption.py` | Mixed-type forward corruption schedules |
| `src/tailscore/baselines/` | ERM, FDS/LDS, DenseLoss, Tong-Reg port, TabDDPM-aug |
| `src/tailscore/data/` | Dataset loaders and evaluation splits |
| `src/tailscore/evaluation/` | Tail MAE and rank diagnostics |
| `configs/` | Smoke, pilot, and flagship YAML configurations |
| `scripts/` | CLI entry points (`smoke_test.py`, `run_pilot.py`, …) |

## Scope

**Included:** TCSE and in-repo baselines on a shared tabular MLP backbone, evaluation metrics, and configuration files needed to rerun the experiments described in the paper.

## Citation

If you use this code, please cite the associated TailScore manuscript when available.

## License

MIT License. See `LICENSE`.
