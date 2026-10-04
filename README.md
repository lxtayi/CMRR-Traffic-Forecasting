# CMRR: Causal Multi-Scale Residual Routing

This repository contains the implementation and reproducibility materials for **Causal Multi-Scale Residual Routing (CMRR)**, a lightweight output-adaptation layer for frozen multi-horizon traffic forecasters.

CMRR releases a residual only after its target is observable, maintains recent sensor- and network-level residual memories, retrieves daily and weekly residual evidence, and combines these signals with a robust horizon-specific router fitted before deployment. The forecasting backbone and router coefficients remain fixed during testing.

## Repository contents

- `cmrr/`: causal residual memory, periodic retrieval, robust routing, metrics, and block bootstrap.
- `configs/paper_config.json`: fixed seeds, temporal protocol, and selected PeMS04/PeMS08 settings.
- `artifacts/routers/`: fitted STAEformer router coefficients and stored metric summaries.
- `source_data/`: numerical source data for the manuscript tables and figures.
- `scripts/run_cmrr.py`: validation calibration and chronological test replay.
- `scripts/generate_figures.py`: regenerates the final quantitative figures from `source_data/`.
- `scripts/verify_source_data.py`: verifies every released source-data file against SHA-256 hashes.
- `tests/`: synthetic checks for delayed revelation, zero-target masking, and causal lag retrieval.

Large PeMS arrays, backbone checkpoints, and prediction tensors are not committed. The public benchmark packages are available through the official [STAEformer](https://github.com/XDZhelheim/STAEformer) and [STID](https://github.com/GestaltCogTeam/STID) repositories.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

Verify the release:

```bash
python scripts/verify_source_data.py
pytest
```

## Prediction input contract

CMRR consumes frozen-backbone predictions. Place `val.npz` and `test.npz` in one directory. Each archive must contain:

- `pred`: backbone forecasts with shape `[origins, horizons, sensors]`;
- `true`: aligned targets with the same shape.

For the PeMS experiments, recorded zero-valued targets are treated as invalid. The same target-defined mask is used for metrics, recent-memory updates, periodic retrieval, and router fitting.

During validation model selection, a fitting pair is retained only when its target time remains inside the fitting prefix. This horizon-dependent purge prevents the router-fitting prefix from using targets that fall in the validation-selection suffix.

## Run CMRR

First calibrate the recent causal memory:

```bash
python -m cmrr.causal_error_memory \
  --prediction-root PATH_TO_PREDICTIONS \
  --output outputs/recent_memory
```

Then fit the horizon-specific router on validation data and replay the test sequence chronologically:

```bash
python scripts/run_cmrr.py \
  --prediction-root PATH_TO_PREDICTIONS \
  --recent-result outputs/recent_memory/result.json \
  --output outputs/cmrr \
  --lags 288,2016
```

The output directory contains `result.json` and `corrected_test.npz`. No test target is used to optimize the router.

## Regenerate figures

```bash
python scripts/generate_figures.py
```

Vector PDF, SVG, PNG, and TIFF files are written to `outputs/figures/`.

## Reproducibility notes

- Main random seed: `2026`.
- Chronological split: 60% training, 20% validation, 20% testing.
- History and forecast lengths: 12 five-minute steps.
- Periodic residual lags: 288 and 2,016 origins (daily and weekly).
- Router: four coefficients per horizon and 12 horizons, for 48 fitted coefficients.
- Router fitting: six IRLS iterations, Huber constant 1.5, coefficient clipping to `[-1, 1]`.
- The normalizing constant in the robust scale is `Phi^{-1}(3/4) = 0.6744897501960817`.

The source-data archive records the complete benchmark comparisons, ablations, temporal-block intervals, feedback stress tests, transfer analyses, runtime measurements, and qualitative-case selection log reported in the manuscript.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the accompanying CMRR manuscript when using this code or source data.

## Contact

Xintian Liu  
Department of Urban Planning and Design, The University of Hong Kong  
xintianliu@connect.hku.hk
