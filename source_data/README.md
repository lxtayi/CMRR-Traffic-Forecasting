# Source data

This directory contains the numerical data behind the CMRR manuscript tables and figures. The files cover overall forecasting results, checkpoint variation, paired temporal-block intervals, sensor-level distributions, ablations, residual diagnostics, feedback stress tests, transfer analyses, runtime measurements, the Figure 7 validation-based selection record, and the absolute node-gain sensitivity analysis. The validation-boundary audit records all 14 runs used to confirm that horizon-dependent target purging leaves the selected router hyperparameters unchanged.

Run the integrity check from the repository root:

```bash
python scripts/verify_source_data.py
```

Run `python scripts/generate_figures.py` to regenerate the released quantitative figures. Public PeMS benchmark arrays and large prediction tensors are not redistributed here.
