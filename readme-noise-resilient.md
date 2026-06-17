# Noise-resilient training (bootstrap outlier down-weighting)

A training mode for MACE that automatically **down-weights configurations whose
loss is an outlier**, so the model does not waste capacity fitting noisy /
corrupted labels. It also logs the **per-config weight** every batch, keyed by a
`config_id`, so you can see exactly which structures were de-emphasised. Please 
see the [paper (arxiv: 2602.08849)](https://arxiv.org/abs/2602.08849) for details.

## Short explanation

At every optimisation step MACE now computes a **per-configuration loss** (one
scalar per structure in the batch) instead of a single batch-averaged number.
A Gaussian `N(mu, sigma)` is fit to the distribution of those per-config losses,
maintained as an exponential moving average across batches/epochs. Each config
gets a soft weight from the Gaussian tail:

```
z      = (loss_config - mu) / sigma
weight = 0.5 * (1 + erf((threshold - z) / sqrt(2)))      # = CDF of N(mu, sigma)
loss   = mean( weight**2 * loss_config )
```

* A config with a **typical** loss (small `z`) gets `weight ≈ 1`.
* A config whose loss sits **far above** the mean (`z` well above `threshold`)
  gets `weight → 0` and barely contributes to the gradient.

This is a soft, self-calibrating form of outlier rejection: nothing is hard-cut,
and the threshold adapts as `(mu, sigma)` track the loss distribution during
training.

Set `bootstrap: false` (the default) and the run is **identical to vanilla
MACE** — all weights are 1.

## Quickstart

```bash
mace_run_train --config noise_resilient_train.yaml
```

See [`noise_resilient_train.yaml`](noise_resilient_train.yaml) for a documented
template. The relevant block is:

```yaml
bootstrap: true
threshold: 2.0
bootstrap_EMA_alpha: 0.9
power_law: 1.0
power_law_coeff: 1.0
config_id_key: config_id
```

## Options

All options are normal MACE CLI flags (use them on the command line or as keys
in the YAML config without the `--`).

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `bootstrap` | bool | `False` | Master switch. `True` enables outlier down-weighting; `False` = vanilla MACE. |
| `threshold` | float | `2.0` | z-score above which a config is treated as an outlier. **Lower = more aggressive** down-weighting. |
| `bootstrap_EMA_alpha` | float | `0.9` | EMA weight on the *previous* `(mu, sigma)` estimate. `mu = (1-alpha)*batch + alpha*prev`. Higher = slower/steadier. |
| `power_law` | float | `1.0` | Controls how the refit cadence changes over time. `>1` makes refits of `(mu, sigma)` progressively sparser. |
| `power_law_coeff` | float | `1.0` | Divisor controlling refit frequency. `1.0` with `power_law=1.0` refits **every batch**; larger = less often. |
| `config_id_key` | str | `config_id` | `atoms.info` key holding the per-config id used in the weight log (see below). |
| `actual_energy_key` | str | `actual_energy` | *Optional.* `atoms.info` key for the "clean" energy in noise-injection studies. |
| `actual_forces_key` | str | `actual_forces` | *Optional.* `atoms.arrays` key for the "clean" forces in noise-injection studies. |

The refit-cadence rule: `(mu, sigma)` is refit on batches where
`progress**(1/power_law) / power_law_coeff` is (numerically) an integer, where
`progress` is the global batch counter. With the defaults this is every batch.

## Tracking the weight of each config (`config_id`)

Every structure carries a `config_id`. It is resolved in this order:

1. **From the xyz** — if a frame has `config_id=<int>` in its info line
   (or whatever key you set with `--config_id_key`), that value is used.
2. **Auto-assigned** — otherwise the **frame index within the file** is used,
   and a log line reports how many ids were auto-assigned.

> ⚠️ Auto-assigned ids are only unique **within one file**. `valid_fraction`
> splits keep the train file's ids (fine), but a separate `test_file` restarts
> at 0. For ids that are unique across all your files, stamp them explicitly.

### Assigning ids when they are not in the xyz

Use the helper script to write persistent, globally-unique ids into the xyz:

```bash
# single file -> ids 0..N-1
mace_assign_config_ids -i train.xyz -o train_ids.xyz

# several files -> one continuous, non-overlapping counter
mace_assign_config_ids \
    -i train.xyz valid.xyz test.xyz \
    -o train_ids.xyz valid_ids.xyz test_ids.xyz

# custom key / overwrite existing ids
mace_assign_config_ids -i data.xyz -o data_ids.xyz --key my_id --overwrite
```

(`mace_assign_config_ids` is installed as a console script; equivalently run
`python -m mace.cli.assign_config_ids`.)

Then train on the `*_ids.xyz` files (and pass `--config_id_key my_id` if you
used a custom key).

## Log files produced (written to the working directory)

| File | When | Columns |
| --- | --- | --- |
| **`config_weights_log.csv`** | every batch, one row **per config** | `epoch, batch_in_epoch, progress, config_id, weight, per_config_loss, z_score, mean, std` |
| `mean_std_log.csv` | every batch, one row | `epoch (fraction), mean, std, mean weight` |
| `training_log.csv` | start of each epoch, per config | `epoch, config_id, actual energy error/atom, energy error/atom, actual force error, force error` |
| `valid_log.csv` | start of each epoch, per config | same columns as `training_log.csv` |

`config_weights_log.csv` is the file that answers *"what weight did each config
get?"* — join it on `config_id` to map weights (and the loss / z-score that
produced them) back to specific structures. Only rank 0 writes it under
distributed training.

The `training_log.csv` / `valid_log.csv` "actual …" columns are only meaningful
if your xyz provides `actual_energy` / `actual_forces` (the clean labels in a
noise-injection experiment); otherwise they compare against zero.

### Quick analysis example

```python
import pandas as pd
df = pd.read_csv("config_weights_log.csv")
# mean weight each config received over the whole run (low = treated as outlier)
print(df.groupby("config_id")["weight"].mean().sort_values().head(20))
```

## Caveats

* Only the common losses were adapted to the per-config (`reduction="none"`)
  path: `weighted` (energy+forces), `stress`, `virials`,
  `energy_forces_dipole`, and `l1l2`. The `universal`/Huber and polarizability
  losses are **not** wired for this and will error with `bootstrap`.
* `config_weights_log.csv` has one row per config per batch — for very large
  datasets over many epochs it grows quickly. Delete/rotate it between runs (it
  is appended to).

## Citation

> Lam, Terry CW, Niamh O'Neill, Christoph Schran, and Lars L. Schaaf. "Cutting
> Through the Noise: On-the-fly Outlier Detection for Robust Training of Machine
> Learning Interatomic Potentials." arXiv preprint arXiv:2602.08849 (2026).

```bibtex
@article{lam2026cutting,
  title={Cutting Through the Noise: On-the-fly Outlier Detection for Robust Training of Machine Learning Interatomic Potentials},
  author={Lam, Terry CW and O'Neill, Niamh and Schran, Christoph and Schaaf, Lars L},
  journal={arXiv preprint arXiv:2602.08849},
  year={2026}
}
```
