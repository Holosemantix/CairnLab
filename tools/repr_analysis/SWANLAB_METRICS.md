# SwanLab Metrics Reading Notes

This note records the working path for reading SwanLab run metrics from this
repo. Do not commit API keys or paste them into scripts.

## Authentication

Use the SwanLab Python SDK. The front-end chart APIs may return `401` or `403`
when called directly with ad-hoc headers, even when the same API key works
through the SDK.

```bash
export SWANLAB_API_KEY='...'
```

The existing temporary environment on this machine already has `swanlab`:

```bash
/tmp/swanlab-read/bin/python -c 'import swanlab; print(swanlab.__version__)'
```

If it is missing, create a throwaway venv under `/tmp` and install `swanlab`.

## Run Lookup

Use the public path format shown in SwanLab URLs:

```python
import os
import swanlab

api = swanlab.Api(api_key=os.environ["SWANLAB_API_KEY"])
run = api.run(path="qunteam/worldmodels/gps6asjv22tmflag9af5m")

print(run.name, run.id, run.state, run.url)
```

To find a run by name:

```python
for run in api.runs(path="qunteam/worldmodels"):
    if "hetero" in run.name:
        print(run.name, run.id, run.state, run.url)
```

Known hetero runs from 2026-05-09:

| Run name | Run id |
|---|---|
| `tworoom_lewm_hetero_default` | `gps6asjv22tmflag9af5m` |
| `pusht_lewm_hetero_default` | `tge50bhmtws06xc7n4wtq` |

## Chart Id To Metric Key

Chart links look like:

```text
https://swanlab.cn/@qunteam/worldmodels/chart/default/<run_id>/<chart_id>
```

Use the authenticated SDK client to map a `chart_id` to its metric key:

```python
chart_id = "kxbr3K_K"
info, _response = run._client.get(f"/experiment/{run.id}/chart/{chart_id}/info")
print(info["title"])
print([axis["key"] for axis in info["config"]["yAxis"]])
```

For the TwoRoom hetero links:

| Chart id | Metric key |
|---|---|
| `kxbr3K_K` | `fit/hetero_s_mean` |
| `-CpThSeH` | `fit/hetero_s_std` |
| `AN8W6K2c` | `fit/hetero_weight_q10` |
| `en8Z6wwj` | `fit/hetero_weight_q90` |
| `hIYga6BL` | `fit/hetero_s_logerr_corr` |

## List Available Metric Keys

The first page of columns:

```python
columns, _response = run._client.get(f"/experiment/{run.id}/column")
for item in columns["list"]:
    print(item["key"])
```

If more than 20 keys exist, page through with query params:

```python
columns, _response = run._client.get(
    f"/experiment/{run.id}/column",
    params={"index": 2, "size": 20},
)
```

Lightning metrics are usually prefixed by stage:

```text
fit/hetero_s_logerr_corr
validate/hetero_s_logerr_corr_epoch
fit/pred_loss_mse_equiv
validate/pred_loss_mse_equiv_epoch
```

Using an unprefixed key such as `hetero_s_logerr_corr` can return `404`.

## Read Metrics

```python
keys = [
    "fit/hetero_s_mean",
    "fit/hetero_s_std",
    "fit/hetero_s_abs_max",
    "fit/hetero_weight_q10",
    "fit/hetero_weight_q90",
    "fit/hetero_weight_q10_q90_ratio",
    "fit/hetero_s_logerr_corr",
    "fit/pred_loss",
    "fit/pred_loss_mse_equiv",
    "fit/sigreg_loss",
    "fit/loss",
    "validate/hetero_s_logerr_corr_epoch",
    "validate/hetero_weight_q10_q90_ratio_epoch",
    "validate/pred_loss_mse_equiv_epoch",
]

df = run.metrics(keys=keys)

for key in keys:
    if key not in df.columns:
        print("MISSING", key)
        continue
    series = df[key].dropna()
    if series.empty:
        print("EMPTY", key)
        continue
    tail_n = min(100, len(series))
    print(
        key,
        "n=", len(series),
        "first=", float(series.iloc[0]),
        "last=", float(series.iloc[-1]),
        "min=", float(series.min()),
        "max=", float(series.max()),
        "tail_mean=", float(series.tail(tail_n).mean()),
    )
```

## Direct API Pitfalls

These direct front-end calls can fail even with a valid API key:

```text
GET /api/experiment/<run_id>/chart/<chart_id>/info
GET /api/project/<workspace>/<project>/runs/metrics
```

Observed failures:

| Status | Meaning in this workflow | Usual fix |
|---|---|---|
| `401 Unauthorized` | Key not accepted by this endpoint/header form | Use `swanlab.Api(api_key=...)` |
| `403 Forbidden` | Endpoint sees no project READ role | Use SDK authenticated client or confirm workspace access |
| `404 Not Found` | Metric key is wrong or missing stage prefix | List columns and use exact key |

## Hetero Interpretation Checklist

For sigma-conditioned JEPA runs, read these first:

| Metric | What to check |
|---|---|
| `fit/hetero_s_logerr_corr` | Whether sigma tracks prediction error; stable positive values mean the head learned difficulty. |
| `fit/hetero_s_std` and `fit/hetero_s_abs_max` | Whether sigma is non-constant and whether it hits clamp bounds. |
| `fit/hetero_weight_q10_q90_ratio` | How aggressively easy/hard tokens are reweighted; very small values indicate strong gradient imbalance. |
| `fit/pred_loss_mse_equiv` vs `fit/pred_loss` | Whether the true MSE keeps improving or hetero loss is only becoming negative through weighting. |
| `validate/*_epoch` versions | Epoch-level stability and train/validation agreement. |

For PushT, combine these curves with representation diagnostics:

```text
clean_nn_cos_dist_median
clean_effective_rank
transition_resolution_ratio_cos
transition_resolution_ratio_l2
id_probe_r2
action_mean_pred_shift_norm
```

If hetero reweighting is strong while these resolution metrics collapse, the
model is likely downweighting hard-but-task-critical transitions.
