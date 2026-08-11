# TSCGlueClassifier

Automatic Time Series Classification library built on top of aeon and scikit-learn.

## Benchmark

Critical difference diagram evaluated on 112 univariate UCR datasets:

![Critical difference diagram](figures/critical_difference.png)

## Installation

```bash
# Base install (no PyTorch)
pip install tscglue

# Generic PyTorch (pip resolves version)
pip install "tscglue[torch]"

# CPU PyTorch (via uv)
uv pip install "tscglue[cpu]"

# CUDA 12.4 PyTorch (via uv)
uv pip install "tscglue[cu124]"
```

If you already have PyTorch installed, just install the base package — it won't reinstall torch.

## Quick Start

```python
from tscglue import utils
from tscglue.models import TSCGlueClassifier
from sklearn.metrics import accuracy_score

# Load a time series classification dataset
X_train, y_train, X_test, y_test = utils.load_dataset("ArrowHead")

# Create and train the model
model = TSCGlueClassifier(
    random_state=270,
    k_folds=10,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```


# TSCGlueV2 — preset composition

## Representations

Which representation is included in which preset.

| Representation | Family | low | medium | high |
|---|---|:--:|:--:|:--:|
| `multirocket` + `hydra` | convolution | ✅ | ✅ | ✅ |
| `quant` | interval quantile | ✅ | ✅ | ✅ |
| `rstsf-random` | interval | ✅ | ✅ | ✅ |
| `rdst` | shapelet | ✅ | ✅ | ✅ |
| `weasel` | dictionary | ❌ | ✅ | ✅ |
| `fm` (`mantis` + `chronos2`) | foundation | ❌ | ✅ | ✅ |
| `drcif` | interval | ❌ | ❌ | ❌ |
| `tsfresh` | feature-based | ❌ | ❌ | ❌ |

## Base models (level 0)

`low` = `medium` minus `weasel` and `fm` — identical model names and heads on the four shared
representations. `high` = same six representations, but **two** heads each.

| Representation | low | medium | high |
|---|---|---|---|
| `multirocket` + `hydra` | `bestk-ridgecv` | `bestk-ridgecv` | `bestk-ridgecv`, `et` |
| `quant` | `et` | `et` | `ridgecv`, `et` |
| `rstsf-random` | `et` | `et` | `ridgecv`, `et` |
| `rdst` | `ridgecv` | `ridgecv` | `ridgecv`, `et` |
| `weasel` | ❌ | `bestk-ridgecv` | `bestk-ridgecv`, `et` |
| `fm` | ❌ | `ridgecv` | `ridgecv`, `et` |
| **total models** | **4** | **6** | **12** |
| **heads per representation** | 1 | 1 | 2 |

## Stacking models (level 1)

Trained on the level-0 models' OOF probabilities. `medium` and `high` train **all five**.
`low` trains **exactly one**, chosen by `eval_metric`.

| Stacking model | low | medium | high |
|---|:--:|:--:|:--:|
| `ridgecv` | ✅ \* | ✅ \*\* | ✅ \*\* |
| `logisticcv` | ❌ | ✅ | ✅ |
| `et` | ✅ \* | ✅ | ✅ |
| `nn` | ❌ | ✅ | ✅ |
| `rf` | ❌ | ✅ | ✅ |
| **total stackers** | **1** | **5** | **5** |

\* `low` trains only **one** of these two, chosen by `eval_metric`: `ridgecv` for `accuracy` /
`f1`, `et` for `log_loss` / `roc_auc` — the existing `TSCGlueClassifier` mapping (ridge wins
accuracy, ExtraTrees wins log-loss and AUC, per the critical-difference study). **`f1` is not
in that mapping yet and needs a decision** (`ridgecv` proposed, matching `accuracy`).

\*\* `ridgecv` is trained and Brier-scored, but **excluded from the stack-mean** — its
decision-function pseudo-probabilities are uncalibrated and would skew the average. It is
therefore only actually served if Brier selection picks it over the mean.

## Level-2 head

What combines the level-1 stackers into the served prediction. One head per preset.

| Level-2 head | low | medium | high |
|---|:--:|:--:|:--:|
| `probability-stack-mean` | ❌ | ✅ | ❌ |
| `probability-et-l2-all` | ❌ | ❌ | ✅ |
