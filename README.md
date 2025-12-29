# Compass-Rec-Engine
# How far are we from the old R/`earth` script — and what’s different?

Below is a side-by-side on **what the legacy R pipeline did** vs **what our latest Compass Rec Engine code does**, plus **what remains to reach the “old R behavior” where that’s still the aim**.

---

## 1) Problem framing & data grain

| Aspect | Old R (`earth`) | New Python (Compass Rec Engine) | Status vs aim |
|---|---|---|---|
| **Training grain** | **Per-member** rows (employee-level; payroll fields in features). | **Plan × Tier × Year** (population-level aggregates + plan config; no member PII). | **Intentionally different** by design to improve portability, privacy, and stability. |
| **Personalization** | Baked into model via member costs & survey. | Happens at **serve time** via overlay: `Final = α·model + (1–α)·overlay(member – population)`. | **Different on purpose** (cleaner compliance; same end-user effect). |
| **Targets** | 3 models: `deductible`, `out_of_pocket_max`, `hsa`. | Framework supports any target; you just pass `--target`. We have been demoing with `PremiumCostYear`. | **Parity achievable**: train 3 runs/targets to mirror R outputs. |

**Takeaway:** We *deliberately* moved away from per-member training to achieve robustness and privacy. We can still produce the **same three target predictors** as the R script by running the pipeline for each target.

---

## 2) Features & leakage handling

| Aspect | Old R | New Python | Status vs aim |
|---|---|---|---|
| **Member cost fields in training** | Used (`prem_amount`, `er_amount`, etc.). | **Excluded from training**; only used at serve-time overlay. | **Different by design**; improves generalization and avoids unit drift. |
| **Config & schedule** | Mixed with survey; fewer explicit config fields. | **Explicit plan config + schedule costs** (pcp/spec copays, coinsurance, Rx tiers, OOP embedded, network_type, metal_level, HSA eligible, etc.). | **Stronger**: clearer semantics, contract-tagged (TRAIN/SERVE/GATE). |
| **Leak guards** | Manual column selection; implicit. | **Automatic**: excludes IDs; checks “target-in-disguise” via name tokens & near-perfect corr; logs top |corr|. | **Improved**: safer & auditable. |
| **Survey fusion** | Concats single/family variants into combined factors (e.g., `doctor_utilization2`). | Supported if present; categorical one-hot with unknowns handled. | **Equivalent/stronger** (less brittle encodings). |

---

## 3) Modeling algorithm

| Aspect | Old R | New Python | Status vs aim |
|---|---|---|---|
| **Model class** | **MARS** via `earth` (degree=2, `nprune=62`). | Currently a **pluggable scikit-learn pipeline** (we’ve shown non-PII regressors). | **Close but not identical**. If strict parity is the goal, we can switch to **Py-Earth** (the Python MARS port). |
| **Hyperparameters** | Fixed (degree=2, `nprune=62`). | Configurable; currently using default/sensible settings for chosen estimator. | **Matchable**: set Py-Earth with same degree/pruning to emulate R. |
| **Outputs for Groovy** | Coefficients from `earth` summarized. | Exports stable artifacts (`model.joblib`, `metrics.json`, `feature_lists.json`), and we can emit **JSON equation forms** for parity harness. | **Equivalent end-state** once MARS is selected. |

> ✅ **Action to fully match the R approach where required**  
> Swap the estimator to **Py-Earth** with `degree=2` and constrained pruning (≈ `nprune=62`) and run three separate targets.

---

## 4) Evaluation methodology

| Aspect | Old R | New Python | Status vs aim |
|---|---|---|---|
| **Train/Test split** | Not standardized; typically simple split; no group holdouts. | **Group-aware** (by PlanId/Tier/Year). Cross-val + holdout metrics reported. | **Stronger**: reflects how models face unseen plan catalogs. |
| **Metrics** | Informal summaries (`summary(earth)`), ad hoc error checks. | **RMSE/MAE/R²** on holdout; **CV RMSE mean ± std**; residual exports for slicing. | **Improved**: reproducible, reviewable. |

---

## 5) Data cleaning & typing

| Aspect | Old R | New Python | Status vs aim |
|---|---|---|---|
| **NA handling** | Manual (`""`, `(null)`, `NULL` → `NA`), `na.omit` per target. | Pandas/Sklearn imputation where needed; categorical “ignore unknowns” avoids crashes. | **Equivalent/safer**; less brittle. |
| **Factor handling** | R factors → numerics; custom merges of single/family survey. | Auto-detect categorical; one-hot with unseen handling; supports the merged survey columns if provided. | **Equivalent/stronger**. |
| **ID columns** | Kept in frame but excluded in formula; risk of accidental use. | **Auto-excluded** and de-duplicated; error guard for non-unique column selection. | **Improved** (this fixed your earlier pipeline crash). |

---

## 6) Privacy, compliance, and portability

| Aspect | Old R | New Python | Status vs aim |
|---|---|---|---|
| **PII in training** | Present indirectly through member costs, payroll semantics. | **Removed** from training. Overlay handles personalization at serve time. | **Better privacy posture** (design goal achieved). |
| **Client/year portability** | Sensitive to payroll units & local quirks. | **Stable across clients/years**; plan catalog drift tolerated by encoder. | **Improved** robustness. |

---

## 7) Deployment & ops

| Aspect | Old R | New Python | Status vs aim |
|---|---|---|---|
| **Reproducibility** | Scripted, but fewer invariant checks. | Dockerized, artifacted outputs, feature lists, and metrics logged. | **Improved**. |
| **Parity harness** | Manual comparison to Groovy. | **≤1% relative error** target, aided by standardized exports; ready to emit JSON equations. | **Equivalent end-state** once equation export is wired for MARS. |

---

## What remains IF the strict “old R” *modeling* feel is required

1. **Use MARS in Python**  
   - Adopt **Py-Earth** for the estimator.  
   - Set `degree=2`, and prune to roughly match `nprune=62`.  
   - Train the **three targets**: `deductible`, `out_of_pocket_max`, `hsa`.  
   - Keep the **current grain and feature contract** (population/config) to preserve the new architecture.

2. **Mirror the R target-specific feature subsets** (optional)  
   - For `deductible` model: exclude `out_of_pocket_max` and `hsa` from features (as R did).  
   - For `out_of_pocket_max`: exclude `deductible` and `hsa`.  
   - For `hsa`: exclude `deductible` and `out_of_pocket_max`.  
   - Our pipeline already supports per-target exclusion lists—just pass them in.

3. **Emit MARS equations for Groovy**  
   - Serialize basis functions (hinges) + coefficients like R’s `summary(earth)` but in **JSON** to keep parity ≤1%.

---

## Bottom line

- We are **architecturally different on purpose** (plan/tier/year grain, privacy-first, overlay at serve time).  
- On **modeling form**, we’re **one estimator swap away** from the old R/`earth` flavor **without losing** the new system’s robustness and compliance.  
- Once we switch to Py-Earth for the three targets and export the basis equations, we’ll match the **behavioral feel of the legacy R models** while keeping all the **modernization wins** (leak safety, portability, monitoring, and artifacted deployments).

