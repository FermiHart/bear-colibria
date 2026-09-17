# SPEC — normative, minimal, testable

## 1. Forecast

- Inputs to any forecast are values at or before the origin index. No future
  label enters prediction. Violation fails closed.
- Predictions and intervals are non-negative in meter units. CDF levels C are
  operator-supplied in the field; bundled levels are synthetic scenarios.

## 2. Calibration

- P(load > C) = 1 − F(C − pred), where F is the frozen empirical residual CDF
  (`experiments/exceedance_cdf_v1.json`, 672 dev residuals, 4 decimals),
  clipped to [0.005, 0.995]. The build fails if the CDF source hash changed.

## 3. Issuance

- Preparation commits the immutable forecast payload. The clock is sampled
  after that commit; only if the lead-time deadline still holds is the
  availability witness committed, linked to the payload hash.
- The exceedance service is read-only over observers. One ledger record per
  `forecast_sha256`. Complete corruption fails closed; only an incomplete final
  line may be truncated. No cross-file atomicity is claimed.

## 4. Non-goals

- No distribution-free coverage guarantee (dependent windows). No autonomous
  control action. No replacement of the operator's decision. No beta claim.
