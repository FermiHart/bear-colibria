# Exceedance product — research package

**Release: research/HOLD.** The product the evidence supports, runnable by a
second person: calibrated exceedance probability with provable issuance. No
completed prospective evaluation, no external user, no public beta. Tested on
Python 3.12, Linux, CPU/stdlib.

## What it is

For an hourly load series (MW), the system:

1. **Predicts the next unpublished hour** with the causal mixture core
   (`adjusted`-class winner: global CDF of dev residuals).
2. **Emits P(load > C)** for contract levels C — committed to a hash-chained
   ledger **before** the predicted hour, linked to the forecast hash
   (`forecast_sha256`). Auditable proof that the probability existed before
   the event.
3. **Records and recovers** everything with one writer, fsync per event, and
   exact recovery (the already-qualified durable pattern).

## The evidence (consumed data, retrospective)

| Metric | Result |
|---|---|
| Brier P75 vs climatology | **0.086 vs 0.183 (53% better)** |
| Brier P90 vs climatology | **0.069 vs 0.098 (30% better)** |
| Alert cost (10·missed+1·false) | **3.3–6x lower than climatology** |
| One-step point NMAE (13.69%) | Earlier campaign, different protocol |
| Pre-event issuance | Mechanism delivered; prospective accumulation in progress |

**HOLD:** the edge is retrospective on consumed data. The negatives are also
registered: liquid core CfC/SSC out of the product (3 negatives, notable H3
partial), d-mechanism without edge, risk blends unqualified.

## Package structure

- `lck/`: forecasting core, durable sessions, journals, public adapters.
- `scripts/exceedance_service.py`: exceedance record emitter (read-only over
  observers, append-only ledger with fsync).
- `scripts/validate_disagreement.py`: prospective validator of the
  disagreement signal (read-only).
- `experiments/exceedance_cdf_v1.json`: **frozen CDF** (672 dev residuals,
  4 decimals) with the source hash pinned. The build verifies the hash before
  packaging; if the source changed, it fails.
- `MANIFEST.json`: SHA256 hashes of all files.

No meter datasets, credentials, native binaries, or personal logs. Single
language (EN), per the publication-incident lesson.

## Usage

From the extracted directory (Linux, Python 3.12):

```sh
# Initialize a future-hour observer with your own historical data
python3 -B scripts/future_electricity_pilot.py initialize \
  --directory ./observer --public-directory ./public-data \
  --release ./reports/release.json
```

The full flow requires an initialized observer (720+ readings) and the
configured public source. To emit exceedance records from existing observers:

```sh
python3 -B scripts/exceedance_service.py --ledger ./ledger.jsonl
```

The ledger accepts one record per `forecast_sha256`, fails closed on
corruption, and never rewrites the chain. Contract levels:
`exceedance_cdf_v1.json` (`contract_levels_mw`) — **in the field, replace them
with the operator's real contracted demand; the artifact's levels are synthetic
scenarios from the consumed window.**

Each record's durability is implied by the ledger fsync; the receiver owns
preserving the file. There is no atomic transaction between the observer
database and the ledger — two durable writes; one-event-gap recovery follows
the field collector pattern.

## Honest limits

- The CDF is from the consumed Jun–Jul window (DE). Applying it to another
  domain (BR) or season without refit is research, not product — the BR pilot
  accumulates targets for its own CDF (24 retrospective are insufficient;
  64+ needed).
- The Brier edge is retrospective. Prospective validation keeps accumulating
  (timers active in the origin workspace).
- The package has not been published or sent to anyone. Beta: HOLD.
