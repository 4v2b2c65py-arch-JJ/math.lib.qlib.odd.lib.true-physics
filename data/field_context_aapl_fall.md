# Field Context: AAPL Fall (Git Tree Fine Print) — Gravity Context Check

<!-- fine print: contextual field note derived from git tree run/state and stored backtest -->
<!-- git tree (timeline, 2026-11-10T16:48Z): branch=master (no commits yet, HEAD unborn) -->
<!-- git status -sb: ## No commits yet on master; ?? .gitignore Cargo.lock Cargo.toml benchmark.gguf benchmark_updated.gguf data/ pyproject.toml python/ src/ -->

## Git Tree Run (fine print)
- repo: `F:\timeline` | branch: `master` | commits: none (`git log`: "your current branch 'master' does not have any commits yet")
- working tree (untracked): `.gitignore`, `Cargo.lock`, `Cargo.toml`, `benchmark.gguf`, `benchmark_updated.gguf`, `data/`, `pyproject.toml`, `python/`, `src/`
- interpretation: "fall" read as **max drawdown/gravity pull** observed in AAPL backtest run, recorded as field context against this git tree state (no commit anchor yet).

## AAPL Backtest (stored run)
- source: `data/stock_backtest_AAPL.json` (yfinance, generated from mathlib hash/omen signal backtest)
- config: ticker `AAPL`, period `2y`, interval `1d`, steps `495`, base_value `8`, window `5`
- metrics:
  - accuracy: `0.5333` (directional) | direction_correlation: `0.01785`
  - total_strategy_return: `0.573195` | total_buy_hold_return: `0.536308` | alpha_vs_buy_hold: `0.036887`
  - final_equity: `15731.95` | final_buy_hold_equity: `15363.08`
  - **max_drawdown (fall/gravity): `0.333605`** (33.36% peak-to-trough)
  - human_predictive_baseline_accuracy: `0.50` | matches_human_predictive_engine: `false` | beats_human_predictive_engine: `true`

## Gravity Context Check
- gravity/relativity resonance base: corrections referenced via `python/mathlib_py/relativity_correction.py` (Einstein SR base plate, c/γ, `correct_times_plank_grid`/`correct_via_event_loop`, hash-stored corrected values) applied as field correction lens over existing structure.
- context note: AAPL "fall" = drawdown event bound to this git tree snapshot; use stored hash-anchored steps in `stock_backtest_AAPL.json` to correlate omen/hash categories and conjecture/qbit structure with realized moves for future prediction checks.

<!-- end fine print -->
