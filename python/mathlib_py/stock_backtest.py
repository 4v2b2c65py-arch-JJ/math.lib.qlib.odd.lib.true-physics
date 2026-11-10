"""Stock API backtest: connect market data, generate hash/mathlib-based predictions
("omens"), correlate with future returns, and store backtest data.

Uses yfinance (if available) for historical data. Predictions are derived from
times-plank/hash/tree/speculative structures to relate gains to existing structure.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import (
    HashBaseStore,
    MathOp,
    MathTree,
    base,
    group_outputs_by_category,
    process_math_event_loop,
    run_series_op,
    speculative_matrix_from_grid,
    sort_by_category,
    CalcEvent,
)


@dataclass
class BacktestConfig:
    ticker: str = "SPY"
    period: str = "2y"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    base_value: int = 8
    window: int = 5
    initial_capital: float = 10000.0


@dataclass
class StepRecord:
    index: int
    date: str
    close: float
    ret: float
    signal: float
    predicted_direction: int
    actual_direction: int
    correct: bool
    strategy_ret: float
    equity: float
    buy_hold_equity: float
    hash_anchor: int
    categories: List[str]
    qbit_alpha_mean: float
    conjecture_ratio: float


def _fetch_history(cfg: BacktestConfig):
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(f"yfinance required for stock API access: {e}")
    kwargs = {"interval": cfg.interval, "progress": False, "auto_adjust": True}
    if cfg.start or cfg.end:
        df = yf.download(cfg.ticker, start=cfg.start, end=cfg.end, **kwargs)
    else:
        df = yf.download(cfg.ticker, period=cfg.period, **kwargs)
    if df is None or len(df) == 0:
        raise RuntimeError(f"no data returned for {cfg.ticker}")
    # handle multi-index columns from newer yfinance
    if hasattr(df.columns, "levels") and len(getattr(df.columns, "levels", [])) > 1:
        try:
            df = df.xs(cfg.ticker, axis=1, level=-1)
        except Exception:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _hash_signal(values: Sequence[float]) -> Tuple[float, int, Dict[str, Any]]:
    """Derive an 'omen' prediction signal from recent values using mathlib hash/tree structure."""
    # map values to integer series for times-plank
    ints = []
    for v in values:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            ints.append(1)
        else:
            ints.append(max(1, int(round(abs(v) * 100)) % 10000) or 1)
    base_v = ints[-1] if ints else 8
    xs = ints[-min(len(ints), 8):] if ints else [1]
    run = run_series_op(MathOp.MUL, base_v, xs)
    store = HashBaseStore(base_num=base_v)
    # record hashes through base signature
    for i, x in enumerate(xs):
        base(base_v, x, store=store, op=MathOp.MUL)
    outs = [c for r in run.grid.rows for c in r.cells]
    if not outs:
        return 0.0, 0, {"categories": [], "qbit_alpha_mean": 0.0, "conjecture_ratio": 0.0}
    # aggregate hash/qbit/category structure
    hvals = [o.hash for o in outs]
    anchor = int(sum(hvals) & 0xFFFFFFFFFFFFFFFF)
    # normalized signal in [-1,1] from hash distribution
    mean_h = sum(hvals) / len(hvals)
    spread = (max(hvals) - min(hvals)) or 1
    sig = ((mean_h % spread) / spread) * 2.0 - 1.0
    grouped = group_outputs_by_category(outs)
    cats = list(grouped.keys())
    alphas = [o.qbit.alpha for o in outs if o.qbit is not None]
    qmean = sum(alphas) / len(alphas) if alphas else 0.5
    conj = sum(1 for o in outs if o.conjecture) / len(outs)
    # bias signal by qbit/conjecture structure (relate gains to existing structure)
    sig = max(-1.0, min(1.0, sig * 0.7 + (qmean - 0.5) * 0.6 + (conj - 0.5) * 0.2))
    meta = {"categories": cats, "qbit_alpha_mean": qmean, "conjecture_ratio": conj, "store": store.summary()}
    return float(sig), anchor, meta


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def run_backtest(cfg: BacktestConfig) -> Dict[str, Any]:
    df = _fetch_history(cfg)
    closes: List[float] = []
    dates: List[str] = []
    for idx, row in df.iterrows():
        # Close column
        val = row["Close"] if "Close" in row else row.iloc[0]
        try:
            c = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
        except Exception:
            c = _to_float(val)
        if math.isnan(c):
            continue
        closes.append(c)
        try:
            dates.append(str(getattr(idx, "date", lambda: idx)()))
        except Exception:
            dates.append(str(idx))

    if len(closes) < cfg.window + 2:
        raise RuntimeError("not enough data for backtest window")

    # returns
    rets = [0.0]
    for i in range(1, len(closes)):
        prev = closes[i - 1] if closes[i - 1] != 0 else 1e-9
        rets.append((closes[i] / prev) - 1.0)

    steps: List[StepRecord] = []
    equity = cfg.initial_capital
    bh_equity = cfg.initial_capital
    pred_dirs: List[float] = []
    actual_dirs: List[float] = []
    correct_flags: List[int] = []

    # start after window
    for i in range(cfg.window, len(closes) - 1):
        window_vals = closes[i - cfg.window:i + 1]
        signal, anchor, meta = _hash_signal(window_vals)
        predicted = 1 if signal >= 0 else -1
        actual_ret = rets[i + 1]
        actual_dir = 1 if actual_ret >= 0 else -1
        correct = (predicted == actual_dir)
        # strategy: go long if predicted up else short (symmetric)
        strat_ret = (predicted * actual_ret)
        equity = equity * (1.0 + strat_ret)
        bh_equity = bh_equity * (1.0 + actual_ret)
        steps.append(StepRecord(
            index=i,
            date=dates[i],
            close=closes[i],
            ret=actual_ret,
            signal=signal,
            predicted_direction=predicted,
            actual_direction=actual_dir,
            correct=correct,
            strategy_ret=strat_ret,
            equity=equity,
            buy_hold_equity=bh_equity,
            hash_anchor=anchor,
            categories=meta.get("categories", [])[:8],
            qbit_alpha_mean=float(meta.get("qbit_alpha_mean", 0.0)),
            conjecture_ratio=float(meta.get("conjecture_ratio", 0.0)),
        ))
        pred_dirs.append(float(predicted))
        actual_dirs.append(float(actual_dir))
        correct_flags.append(1 if correct else 0)

    n = len(steps)
    accuracy = sum(correct_flags) / n if n else 0.0
    correlation = _corr(pred_dirs, [s.ret for s in steps])
    total_strat = (equity / cfg.initial_capital - 1.0) if cfg.initial_capital else 0.0
    total_bh = (bh_equity / cfg.initial_capital - 1.0) if cfg.initial_capital else 0.0
    # max drawdown
    peak = -1e18
    max_dd = 0.0
    for s in steps:
        if s.equity > peak:
            peak = s.equity
        dd = (peak - s.equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    summary = {
        "ticker": cfg.ticker,
        "period": cfg.period,
        "interval": cfg.interval,
        "steps": n,
        "accuracy": accuracy,
        "direction_correlation": correlation,
        "total_strategy_return": total_strat,
        "total_buy_hold_return": total_bh,
        "alpha_vs_buy_hold": total_strat - total_bh,
        "final_equity": equity,
        "final_buy_hold_equity": bh_equity,
        "max_drawdown": max_dd,
        "human_predictive_baseline_accuracy": 0.5,
        "matches_human_predictive_engine": abs(accuracy - 0.5) < 0.03,
        "beats_human_predictive_engine": accuracy > 0.53,
    }

    result = {
        "config": asdict(cfg),
        "generated_at_ns": time.time_ns(),
        "source": "yfinance",
        "summary": summary,
        "steps": [asdict(s) for s in steps],
        "notes": [
            "Signals derived from mathlib times-plank/hash/qbit/category structures (omens) over price windows.",
            "Backtest is long/short by predicted direction; for research comparison only, not financial advice.",
            "Human predictive engine baseline approximated at ~50% directional accuracy on noisy markets.",
        ],
    }
    return result


def store_backtest(result: Dict[str, Any], out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return str(out_path)


def run_and_store(cfg: BacktestConfig, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    result = run_backtest(cfg)
    out_dir = out_dir or (Path(__file__).resolve().parents[2] / "data")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in cfg.ticker)
    out_path = out_dir / f"stock_backtest_{safe}.json"
    result["stored_path"] = store_backtest(result, out_path)
    return result
