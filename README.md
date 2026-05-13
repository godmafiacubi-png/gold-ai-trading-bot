# Gold Market Research Bot

Starter Python project for gold market research, MT5 data access, feature engineering, backtesting, and paper-trading simulation.

This project is educational/research-focused. It starts with simulation mode enabled and does not place real orders by default.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py --config config.yaml
```

## Modules

- `core/` application services
- `strategies/` market structure and feature logic
- `ml/` model training and prediction helpers
- `backtest/` historical simulation
- `utils/` logging and helpers

## Development path

1. Verify local Python environment.
2. Confirm market data loading.
3. Run historical simulations.
4. Train and evaluate models.
5. Review risk metrics before any real-world use.

## Hybrid AI engine

The live entry point now runs a six-layer hybrid decision stack after the base ICT/SMC signal is generated:

1. **Regime classifier** - labels trend/range and high/normal/low volatility regimes, with optional `models/regime_classifier.pkl` override.
2. **Boosting signal scoring** - blends the base setup confidence with an optional boosted classifier at `models/hybrid_signal_scorer.pkl`; if no model exists, deterministic scoring rules are used.
3. **Anomaly detection** - blocks abnormal return/range/spread bars using rolling z-scores.
4. **Volatility model** - forecasts ATR with EWMA and adapts SL, TP, and size multipliers.
5. **Risk AI** - approves, blocks, or reduces size based on signal quality, anomaly score, regime confidence, and volatility.
6. **RL execution policy** - chooses execution aggressiveness and slippage limits from an optional Q-table at `models/rl_execution_q_table.json`; if no Q-table exists, safe rule-based execution is used.

The bot is still research/paper-first. `execution.dry_run` remains enabled by default, and missing model files do not crash the program because every AI layer has a conservative fallback.

### Hybrid configuration

Hybrid AI is controlled from `config.yaml`:

```yaml
hybrid_ai:
  enabled: true
  regime_model_path: models/regime_classifier.pkl
  signal_model_path: models/hybrid_signal_scorer.pkl
  rl_q_table_path: models/rl_execution_q_table.json
  min_signal_score: 0.68
  anomaly_z_threshold: 3.0
  volatility_ewm_span: 20
```

Run the paper/dry-run flow as usual:

```bash
python main.py --config config.yaml
```

For production-grade research, train each optional model only on walk-forward splits, keep a never-seen holdout period, and compare against the built-in rule fallbacks before increasing risk.
