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
