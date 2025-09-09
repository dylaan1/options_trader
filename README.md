# Options Trader

Backtesting framework for options strategies using composite scores for trade decisions.

## Environment

- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`

## Project Layout

```
config/
  weights.yaml
  risk.yaml
data/
  chains/
  underliers/
src/
sqlite/
```

Configuration lives in `config/`, market data in `data/`, source code in `src/`, and SQLite results will be written under `sqlite/`.

