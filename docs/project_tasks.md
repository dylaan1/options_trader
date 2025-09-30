# Project Task Plan: Options Pricing & Mispricing Scanner

This document outlines the major workstreams required to deliver the Monte Carlo pricing engine and the live mispricing scanner. Each task block includes a scope summary, implementation checklist, and the specific inputs or decisions needed from you before work can proceed. Use the table of contents below to navigate directly to any task.

## Table of Contents
- [1. Core Monte Carlo Pricing Engine](#1-core-monte-carlo-pricing-engine)
- [2. Model Calibration & Market Data Normalisation](#2-model-calibration--market-data-normalisation)
- [3. Pricing Validation & Regression Harness](#3-pricing-validation--regression-harness)
- [4. Performance Optimisation Strategy](#4-performance-optimisation-strategy)
- [5. Live Market Data Ingestion Service](#5-live-market-data-ingestion-service)
- [6. Mispricing Analytics & Ranking Layer](#6-mispricing-analytics--ranking-layer)
- [7. Risk Management & Portfolio Constraints Module](#7-risk-management--portfolio-constraints-module)
- [8. Persistence & Reporting Interfaces](#8-persistence--reporting-interfaces)
- [9. Deployment, Configuration, & Observability](#9-deployment-configuration--observability)

---

## 1. Core Monte Carlo Pricing Engine
**Goal:** Implement a flexible, vectorised Monte Carlo simulator for vanilla and extendable to exotic options.

**Checklist:**
- Design GBM (and alternative) stochastic process generators with variance reduction hooks.
- Implement payoff evaluators for calls, puts, spreads, and path-dependent prototypes.
- Compute discounted expected values and confidence intervals (CLT and bootstrap variants).
- Surface API for batch pricing requests (single options or strategy baskets).

**Info Needed From You:**
1. Preferred underlier models beyond GBM (e.g., Heston, jump diffusion)?
2. Target option types (European only, Americans, barriers)?
3. Time-step granularity and path count expectations?
4. Required output fields (price, delta, vega, CI bounds, etc.)?

---

## 2. Model Calibration & Market Data Normalisation
**Goal:** Ensure the simulator inputs (volatility, rates, dividends) are sourced and calibrated from market data.

**Checklist:**
- Define volatility surface bootstrapping (historical, implied, or blended).
- Establish curve inputs for interest rates and dividends.
- Normalise option metadata (maturity, strike conventions, contract multipliers).
- Create configuration schema linking data sources to model parameters.

**Decisions Confirmed:**
- Market data provider: Schwab Developer API for volatility, rates, dividends, quotes, and account balances.
- Calibration cadence: daily at 23:50 Pacific Time using the latest available data.
- Missing data handling: persist `null` values with explicit missing-field flags and retry via dedicated refetch logic.
- Calibrated-parameter storage: JSON files persisted by the calibration engine's `JsonCalibrationStore`.

**Info Needed From You:**
1. Any additional symbols/universes beyond the primary calibration underlier?
2. Credentials rotation policy or token refresh workflow for the Schwab API?
3. Storage format for calibrated parameters (e.g., SQLite schema, JSON documents, cloud store object keys) so the persistence layer can be aligned with your operational preferences.

**Info Needed From You:**
1. Data providers/API endpoints for vol, rates, dividends?
2. Calibration frequency (intraday, daily)?
3. Preferred fallback rules when data is missing?
4. Storage format for calibrated parameters?

---

## 3. Pricing Validation & Regression Harness
**Goal:** Build confidence through automated comparisons against analytical benchmarks and historical cases.

**Checklist:**
- Set up test suite comparing Monte Carlo prices to Black–Scholes for vanilla options.
- Define tolerance thresholds and statistical validation metrics.
- Include scenario backtests using historical market regimes.
- Automate regression tests in CI or scheduled jobs.

**Decisions Confirmed:**
- Pricing tolerance defaults to an absolute error check (e.g., \$0.05) with an internal switch to relative tolerance when required for deep in/out-of-the-money contracts.
- Validation universe prioritises SPX, SPY, NDX, QQQ, TSLA, AAPL, GOOGL, META, PLTR, ORCL and supplements with the 50 instruments exhibiting the largest absolute fair-value deviations each run.
- Regression harness will run three times per trading day at 06:45, 12:00, and 15:30 Pacific Time, with an optional recommendation to add 10:30 and 16:15 Pacific sweeps for heightened market regimes.

**Outstanding Info Needed From You:**
1. Historical periods or instruments to prioritise for benchmark comparisons?

**Info Needed From You:**
1. Acceptable pricing error tolerance (absolute/relative)?
2. Historical periods or instruments to prioritise?
3. Testing cadence (on commit, nightly)?

---

## 4. Performance Optimisation Strategy
**Goal:** Profile and accelerate hot paths while keeping code maintainable.

**Checklist:**
- Instrument baseline NumPy implementation with profiling hooks.
- Evaluate Numba JIT vs. C++ (pybind11) for critical sections.
- Establish benchmarking scenarios and KPIs (paths/sec, latency).
- Build continuous performance regression tests.

**Info Needed From You:**
1. Hardware targets (CPU specs, availability of GPU)?
2. Latency/throughput requirements for fair value updates?
3. Appetite for C++ toolchain complexity vs. pure Python acceleration?

---

## 5. Live Market Data Ingestion Service
**Goal:** Pull live (or delayed) option and underlier data into the local store for pricing comparisons.

**Checklist:**
- Implement API client with robust error handling and retry logic.
- Schedule polling cadence and throttling respecting API limits.
- Normalise and persist quotes, greeks, and metadata into SQLite (or chosen DB).
- Log ingestion metrics and anomalies.

**Info Needed From You:**
1. API provider, credentials, and rate limits?
2. Required asset universe (tickers, option chains)?
3. Desired storage schema (tables, columns)?
4. SLA for data freshness?

---

## 6. Mispricing Analytics & Ranking Layer
**Goal:** Identify contracts with significant deviations between market quotes and model fair values.

**Checklist:**
- Define mispricing metric (e.g., z-score, percentile rank, raw difference).
- Apply liquidity filters (volume, open interest, bid-ask spread).
- Surface actionable rankings with contextual data (expiry, strike, moneyness).
- Integrate alerting or export paths (email, dashboard, CSV).

**Info Needed From You:**
1. Preferred mispricing formula and thresholds?
2. Minimum liquidity constraints?
3. Output format (CLI, web dashboard, CSV reports)?
4. Frequency for re-ranking (real-time, periodic)?

---

## 7. Risk Management & Portfolio Constraints Module
**Goal:** Enforce prudent exposure limits and scenario awareness for trade recommendations.

**Checklist:**
- Implement position sizing rules based on account equity and Greeks.
- Model scenario shocks (volatility, underlying moves) and stress impacts.
- Track aggregate risk metrics (delta, gamma, vega, theta) across positions.
- Provide alerting or blocking when constraints are breached.

**Info Needed From You:**
1. Account capital base and leverage assumptions?
2. Risk tolerance thresholds (max loss per trade, per day)?
3. Existing risk frameworks to integrate with?

---

## 8. Persistence & Reporting Interfaces
**Goal:** Make results accessible and auditable via structured storage and reports.

**Checklist:**
- Design database schema for simulation outputs, market snapshots, and trade logs.
- Build reporting jobs (daily summaries, performance dashboards).
- Expose data access via CLI commands or lightweight API endpoints.
- Implement archival and backup routines.

**Info Needed From You:**
1. Preferred reporting cadence and format?
2. Retention policy for historical data?
3. Access control requirements?

---

## 9. Deployment, Configuration, & Observability
**Goal:** Package the system for reliable operation, monitoring, and configurable behaviour.

**Checklist:**
- Containerise services or provide virtualenv setup scripts.
- Centralise configuration (YAML/JSON + environment overrides).
- Add logging, metrics, and health checks for long-running services.
- Document operational runbooks and troubleshooting steps.

**Info Needed From You:**
1. Target deployment environment (local desktop, cloud VM)?
2. Preference for orchestration (systemd, Docker Compose, Kubernetes)?
3. Monitoring/alerting stack availability?

---

### How to Use This Plan
1. Review each section and provide the requested inputs under “Info Needed From You.”
2. We will refine the scope and prioritise based on your feedback.
3. Implementation tickets can then be created from each checklist item in your preferred task tracker or as GitHub issues.

Feel free to annotate this document or respond inline with answers, and we can turn your responses into actionable development tickets.
