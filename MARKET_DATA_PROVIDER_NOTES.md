# Market Data Provider Notes

This file preserves the current decision context for revisiting Kalshi networking
and a possible Polymarket provider swap later.

## 2026-04-27 Kalshi / SOCKS Decision Snapshot

Current best path for privacy-preserving Kalshi market-data reads:

- Keep the bot local.
- Keep ProjectX / Topstep execution on the normal local network.
- Route only Kalshi / market-data HTTP requests through a dedicated, stable US
  egress path.
- Prefer a personally controlled VPS plus SSH SOCKS over a commercial rotating
  proxy.
- Use `socks5h://127.0.0.1:1080` so DNS resolution also happens through the
  tunnel.
- Avoid rotating proxies, residential proxy pools, and fake-state routing.

Recommended California VPS fit:

- Best fit: Akamai/Linode in Los Angeles, CA or Fremont, CA.
- Good alternatives: Vultr Los Angeles / Silicon Valley, or DigitalOcean SFO2 /
  SFO3.
- AWS Lightsail is stable but does not currently satisfy a California-specific
  target because its US West Lightsail region is Oregon.

Operational note:

- SOCKS makes Kalshi/Polymarket see the VPS public IP, not the home IP.
- If the VPS is in California and geolocation databases classify it that way,
  market-data services generally see a stable US/CA datacenter egress.
- This should be treated as privacy/scoped-egress isolation, not location
  spoofing.
- The current Kalshi provider signs GET requests with the Kalshi API key, so
  those reads are authenticated market data, not anonymous public browsing.
- Cleaner account behavior means all local bot iterations that read Kalshi
  market data should use the same stable egress path rather than mixing home IP
  and VPS IP for the same key.

Local test module already created:

- `tools/market_data_proxy_probe.py`
- `test_market_data_proxy_probe.py`

Important finding:

- The current venv has `requests` but not `PySocks`, so
  `socks5h://127.0.0.1:1080` requires installing `requests[socks]` or `PySocks`
  before provider wiring can work.

Suggested tunnel command:

```powershell
ssh -N -D 127.0.0.1:1080 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 ubuntu@YOUR_CA_VPS_IP
```

Suggested env:

```powershell
$env:MARKET_DATA_PROXY_URL="socks5h://127.0.0.1:1080"
$env:KALSHI_BASE_URL="https://api.elections.kalshi.com/trade-api/v2"
$env:POLYMARKET_US_BASE_URL="https://gateway.polymarket.us"
```

## 2026-04-27 Polymarket Swap Deep Dive

Short verdict:

- A clean Polymarket migration is possible only if Polymarket can provide the
  same kind of intraday SPX/ES strike ladder that the current Kalshi provider
  provides.
- It is not a safe one-for-one swap from the bot side today without a discovery
  adapter and historical recalibration.
- The lowest-risk path is to keep Kalshi as the live baseline and build a
  Polymarket provider in shadow mode first.

Why this is not just a URL swap:

- `KalshiProvider` is not just a fetcher. It exposes a full probability-curve
  interface: event selection, strike parsing, ES/SPX basis conversion,
  interpolation, target-price probability, sentiment classification, gradients,
  implied distribution, momentum, cached reads, and dashboard relative-strike
  rows.
- `kalshi_trade_overlay.py` expects a curve with at least 8 useful points by
  default and rejects flat/sparse curves. A single Polymarket yes/no market does
  not satisfy the overlay.
- `regime_manifold_engine.py` uses Kalshi-derived probabilities for crowd
  confirmation and sizing multipliers.
- `risk_engine.py` uses Kalshi sentiment and probability gradient for hour-turn
  exit logic.
- `launch_filterless_live.py` and `tools/filterless_dashboard_bridge.py` build
  dashboard state from provider methods such as `get_sentiment`,
  `get_target_probability`, `get_relative_markets_for_ui`,
  `active_settlement_hour_et`, `_current_event_ticker`, `basis_offset`,
  `enabled`, and `is_healthy`.
- The ML scripts train on Kalshi-native feature names and semantics:
  `entry_probability`, `probe_probability`, `momentum_delta`,
  `momentum_retention`, `support_score`, `threshold`, TP-aligned probability,
  minutes to settlement, nearest strike distance, open interest, volume, and
  Kalshi settlement-hour selection.

Polymarket data surfaces found:

- Polymarket US has a public market-data gateway at
  `https://gateway.polymarket.us`.
- The US public API includes market listing/search, market book, BBO, settlement,
  events, series, sports data, and search.
- Polymarket US market WebSocket data is under
  `wss://api.polymarket.us/v1/ws/markets` and requires API key authentication.
- The international Polymarket site shows SPX-related markets, including daily
  up/down and longer-dated strike-style markets, but Polymarket itself separates
  that international platform from Polymarket US. For this bot, any non-US API
  path should be treated as a compliance/ToS decision before it becomes a data
  dependency.

Local probe result:

- Added isolated discovery module: `tools/polymarket_discovery_probe.py`.
- Added tests: `test_polymarket_discovery_probe.py`.
- Live run against `https://gateway.polymarket.us` with queries `SPX`,
  `S&P 500`, `SPY`, `ES`, and `S&P` returned 0 index-related candidates at
  limit 5.
- The same run returned 52 raw search candidates, but they were sports/esports
  false positives, so they are not usable for JULIE's SPX/ES overlay.
- This makes a US-gateway Polymarket replacement look blocked unless a more
  specific endpoint, series, or newly listed finance market set is discovered.

Provider interface a Polymarket adapter must satisfy:

- State: `enabled`, `is_healthy`, `basis_offset`, `last_resolved_ticker`,
  `consecutive_failures`.
- Lifecycle: `refresh`, `async_refresh`, `clear_cache`,
  `fetch_daily_contracts`, `active_settlement_hour_et`,
  `_current_event_ticker`.
- Price transforms: `es_to_spx`, `spx_to_es`, `update_basis`.
- Curve access: `get_probability`, `get_cached_probability`,
  `get_probability_curve`, `get_nearest_market`,
  `get_nearest_market_for_es_price`, `get_relative_markets_for_ui`,
  `get_cached_relative_markets_for_ui`.
- Bot features: `get_target_probability`, `get_sentiment`,
  `get_cached_sentiment`, `get_probability_gradient`,
  `get_implied_distribution`, `get_sentiment_momentum`,
  `get_cached_sentiment_momentum`.

Data model conversion requirements:

- Discover whether Polymarket US has an active SPX/S&P/ES product set with
  markets that can be grouped into the same settlement window and interpreted as
  a monotonic strike ladder.
- Normalize outcome direction. Kalshi YES currently means probability of
  settlement above a strike. Polymarket markets may be phrased as up/down,
  above/below, hit-by-date, range, or multi-outcome. A polarity mistake would
  invert the trade overlay.
- Convert Polymarket book/BBO prices into a probability. Prefer midpoint when
  both bid and ask exist, use current/last price only as a fallback, and reject
  stale or wide-spread markets.
- Preserve liquidity checks. Low depth or stale books should mark the provider
  uninformative instead of allowing the overlay to make decisions from weak
  prices.
- Rebuild event rollover logic. Kalshi uses explicit settlement-hour tickers
  around 10:00-16:00 ET with a 5-minute rollover. Polymarket market close/end
  times must be matched before the bot can reuse the current hour gate.
- Preserve ES/SPX basis handling. JULIE trades ES but Kalshi-style markets are
  SPX-referenced, so any Polymarket SPX source still needs `basis_offset`.

Tunneling requirements:

- For REST-only Polymarket market-data polling, use the same scoped session
  pattern planned for Kalshi: a dedicated `requests.Session`, `trust_env=False`,
  and `MARKET_DATA_PROXY_URL` applied only inside the market-data provider.
- Do not route ProjectX/Topstep execution through this tunnel.
- `socks5h://127.0.0.1:1080` still requires PySocks / `requests[socks]`.
- Polymarket US WebSocket market data is not covered by the current
  requests-only probe. If WebSocket streaming becomes necessary, use either a
  client with explicit SOCKS support or a proper split tunnel such as WireGuard.
  The first safe implementation should poll REST BBO/book endpoints in shadow
  mode before adding WebSocket complexity.
- Treat tunneling as stable privacy/scoped-egress routing. Do not use it to
  bypass market access, account, or jurisdiction rules.

Recalibration answer:

- Yes, recalibration is required for any active Polymarket replacement that can
  affect entry, sizing, TP behavior, trailing, or exit logic.
- No recalibration is needed only if Polymarket is dashboard-only or pure shadow
  telemetry with no trade decisions.
- Current rule thresholds and model artifacts are Kalshi-calibrated. The
  Kalshi-trained entry gate and TP gate should not be reused as live
  Polymarket decision models.

Historical work required before live use:

- Build a Polymarket snapshot fetcher analogous to the Kalshi historical fetchers.
- Store normalized rows with at least: timestamp, event/market slug, settlement
  window, strike or threshold, normalized side/polarity, probability, bid, ask,
  spread, liquidity/depth, volume, open interest or closest equivalent, market
  state, and source endpoint.
- Build a `HistoricalPolymarketProvider` that implements the same provider
  protocol from parquet snapshots.
- Replay the canonical DE3 windows and compare Kalshi vs Polymarket decisions
  in untouched holdouts.
- Retrain/recalibrate the entry and TP models with Polymarket-native features
  only after the normalized historical dataset is stable.

Clean migration phases:

1. Discovery probe: query Polymarket US search/markets for SPX, SPY, S&P 500,
   ES, and index markets; verify whether a same-hour strike ladder exists.
2. Isolated adapter: create `services/polymarket_provider.py` with the Kalshi
   provider protocol, but do not wire it into live trading yet.
3. Shadow mode: add `MARKET_DATA_PROVIDER=kalshi|polymarket|shadow_polymarket`
   with default `kalshi`; in shadow mode, keep Kalshi active and log
   Polymarket-derived probabilities beside the current Kalshi fields.
4. Historical capture: collect/replay enough Polymarket snapshots to measure
   whether the curve is informative and stable.
5. Recalibration: retune rule thresholds and retrain ML artifacts against
   Polymarket features.
6. Controlled opt-in: only allow live Polymarket gating behind an explicit env
   flag after shadow/replay results are acceptable.

Current recommendation:

- Do not replace Kalshi live yet.
- Build the Polymarket discovery probe next.
- If Polymarket US lacks a same-hour SPX/ES strike ladder, keep Kalshi for the
  active overlay and use Polymarket only as auxiliary dashboard sentiment.
- If Polymarket US does have a comparable ladder, build the adapter in shadow
  mode and treat recalibration as mandatory before enabling it for decisions.

## 2026-04-27 Alternative Avenues

Best alternatives to avoid local/VPS/API-key mixing:

1. Public Kalshi read mode.
   - Kalshi's docs say public market-data REST endpoints do not require API
     keys.
   - Local probe confirmed `GET /markets?series_ticker=KXINXU&limit=1` returns
     `200` without credentials.
   - This can remove the same-API-key-from-two-egress-paths problem for the live
     market-data ladder.

2. Single market-data sidecar.
   - Run one Kalshi fetcher, either locally or on the CA VPS.
   - Both bot iterations read a fresh local JSON/cache produced by that fetcher.
   - Only the sidecar contacts Kalshi, so there is one egress identity and one
     rate-limit surface.

3. VPS relay instead of per-process SOCKS.
   - Run the Kalshi market-data sidecar on the CA VPS.
   - Local bot instances read from it through SSH/WireGuard/Tailscale or a
     private HTTPS endpoint.
   - ProjectX/Topstep execution remains local/direct.

4. Third-party historical data only.
   - DeltaBase currently offers Kalshi/Polymarket historical trades and metadata,
     refreshed daily.
   - This is useful for research/backtesting, not for JULIE's live intraday
     gating.

5. Options-implied probability replacement.
   - Providers such as Polygon/Massive, ThetaData, Cboe/OPRA vendors, and Cboe
     index feeds can support SPX/index/options-derived probability models.
   - This would be a new signal family, not a Kalshi replacement. It requires
     feature engineering, backtesting, and recalibration before it can gate live
     trades.

Current best path:

- Keep Kalshi's signal semantics.
- Change the networking/data-flow architecture first:
  public REST where possible, single sidecar/cache, one egress path.
- Only explore options-implied or other prediction-market replacements after
  the current Kalshi ladder is stable and isolated.

## 2026-04-27 Public Kalshi Bridge Implementation

Prepared code:

- `services/kalshi_provider.py`
  - Adds `public_read_only` so the provider can use Kalshi public REST market
    data without API credentials.
  - Uses a dedicated `requests.Session`.
  - Defaults `trust_env=False` so broken ambient Windows/system proxy settings
    do not leak into market-data reads.
  - Supports `market_data_proxy_url` / `MARKET_DATA_PROXY_URL` for scoped HTTP
    or SOCKS routing.
  - Supports `bridge_cache_path`, `bridge_cache_only`, and
    `bridge_cache_max_age_seconds` so bot instances can consume a shared local
    snapshot without contacting Kalshi directly.
  - In cache-only bridge mode, stale or flat/non-informative snapshots mark the
    provider unhealthy so the bot falls back instead of treating initialized
    zero-price markets as a real Kalshi signal.

- `tools/kalshi_public_bridge.py`
  - Runs a public-read-only Kalshi fetcher.
  - Writes a local JSON snapshot using the same parsed market shape the bot
    expects.
  - Reports fetch latency, event ticker, market count, nonzero probability
    count, and whether the curve is informative enough for the trade overlay.

- `requirements.txt`
  - Adds `PySocks==1.7.1` so `socks5h://127.0.0.1:1080` works after deps are
    installed.

Recommended runtime layout:

1. Start SSH SOCKS to the California VPS.
2. Start one bridge process with public Kalshi reads routed through SOCKS.
3. Point both local bot iterations at the same bridge JSON in
   `KALSHI_BRIDGE_CACHE_ONLY=1` mode.
4. Keep ProjectX/Topstep execution on the normal local network.

Bridge command:

```powershell
$env:KALSHI_PUBLIC_READ_ONLY="1"
$env:MARKET_DATA_PROXY_URL="socks5h://127.0.0.1:1080"
.\.venv\Scripts\python.exe tools\kalshi_public_bridge.py --output .\kalshi_public_bridge_snapshot.json --interval 5
```

Bot-side bridge env:

```powershell
$env:KALSHI_PUBLIC_READ_ONLY="1"
$env:KALSHI_BRIDGE_CACHE_PATH="D:\test projekt\JULIE-main\kalshi_public_bridge_snapshot.json"
$env:KALSHI_BRIDGE_CACHE_ONLY="1"
$env:KALSHI_BRIDGE_CACHE_MAX_AGE_SECONDS="15"
```

Validation commands:

```powershell
.\.venv\Scripts\python.exe tools\market_data_proxy_probe.py --probe kalshi-public --timeout 10
.\.venv\Scripts\python.exe tools\kalshi_public_bridge.py --once --require-informative --output .\kalshi_public_bridge_snapshot.test.json
.\.venv\Scripts\python.exe -m unittest test_kalshi_provider_public_mode.py test_market_data_proxy_probe.py test_polymarket_discovery_probe.py test_kalshi_trade_overlay.py
```

Latest local result:

- Unit tests passed: 19 focused tests.
- Syntax compile passed for touched Python files.
- SOCKS validation passed after installing `PySocks==1.7.1` into the venv.
- Public bridge fetched a snapshot in about 590-675 ms premarket.
- Because the local system time was 2026-04-27 01:39 ET, Kalshi returned
  initialized 10:00 ET markets with zero prices, so the bridge correctly marked
  the curve as not informative.
- Cache-only provider consumption now rejects that flat snapshot and marks the
  provider unhealthy, which protects live gating from fake zero-probability
  reads. During active market hours `informative_curve=true` is the sanity check
  to verify the bot is getting the decision-quality ladder.
