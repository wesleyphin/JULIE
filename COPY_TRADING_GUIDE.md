# Custom Copy Trader for JULIE

## Overview

This copy trader implements a **Leader-Follower architecture** that allows you to replicate trades from one account (Leader) to multiple accounts (Followers) automatically and safely.

### Key Features

✅ **Multi-Account Support**: Copy trades to unlimited follower accounts
✅ **Flexible Position Sizing**: Scale positions per follower (e.g., 1:1, 1:0.5)
✅ **Safety Mechanisms**: Built-in circuit breaker and rate limiting
✅ **Resilient**: Handles partial failures gracefully
✅ **Integrated**: Works seamlessly with JULIE's existing trading logic
✅ **Compliant**: Designed to stay within TopStepX rate limits and policies

---

## 🚨 Critical Rules (Avoid Getting Banned)

**TopStepX has strict policies. Follow these rules:**

### ✅ DO:
- Copy trades ONLY between your own accounts
- Use reasonable execution speeds (market orders recommended)
- Monitor the circuit breaker and rate limits
- Keep follower credentials secure

### ❌ DON'T:
- Copy trades from/to someone else's account (they track IP addresses)
- Build HFT/ultra-high-frequency systems
- Try to copy limit orders (creates desync risk)
- Disable safety mechanisms without understanding consequences
- Send 100+ orders per minute (will trigger mass data entry flags)

---

## Setup Guide

### Step 1: Configure Follower Accounts

Edit `config.py` and add your follower accounts:

```python
"COPY_TRADING": {
    "enabled": True,  # Set to True to enable
    "followers": [
        {
            "username": "your_follower_username_1",
            "api_key": "YOUR_FOLLOWER_API_KEY_1",
            "account_id": "FOLLOWER_ACCOUNT_ID_1",
            "contract_id": "CON.F.US.MES.H25",  # Can trade same or different contract
            "size_ratio": 1.0,  # Same size as leader
            "enabled": True
        },
        {
            "username": "your_follower_username_2",
            "api_key": "YOUR_FOLLOWER_API_KEY_2",
            "account_id": "FOLLOWER_ACCOUNT_ID_2",
            "contract_id": "CON.F.US.MES.H25",
            "size_ratio": 0.5,  # Half the size of leader
            "enabled": True
        }
    ]
}
```

### Step 2: Get Your Follower Account Credentials

**For each follower account:**

1. Log into TopStepX dashboard
2. Navigate to **API Settings**
3. Generate a new API Key
4. Copy the `username` and `api_key`
5. Note the `account_id` (visible in account details)
6. Find the `contract_id` for the instrument you want to trade:
   - MES March 2025: `CON.F.US.MES.H25`
   - ES March 2025: `CON.F.US.ES.H25`
   - Run `python account_selector.py` to see available contracts

### Step 3: Initialize Copy Trader in JULIE

Edit `julie001.py` to initialize the copy trader:

```python
from copy_trader import create_copy_trader_from_config

# Near the top of run_bot(), after client initialization
copy_trader = create_copy_trader_from_config(CONFIG['COPY_TRADING'])

# Pass copy_trader to the client
client = ProjectXClient(copy_trader=copy_trader)
```

**Full integration example:**

```python
def run_bot():
    # Initialize copy trader (if enabled in config)
    copy_trader = None
    if CONFIG.get('COPY_TRADING', {}).get('enabled', False):
        from copy_trader import create_copy_trader_from_config
        copy_trader = create_copy_trader_from_config(CONFIG['COPY_TRADING'])
        if copy_trader:
            logging.info("✅ Copy trader initialized successfully")
        else:
            logging.warning("⚠️ Copy trader enabled but failed to initialize")

    # Initialize client with copy trader
    client = ProjectXClient(copy_trader=copy_trader)
    client.login()

    # ... rest of your bot logic
```

### Step 4: Test in Dry Run Mode

Before going live, test with dry run:

```python
# In copy_trader.py, modify copy_trade call:
copy_results = copy_trader.copy_trade(
    signal=test_signal,
    leader_price=4500.0,
    leader_account_id="leader_acc",
    dry_run=True  # Enable dry run
)
```

---

## Configuration Options

### Position Sizing Ratios

The `size_ratio` parameter controls how many contracts each follower trades:

| Ratio | Leader Trades | Follower Trades | Use Case |
|-------|--------------|----------------|----------|
| `1.0` | 5 contracts | 5 contracts | Same size accounts |
| `0.5` | 5 contracts | 2 contracts | Smaller follower account |
| `2.0` | 5 contracts | 10 contracts | Larger follower account |
| `0.2` | 5 contracts | 1 contract | Micro-account |

**Example:**
```python
"size_ratio": 0.4,  # If leader trades 5, follower trades 2 (5 * 0.4 = 2)
```

### Trading Different Instruments

You can trade different instruments on follower accounts:

```python
# Leader trades MES
{
    "contract_id": "CON.F.US.MES.H25",  # Micro E-mini S&P 500
    "size_ratio": 1.0
}

# Follower trades ES (full-size)
{
    "contract_id": "CON.F.US.ES.H25",   # E-mini S&P 500
    "size_ratio": 0.2  # Much smaller ratio due to contract size difference
}
```

### Safety Mechanisms

#### Circuit Breaker
Trips if too many orders are placed too quickly:

```python
# In copy_trader.py
CircuitBreaker(
    max_orders=5,      # Maximum orders
    time_window=1.0    # In 1 second
)
```

**When it trips:**
- Copy trading is immediately disabled
- Leader orders still execute normally
- Manual reset required: `copy_trader.reset_circuit_breaker()`

#### Rate Limiter
Prevents hitting TopStepX rate limits (200 requests/60s):

```python
RateLimiter(
    max_requests=180,  # Conservative buffer (200 limit)
    time_window=60.0   # 60 seconds
)
```

**Behavior:**
- Automatically waits if approaching limit
- Logs warnings when throttling occurs
- Prevents API ban

---

## Usage Examples

### Basic Usage (Integrated)

Once configured, the copy trader works automatically:

```python
# Leader places a trade via JULIE
signal = {
    'side': 'LONG',
    'tp_dist': 6.0,
    'sl_dist': 4.0,
    'size': 5,
    'strategy': 'RegimeAdaptive'
}

client.place_order(signal, current_price=4500.0)

# 👆 This automatically:
# 1. Places order on leader account
# 2. Copies to all enabled follower accounts
# 3. Applies size ratios
# 4. Logs all results
```

### Standalone Usage (Advanced)

Use the copy trader independently:

```python
from copy_trader import CopyTrader, FollowerAccount

# Define followers
followers = [
    FollowerAccount(
        username="follower1",
        api_key="key1",
        account_id="acc1",
        contract_id="CON.F.US.MES.H25",
        size_ratio=1.0,
        enabled=True
    )
]

# Initialize
copy_trader = CopyTrader(followers)
copy_trader.authenticate_followers()

# Copy a trade
signal = {'side': 'LONG', 'tp_dist': 6.0, 'sl_dist': 4.0, 'size': 5, 'strategy': 'TestStrat'}
results = copy_trader.copy_trade(signal, leader_price=4500.0, leader_account_id="leader")

# Check results
for acc_id, (success, error) in results.items():
    if success:
        print(f"✅ {acc_id}: Success")
    else:
        print(f"❌ {acc_id}: {error}")
```

### Monitoring Statistics

```python
# Get copy trading stats
stats = copy_trader.get_stats()
print(stats)

# Output:
# {
#     'total_copies': 47,
#     'successful_copies': 45,
#     'failed_copies': 2,
#     'circuit_breaker_trips': 0
# }
```

### Disabling Specific Followers

Temporarily disable a follower without removing config:

```python
"followers": [
    {
        "username": "follower_1",
        "api_key": "key1",
        "account_id": "acc1",
        "contract_id": "CON.F.US.MES.H25",
        "size_ratio": 1.0,
        "enabled": False  # 👈 Temporarily disabled
    }
]
```

---

## Architecture Details

### Order Execution Flow

```
┌─────────────────────────────────────────────────────┐
│ LEADER: JULIE places order via client.place_order() │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Order Success?│
         └───────┬───────┘
                 │ Yes
                 ▼
      ┌──────────────────────┐
      │ Copy Trader Enabled? │
      └─────────┬────────────┘
                │ Yes
                ▼
    ┌───────────────────────────┐
    │ copy_trader.copy_trade()   │
    └────────┬──────────────────┘
             │
             ▼
   ┌─────────────────────┐
   │ Circuit Breaker OK? │
   └──────────┬──────────┘
              │ Yes
              ▼
   ┌──────────────────────────┐
   │ For Each Follower Account│
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │ Rate Limiter (wait if    │
   │ approaching limit)       │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │ Adjust Size (size_ratio) │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │ Place Order on Follower  │
   │ (via ProjectXClient)     │
   └──────────┬───────────────┘
              │
              ▼
      ┌───────────────┐
      │ Log Results   │
      └───────────────┘
```

### Why Market Orders?

**The copy trader uses Market Orders (not Limit Orders) for followers:**

✅ **Guaranteed Fill**: Market orders ensure followers sync with leader
✅ **Avoid Desync**: Limit orders might not fill, causing portfolio drift
✅ **Simplicity**: No need to manage partial fills or order modifications

**Trade-off**: Slightly worse fill price (slippage) vs. guaranteed execution

---

## Troubleshooting

### Issue: "Circuit breaker tripped"

**Cause**: Too many orders placed too quickly

**Solution**:
```python
# Reset the circuit breaker
copy_trader.reset_circuit_breaker()

# Or adjust sensitivity in copy_trader.py:
CircuitBreaker(max_orders=10, time_window=2.0)  # More lenient
```

### Issue: "Authentication failed for follower"

**Cause**: Invalid credentials or expired API key

**Solution**:
1. Verify username/api_key in config.py
2. Regenerate API key from TopStepX dashboard
3. Check account_id matches the username
4. Ensure API access is enabled for the account

### Issue: "Rate limit approaching, waiting X seconds"

**Cause**: Approaching TopStepX rate limit (200 req/60s)

**Solution**:
- This is normal and automatic (no action needed)
- Reduce number of followers if this happens frequently
- Increase `max_requests` buffer if too aggressive

### Issue: Copy trades not executing

**Checklist**:
1. ✅ Is `COPY_TRADING['enabled']` set to `True` in config.py?
2. ✅ Did you pass `copy_trader` to `ProjectXClient` constructor?
3. ✅ Are follower accounts authenticated? (Check logs for "✅ Authenticated")
4. ✅ Are followers set to `"enabled": True` in config?
5. ✅ Is circuit breaker tripped? (Check logs for "🚨 CIRCUIT BREAKER")

### Issue: Follower gets different fill price than leader

**Cause**: Market orders fill at current bid/ask, which can vary by milliseconds

**Solution**:
- This is expected behavior (slippage is normal)
- For MES, expect 0.25-1.0 point difference
- Use bracket orders (SL/TP) to manage risk, not exact entry price

---

## Performance Considerations

### Recommended Limits

| Metric | Recommended | Maximum |
|--------|-------------|---------|
| Follower accounts | 2-5 | 10 |
| Orders per minute | < 20 | 50 |
| Concurrent API workers | 3 | 5 |

### API Call Budget

**Each copied trade uses:**
- 1 leader order = 1 API call
- N follower orders = N API calls
- **Total per trade**: 1 + N calls

**Example with 3 followers:**
- 10 trades/hour = 10 leader + 30 follower = **40 API calls/hour**
- Well within 200 calls/60s limit

---

## Legal & Compliance

### TopStepX Policies

From TopStepX Terms of Service:

> **Prohibited Conduct includes:**
> - Account sharing or trading on behalf of others
> - Mass data entry or abusive API usage
> - High-frequency trading or latency arbitrage

### How This Copy Trader Complies

✅ **Own Accounts Only**: You must own all accounts (leader + followers)
✅ **Rate Limited**: Stays well under 200 req/60s limit
✅ **Human Speed**: Market orders execute at normal human speeds
✅ **Circuit Breaker**: Prevents runaway order placement
✅ **Not HFT**: Designed for discretionary/systematic trading, not HFT

### Your Responsibility

- Ensure you own all accounts you're copying to
- Do not sell this as a "signal service"
- Do not copy trades from/to friends' accounts
- Monitor your usage and stay within platform limits

---

## Advanced Topics

### Async Execution (Future Enhancement)

Currently, the copy trader uses `ThreadPoolExecutor` for parallel execution. For even better performance, consider upgrading to `asyncio`:

```python
async def copy_trade_async(self, signal, leader_price, leader_account_id):
    # Use aiohttp for async HTTP requests
    tasks = [
        self._place_follower_order_async(follower, signal, leader_price)
        for follower in self.follower_accounts
    ]
    return await asyncio.gather(*tasks)
```

### WebSocket Order Listener (Future Enhancement)

For real-time order fill detection (instead of polling):

```python
# Connect to RTC User Hub for order events
async with websockets.connect(CONFIG['RTC_USER_HUB']) as ws:
    await ws.send(json.dumps({'method': 'SubscribeToOrders'}))
    async for message in ws:
        event = json.parse(message)
        if event['type'] == 'OrderFilled':
            # Trigger copy trade immediately
            copy_trader.copy_trade(...)
```

### Selective Strategy Copying

Only copy certain strategies to certain followers:

```python
# In config.py
"followers": [
    {
        "username": "follower_conservative",
        "allowed_strategies": ["RegimeAdaptive", "Confluence"],  # Only copy these
        ...
    },
    {
        "username": "follower_aggressive",
        "allowed_strategies": ["ICT_Model", "DynamicEngine2"],  # Only copy these
        ...
    }
]

# In copy_trader.py, filter before placing order:
if signal['strategy'] not in follower.allowed_strategies:
    continue  # Skip this follower
```

---

## Support

### Questions?

1. Check logs: `tail -f julie_bot.log` for detailed execution info
2. Review stats: `copy_trader.get_stats()` for performance metrics
3. Test in dry run: `dry_run=True` before going live

### Contributing

Found a bug or have an enhancement idea?
1. Open an issue on the repository
2. Submit a pull request with your changes
3. Include test cases and documentation

---

## Quick Reference

### Enable Copy Trading
```python
# config.py
"COPY_TRADING": {"enabled": True, "followers": [...]}
```

### Initialize in Bot
```python
# julie001.py
copy_trader = create_copy_trader_from_config(CONFIG['COPY_TRADING'])
client = ProjectXClient(copy_trader=copy_trader)
```

### Check Stats
```python
stats = copy_trader.get_stats()
```

### Reset Circuit Breaker
```python
copy_trader.reset_circuit_breaker()
```

### Dry Run Test
```python
copy_trader.copy_trade(signal, price, account_id, dry_run=True)
```

---

**Built with ❤️ for safe, compliant multi-account trading**
