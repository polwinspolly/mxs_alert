"""
MXS v5 Perp Alert Bot — v4
Monitors BTC, ETH, SOL, LINK on KuCoin Futures

HTF Bias: Break-of-Structure (BOS) sticky logic
  - Tracks key high and key low (pivot-based horizontal levels)
  - Bullish when close breaks above key high → stays bullish until key low breaks
  - Bearish when close breaks below key low  → stays bearish until key high breaks
  - Matches MXS candle coloring: blue = long, red = short, sticky, no random neutral flips

LTF Signal: Aggressive close beyond nearest 15M swing high/low
Stop: Deviation extreme (wick range from swing point to signal candle)
Target: 1.6R | BE: 1R | ATR stop filter: max 2×ATR(14)
Dedup: ATR bucket grouping (prevents re-alerting same setup)
"""

import ccxt
import pandas as pd
import time
import logging
import os
from datetime import datetime, timezone

# ── CONFIG ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

# HTF per symbol
SYMBOL_HTF = {
    "BTC/USDT:USDT":  "1h",
    "ETH/USDT:USDT":  "1h",
    "SOL/USDT:USDT":  "1d",
    "LINK/USDT:USDT": "1h",
}

LTF            = "15m"
TARGET_R       = 1.6
BE_R           = 1.0
ATR_LENGTH     = 14
MAX_STOP_ATR   = 2.0   # Max stop = 2× ATR(14)
HTF_PIVOT_LB   = 5     # Swing pivot lookback for HTF key levels
LTF_PIVOT_LB   = 3     # Swing pivot lookback for LTF signal
LTF_SCAN       = 4     # Candles to scan for fresh LTF signal
CHECK_INTERVAL = 60 * 15
FETCH_RETRIES  = 3
RETRY_DELAY    = 10

# ── STATE ────────────────────────────────────────────────────────────────────
last_signals = {}

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── EXCHANGE ─────────────────────────────────────────────────────────────────
exchange = ccxt.kucoinfutures({
    "enableRateLimit": True,
    "rateLimit": 2000,
})

# ── FETCH ────────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 150) -> pd.DataFrame:
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not raw or len(raw) < 20:
                raise ValueError(f"Only {len(raw) if raw else 0} candles returned")
            df = pd.DataFrame(raw, columns=["ts","open","high","low","close","vol"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df.set_index("ts").sort_index()
        except Exception as e:
            if attempt < FETCH_RETRIES:
                log.warning(f"Fetch attempt {attempt} failed [{symbol} {timeframe}]: {e}. Retry in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                raise

# ── ATR ──────────────────────────────────────────────────────────────────────
def calc_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()

# ── PIVOT DETECTION ───────────────────────────────────────────────────────────
def find_pivots(highs, lows, lb: int):
    """
    Pivot high at i: highs[i] is max in window [i-lb : i+lb]
    Pivot low  at i: lows[i]  is min in window [i-lb : i+lb]
    Returns (pivot_highs, pivot_lows) as lists of (index, price) tuples.
    """
    ph, pl = [], []
    n = len(highs)
    for i in range(lb, n - lb):
        if highs[i] == max(highs[i - lb: i + lb + 1]):
            ph.append((i, float(highs[i])))
        if lows[i] == min(lows[i - lb: i + lb + 1]):
            pl.append((i, float(lows[i])))
    return ph, pl

# ── HTF BIAS — BOS STICKY ────────────────────────────────────────────────────
def get_htf_bias(df: pd.DataFrame) -> str:
    """
    Break-of-Structure sticky bias matching MXS candle coloring:

    Rules (matches what Pol described):
    - Bias is STICKY — once set, it holds until the opposite key level breaks
    - Bullish: when close breaks ABOVE the key high → bias = long
      Stays long as long as price doesn't close BELOW the key low
    - Bearish: when close breaks BELOW the key low → bias = short
      Stays short as long as price doesn't close ABOVE the key high

    Key levels update as new pivot highs/lows form:
    - In a long trend: key_low updates to the most recent higher pivot low
    - In a short trend: key_high updates to the most recent lower pivot high

    No "neutral" once structure is established — bias holds last confirmed direction.
    """
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(df)

    pivot_highs, pivot_lows = find_pivots(highs, lows, HTF_PIVOT_LB)

    if not pivot_highs or not pivot_lows:
        return "neutral"

    # Index-keyed pivot lookups
    ph_at = {i: p for i, p in pivot_highs}
    pl_at = {i: p for i, p in pivot_lows}

    # Seed initial key levels from the first confirmed pivots
    first_sh_idx = pivot_highs[0][0]
    first_sl_idx = pivot_lows[0][0]

    if first_sh_idx < first_sl_idx:
        key_high = pivot_highs[0][1]
        key_low  = None
    else:
        key_low  = pivot_lows[0][1]
        key_high = None

    bias      = "neutral"
    start_idx = min(first_sh_idx, first_sl_idx)

    for i in range(start_idx, n - 1):  # skip last candle — still forming

        # Update key levels as new pivots are confirmed at this candle
        if i in ph_at:
            new_ph = ph_at[i]
            if bias == "long":
                # Uptrend: update key_high to latest pivot high (tracks HH)
                key_high = new_ph
            elif bias == "short":
                # Downtrend: track most recent lower pivot high as resistance
                if key_high is None or new_ph < key_high:
                    key_high = new_ph
            else:
                key_high = new_ph

        if i in pl_at:
            new_pl = pl_at[i]
            if bias == "short":
                # Downtrend: update key_low to latest pivot low (tracks LL)
                key_low = new_pl
            elif bias == "long":
                # Uptrend: track most recent higher pivot low as support
                if key_low is None or new_pl > key_low:
                    key_low = new_pl
            else:
                key_low = new_pl

        c = closes[i]

        # BOS UP: close above key high → flip long
        if key_high is not None and c > key_high:
            bias = "long"
            prior_pl = [p for idx, p in pivot_lows if idx <= i]
            key_low  = prior_pl[-1] if prior_pl else None
            key_high = None  # will update on next confirmed pivot high

        # BOS DOWN: close below key low → flip short
        elif key_low is not None and c < key_low:
            bias = "short"
            prior_ph = [p for idx, p in pivot_highs if idx <= i]
            key_high = prior_ph[-1] if prior_ph else None
            key_low  = None

    return bias

# ── LTF SIGNAL ───────────────────────────────────────────────────────────────
def get_ltf_signal(df: pd.DataFrame, bias: str):
    """
    Aggressive LTF signal: close beyond the nearest confirmed swing in bias direction.
    Stop = deviation extreme (wick range from swing point to signal candle).
    Scans last LTF_SCAN closed candles. Always skips the last (still forming).
    Returns: (direction, entry_price, stop_price) or (None, None, None)
    """
    if bias == "neutral":
        return None, None, None

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(df)

    pivot_highs, pivot_lows = find_pivots(highs, lows, LTF_PIVOT_LB)

    for offset in range(2, LTF_SCAN + 2):
        signal_idx = n - offset
        if signal_idx < LTF_PIVOT_LB + 5:
            continue

        entry = closes[signal_idx]

        if bias == "long":
            prior_highs = [ph for ph in pivot_highs if ph[0] < signal_idx]
            if not prior_highs:
                continue
            sh_idx, sh_price = prior_highs[-1]
            if entry > sh_price:
                stop = float(min(lows[sh_idx: signal_idx + 1]))
                return "long", float(entry), stop

        elif bias == "short":
            prior_lows = [pl for pl in pivot_lows if pl[0] < signal_idx]
            if not prior_lows:
                continue
            sl_idx, sl_price = prior_lows[-1]
            if entry < sl_price:
                stop = float(max(highs[sl_idx: signal_idx + 1]))
                return "short", float(entry), stop

    return None, None, None

# ── DEDUP KEY ────────────────────────────────────────────────────────────────
def make_sig_key(signal: str, entry: float, atr: float) -> str:
    """ATR bucket grouping — prevents re-alerting same setup on consecutive cycles."""
    bucket = round(entry / atr)
    return f"{signal}_{bucket}"

# ── TELEGRAM ─────────────────────────────────────────────────────────────────
def send_telegram(message: str):
    import urllib.request, urllib.parse
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML"
    }).encode()
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=15)
            log.info("Telegram sent.")
            return
        except Exception as e:
            if attempt < 3:
                time.sleep(5)
            else:
                log.error(f"Telegram failed after 3 attempts: {e}")

# ── FORMAT ALERT ─────────────────────────────────────────────────────────────
def format_alert(symbol: str, direction: str, entry: float, stop: float,
                 atr_val: float, htf: str) -> str:
    stop_dist = abs(entry - stop)
    be  = entry + stop_dist * BE_R     if direction == "long" else entry - stop_dist * BE_R
    tp  = entry + stop_dist * TARGET_R if direction == "long" else entry - stop_dist * TARGET_R
    pct = (stop_dist / entry) * 100
    emoji = "🟢 LONG" if direction == "long" else "🔴 SHORT"
    coin  = symbol.split("/")[0]
    now   = datetime.now(timezone.utc).strftime("%H:%M UTC")
    d     = 2 if entry > 100 else (4 if entry > 1 else 6)
    fmt   = f".{d}f"

    return (
        f"<b>⚡ MXS SIGNAL — {coin} {emoji}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🕐 <b>Time:</b> {now}\n"
        f"📍 <b>Entry:</b> {entry:{fmt}}\n"
        f"🛑 <b>Stop:</b> {stop:{fmt}} ({pct:.2f}%)\n"
        f"🎯 <b>BE (1R):</b> {be:{fmt}}\n"
        f"🎯 <b>TP (1.6R):</b> {tp:{fmt}}\n"
        f"📊 <b>ATR(14):</b> {atr_val:{fmt}}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<i>MXS v5 Perp · {htf.upper()} bias + 15M flip</i>"
    )

# ── CHECK SYMBOL ─────────────────────────────────────────────────────────────
def check_symbol(symbol: str):
    htf = SYMBOL_HTF.get(symbol, "1h")
    try:
        log.info(f"Checking {symbol}...")

        df_htf = fetch_ohlcv(symbol, htf, limit=150)
        df_ltf = fetch_ohlcv(symbol, LTF, limit=100)

        # Step 1: BOS sticky HTF bias
        bias = get_htf_bias(df_htf)
        log.info(f"  {symbol} {htf.upper()} bias: {bias}")
        if bias == "neutral":
            return

        # Step 2: LTF aggressive signal
        signal, entry, stop = get_ltf_signal(df_ltf, bias)
        if not signal:
            log.info(f"  {symbol} no LTF signal")
            return

        # Step 3: ATR stop distance filter
        atr_series  = calc_atr(df_ltf, ATR_LENGTH)
        current_atr = float(atr_series.iloc[-2])
        stop_dist   = abs(entry - stop)
        max_stop    = MAX_STOP_ATR * current_atr

        if stop_dist > max_stop:
            log.info(f"  {symbol} skipped — stop {stop_dist:.4f} > 2×ATR {max_stop:.4f}")
            return

        # Step 4: ATR bucket deduplication
        key = make_sig_key(signal, entry, current_atr)
        if last_signals.get(symbol) == key:
            log.info(f"  {symbol} duplicate, skipping")
            return
        last_signals[symbol] = key

        # Step 5: Send alert
        msg = format_alert(symbol, signal, entry, stop, current_atr, htf)
        send_telegram(msg)
        log.info(f"  ✅ Alert: {symbol} {signal} entry={entry:.4f} stop={stop:.4f}")

    except Exception as e:
        log.error(f"Error [{symbol}]: {e}")

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
def run():
    symbols = list(SYMBOL_HTF.keys())
    coins   = " · ".join(s.split("/")[0] for s in symbols)
    log.info(f"MXS Alert Bot v4 started. Monitoring: {coins}")
    send_telegram(
        "🤖 <b>MXS Alert Bot v4 started</b>\n"
        f"Monitoring: {coins}\n"
        "HTF: 1H (BTC/ETH/LINK) · 1D (SOL)\n"
        "Bias: BOS sticky (key high/low breaks)\n"
        "Entry: 15M flip · Target: 1.6R · BE: 1R"
    )

    while True:
        for symbol in symbols:
            check_symbol(symbol)
            time.sleep(5)

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        log.info(f"Cycle done. Next in 15 min [{now}]")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run()
