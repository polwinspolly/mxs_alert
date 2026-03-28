"""
MXS v5 Perp Alert Bot — v6
Monitors BTC, ETH, SOL, LINK on KuCoin Futures

HTF Bias: Hybrid 2-layer BOS (matches MXS candle coloring)

  INTERNAL structure (lb=2, fast):
    - Tracks internal LH when bias=short → close above = flip long
    - Tracks internal HL when bias=long  → close below = flip short
    - Direction-aware: only tracks the level relevant to current bias
    - Fires first (priority) — catches early flips like the ETH/BTC screenshots

  EXTERNAL structure (lb=5, slow):
    - Confirmed pivot highs/lows with forward confirmation
    - Establishes initial bias from first confirmed pivots
    - Acts as confirmation and catches moves internal misses

  Bias is sticky — holds last confirmed direction until opposite BOS fires.
  No random neutral flips mid-trend.

LTF Signal: Aggressive close beyond nearest 15M swing high/low
Stop: Deviation extreme (wick range from swing to signal candle)
Target: 1.6R | BE: 1R | ATR stop filter: max 2×ATR(14)
Dedup: ATR bucket grouping
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
MAX_STOP_ATR   = 2.0
HTF_PIVOT_LB   = 5     # External pivot lookback
LTF_PIVOT_LB   = 3     # LTF signal pivot lookback
LTF_SCAN       = 4     # Candles to scan for LTF signal
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
    Returns lists of (index, price) tuples, oldest first.
    """
    ph, pl = [], []
    n = len(highs)
    for i in range(lb, n - lb):
        if highs[i] == max(highs[i - lb: i + lb + 1]):
            ph.append((i, float(highs[i])))
        if lows[i] == min(lows[i - lb: i + lb + 1]):
            pl.append((i, float(lows[i])))
    return ph, pl

# ── HTF BIAS — HYBRID BOS ────────────────────────────────────────────────────
def get_htf_bias(df: pd.DataFrame) -> str:
    """
    Hybrid 2-layer BOS bias matching MXS candle coloring.

    INTERNAL (fast, lb=2 equivalent — 3-bar detection):
      Tracks direction-relevant level only:
      - bias=short: watches internal LH → close above = flip long
      - bias=long:  watches internal HL → close below = flip short
      Direction-aware prevents false flips from normal pullbacks.

    EXTERNAL (slow, lb=HTF_PIVOT_LB):
      - Uses confirmed pivots with forward-movement confirmation
      - Establishes initial bias
      - Catches major structure breaks internal misses

    Priority: internal fires first → external as fallback.
    Bias is sticky — no neutral once established.
    """
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(df)

    ext_ph, ext_pl = find_pivots(highs, lows, HTF_PIVOT_LB)
    if not ext_ph or not ext_pl:
        return "neutral"

    ph_at = {i: p for i, p in ext_ph}
    pl_at = {i: p for i, p in ext_pl}

    bias         = "neutral"
    key_high     = None   # external resistance level
    key_low      = None   # external support level
    last_valid_ph = None  # last confirmed external pivot high
    last_valid_pl = None  # last confirmed external pivot low
    int_key_high  = None  # internal LH — watched when bias=short
    int_key_low   = None  # internal HL — watched when bias=long

    def confirms_down_move(idx):
        future = lows[idx + 1: min(idx + 5, n)]
        return len(future) > 0 and min(future) < lows[idx]

    def confirms_up_move(idx):
        future = highs[idx + 1: min(idx + 5, n)]
        return len(future) > 0 and max(future) > highs[idx]

    first_sh = ext_ph[0][0]
    first_sl = ext_pl[0][0]
    start    = min(first_sh, first_sl)

    for i in range(max(start, 2), n - 1):
        c = closes[i]

        # ── Internal swing point detection (3-bar: bar i-1 is local extreme) ──
        if lows[i-1] < lows[i-2] and lows[i-1] < lows[i]:
            if bias == "long":
                # In uptrend: track this HL as support — BOS down if broken
                int_key_low = lows[i-1]

        if highs[i-1] > highs[i-2] and highs[i-1] > highs[i]:
            if bias == "short":
                # In downtrend: track this LH as resistance — BOS up if broken
                int_key_high = highs[i-1]

        # ── External pivot confirmation ───────────────────────────────────────
        if i in ph_at:
            if confirms_down_move(i):
                last_valid_ph = ph_at[i]
                if bias in ("short", "neutral"):
                    key_high = last_valid_ph

        if i in pl_at:
            if confirms_up_move(i):
                last_valid_pl = pl_at[i]
                if bias in ("long", "neutral"):
                    key_low = last_valid_pl

        # ── INTERNAL BOS (priority) ───────────────────────────────────────────
        if bias == "short" and int_key_high is not None and c > int_key_high:
            bias = "long"
            key_low      = last_valid_pl
            key_high     = None
            int_key_high = None
            int_key_low  = None
            continue

        if bias == "long" and int_key_low is not None and c < int_key_low:
            bias = "short"
            key_high     = last_valid_ph
            key_low      = None
            int_key_low  = None
            int_key_high = None
            continue

        # ── EXTERNAL BOS (establishes initial bias + confirmation) ────────────
        if key_high is not None and c > key_high:
            bias = "long"
            key_low      = last_valid_pl
            key_high     = None
            int_key_high = None
            int_key_low  = None

        elif key_low is not None and c < key_low:
            bias = "short"
            key_high     = last_valid_ph
            key_low      = None
            int_key_low  = None
            int_key_high = None

    return bias

# ── LTF SIGNAL ───────────────────────────────────────────────────────────────
def get_ltf_signal(df: pd.DataFrame, bias: str):
    """
    Aggressive LTF signal: close beyond nearest confirmed swing in bias direction.
    Stop = deviation extreme (wick range from swing to signal candle).
    Scans last LTF_SCAN closed candles. Skips last (still forming).
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
    return f"{signal}_{round(entry / atr)}"

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

        bias = get_htf_bias(df_htf)
        log.info(f"  {symbol} {htf.upper()} bias: {bias}")
        if bias == "neutral":
            return

        signal, entry, stop = get_ltf_signal(df_ltf, bias)
        if not signal:
            log.info(f"  {symbol} no LTF signal")
            return

        atr_series  = calc_atr(df_ltf, ATR_LENGTH)
        current_atr = float(atr_series.iloc[-2])
        stop_dist   = abs(entry - stop)
        max_stop    = MAX_STOP_ATR * current_atr

        if stop_dist > max_stop:
            log.info(f"  {symbol} skipped — stop {stop_dist:.4f} > 2×ATR {max_stop:.4f}")
            return

        key = make_sig_key(signal, entry, current_atr)
        if last_signals.get(symbol) == key:
            log.info(f"  {symbol} duplicate, skipping")
            return
        last_signals[symbol] = key

        msg = format_alert(symbol, signal, entry, stop, current_atr, htf)
        send_telegram(msg)
        log.info(f"  ✅ Alert: {symbol} {signal} entry={entry:.4f} stop={stop:.4f}")

    except Exception as e:
        log.error(f"Error [{symbol}]: {e}")

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
def run():
    symbols = list(SYMBOL_HTF.keys())
    coins   = " · ".join(s.split("/")[0] for s in symbols)
    log.info(f"MXS Alert Bot v6 started. Monitoring: {coins}")
    send_telegram(
        "🤖 <b>MXS Alert Bot v6 started</b>\n"
        f"Monitoring: {coins}\n"
        "HTF: 1H (BTC/ETH/LINK) · 1D (SOL)\n"
        "Bias: Hybrid BOS — internal LH/HL + external pivot\n"
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
