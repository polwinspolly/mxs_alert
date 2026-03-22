"""
MXS v5 Perp Alert Bot — v2
Monitors BTC, ETH, SOL, LINK on KuCoin Futures
Replicates: HTF bias (structure) + 15M LTF flip (aggressive)
ATR stop filter, 1.6R target, 1R BE
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

# Per-symbol HTF — SOL uses 1D, all others use 4H
SYMBOL_HTF = {
    "BTC/USDT:USDT":  "4h",
    "ETH/USDT:USDT":  "4h",
    "SOL/USDT:USDT":  "1d",
    "LINK/USDT:USDT": "4h",
}

LTF            = "15m"
TARGET_R       = 1.6
BE_R           = 1.0
ATR_LENGTH     = 14
MAX_STOP_ATR   = 2.0   # Max stop distance = 2x ATR(14)
HTF_PIVOT_LB   = 5     # Pivot lookback for HTF structure
LTF_PIVOT_LB   = 3     # Pivot lookback for LTF signal (smaller = more sensitive)
LTF_SCAN       = 3     # How many recent candles to scan for a fresh signal
CHECK_INTERVAL = 60 * 15
FETCH_RETRIES  = 3
RETRY_DELAY    = 10    # Seconds between retries

# ── STATE ────────────────────────────────────────────────────────────────────
last_signals = {}  # symbol -> sig_key, prevents duplicate alerts

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

# ── HELPERS ──────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 150) -> pd.DataFrame:
    """Fetch OHLCV with retry logic."""
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not raw or len(raw) < 20:
                raise ValueError(f"Insufficient data ({len(raw) if raw else 0} candles)")
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.set_index("ts").sort_index()
            return df
        except Exception as e:
            if attempt < FETCH_RETRIES:
                log.warning(f"Fetch attempt {attempt} failed for {symbol} {timeframe}: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise


def calc_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def find_pivots(highs, lows, lb: int):
    """
    Detect swing highs and lows using a pivot lookback window.
    Returns lists of (index, price) tuples.
    """
    pivot_highs, pivot_lows = [], []
    n = len(highs)
    for i in range(lb, n - lb):
        window_h = highs[i - lb: i + lb + 1]
        window_l = lows[i - lb: i + lb + 1]
        if highs[i] == max(window_h):
            pivot_highs.append((i, highs[i]))
        if lows[i] == min(window_l):
            pivot_lows.append((i, lows[i]))
    return pivot_highs, pivot_lows


def get_htf_bias(df: pd.DataFrame) -> str:
    """
    Market structure bias using pivot highs/lows.
    Bullish = HH + HL (last 2 pivots of each type trending up)
    Bearish = LH + LL (last 2 pivots of each type trending down)
    """
    highs = df["high"].values
    lows  = df["low"].values

    pivot_highs, pivot_lows = find_pivots(highs, lows, HTF_PIVOT_LB)

    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return "neutral"

    sh1, sh2 = pivot_highs[-2][1], pivot_highs[-1][1]
    sl1, sl2 = pivot_lows[-2][1],  pivot_lows[-1][1]

    if sh2 > sh1 and sl2 > sl1:
        return "long"
    if sh2 < sh1 and sl2 < sl1:
        return "short"
    return "neutral"


def get_ltf_signal(df: pd.DataFrame, bias: str):
    """
    Detect a fresh LTF structure break in the direction of bias.
    Scans the last LTF_SCAN closed candles for a valid signal.
    Aggressive mode: signal fires on close beyond the nearest swing.

    Returns: (direction, entry, stop_level) or (None, None, None)
    """
    if bias == "neutral":
        return None, None, None

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(df)

    pivot_highs, pivot_lows = find_pivots(highs, lows, LTF_PIVOT_LB)

    # Scan recent closed candles for a signal (skip last — still forming)
    for offset in range(2, LTF_SCAN + 2):
        signal_idx = n - offset
        if signal_idx < 10:
            continue

        entry = closes[signal_idx]

        if bias == "long":
            prior_highs = [ph for ph in pivot_highs if ph[0] < signal_idx]
            if not prior_highs:
                continue
            nearest_sh_idx, nearest_sh_price = prior_highs[-1]
            if entry > nearest_sh_price:
                stop = min(lows[nearest_sh_idx:signal_idx + 1])
                return "long", entry, stop

        elif bias == "short":
            prior_lows = [pl for pl in pivot_lows if pl[0] < signal_idx]
            if not prior_lows:
                continue
            nearest_sl_idx, nearest_sl_price = prior_lows[-1]
            if entry < nearest_sl_price:
                stop = max(highs[nearest_sl_idx:signal_idx + 1])
                return "short", entry, stop

    return None, None, None


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
            log.info("Telegram message sent.")
            return
        except Exception as e:
            if attempt < 3:
                time.sleep(5)
            else:
                log.error(f"Telegram send failed after 3 attempts: {e}")


def format_alert(symbol: str, direction: str, entry: float, stop: float,
                 atr_val: float, htf: str) -> str:
    stop_dist = abs(entry - stop)
    r1  = entry + stop_dist * BE_R     if direction == "long" else entry - stop_dist * BE_R
    tp  = entry + stop_dist * TARGET_R if direction == "long" else entry - stop_dist * TARGET_R
    pct = (stop_dist / entry) * 100
    emoji = "🟢 LONG" if direction == "long" else "🔴 SHORT"
    coin  = symbol.split("/")[0]
    now   = datetime.now(timezone.utc).strftime("%H:%M UTC")

    # Dynamic decimal places based on price magnitude
    decimals = 2 if entry > 100 else (4 if entry > 1 else 6)
    fmt = f".{decimals}f"

    return (
        f"<b>⚡ MXS SIGNAL — {coin} {emoji}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🕐 <b>Time:</b> {now}\n"
        f"📍 <b>Entry:</b> {entry:{fmt}}\n"
        f"🛑 <b>Stop:</b> {stop:{fmt}} ({pct:.2f}%)\n"
        f"🎯 <b>BE (1R):</b> {r1:{fmt}}\n"
        f"🎯 <b>TP (1.6R):</b> {tp:{fmt}}\n"
        f"📊 <b>ATR(14):</b> {atr_val:{fmt}}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<i>MXS v5 Perp · {htf.upper()} bias + 15M flip</i>"
    )


# ── MAIN ─────────────────────────────────────────────────────────────────────

def check_symbol(symbol: str):
    htf = SYMBOL_HTF.get(symbol, "4h")
    try:
        log.info(f"Checking {symbol}...")

        # Fetch data
        df_htf = fetch_ohlcv(symbol, htf, limit=150)
        df_ltf = fetch_ohlcv(symbol, LTF, limit=100)

        # Step 1: HTF bias
        bias = get_htf_bias(df_htf)
        log.info(f"  {symbol} {htf.upper()} bias: {bias}")
        if bias == "neutral":
            return

        # Step 2: LTF signal
        signal, entry, stop = get_ltf_signal(df_ltf, bias)
        if not signal:
            log.info(f"  {symbol} no LTF signal")
            return

        # Step 3: ATR stop filter
        atr_series  = calc_atr(df_ltf, ATR_LENGTH)
        current_atr = atr_series.iloc[-2]
        stop_dist   = abs(entry - stop)
        max_stop    = MAX_STOP_ATR * current_atr

        if stop_dist > max_stop:
            log.info(f"  {symbol} signal skipped — stop too wide ({stop_dist:.4f} > 2xATR {max_stop:.4f})")
            return

        # Step 4: Deduplicate
        sig_key = f"{signal}_{round(entry, 4)}"
        if last_signals.get(symbol) == sig_key:
            log.info(f"  {symbol} duplicate signal, skipping.")
            return

        last_signals[symbol] = sig_key

        # Step 5: Alert
        msg = format_alert(symbol, signal, entry, stop, current_atr, htf)
        send_telegram(msg)
        log.info(f"  Alert sent: {symbol} {signal} @ {entry}")

    except Exception as e:
        log.error(f"Error checking {symbol} ({htf}): {e}")


def run():
    symbols = list(SYMBOL_HTF.keys())
    log.info("MXS Alert Bot v2 started. Monitoring: " + ", ".join(s.split("/")[0] for s in symbols))
    send_telegram(
        "🤖 <b>MXS Alert Bot v2 started</b>\n"
        "Monitoring: BTC · ETH · SOL · LINK\n"
        "HTF: 4H (BTC/ETH/LINK) · 1D (SOL)\n"
        "Entry: 15M flip · Target: 1.6R · BE: 1R"
    )

    while True:
        for symbol in symbols:
            check_symbol(symbol)
            time.sleep(5)  # Delay between symbols to avoid rate limits

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        log.info(f"Cycle complete. Next check in 15 min. [{now}]")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run()
