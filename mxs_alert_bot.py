"""
MXS v5 Perp Alert Bot
Monitors BTC, ETH, SOL, LINK on Bybit
Replicates: 4H HTF bias + 15M LTF flip + ATR stop validation
Sends Telegram alerts with entry, stop, TP (1.6R), BE (1R)
"""

import ccxt
import pandas as pd
import time
import logging
import os
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "LINK/USDT:USDT"]
LTF = "15m"

# Per-symbol HTF — SOL uses 1D, all others use 4H
SYMBOL_HTF = {
    "BTC/USDT:USDT":  "4h",
    "ETH/USDT:USDT":  "4h",
    "SOL/USDT:USDT":  "1d",
    "LINK/USDT:USDT": "4h",
}

# Perp settings
TARGET_R = 1.6
BE_R = 1.0
ATR_LENGTH = 14
MAX_STOP_ATR = 2.0        # Max stop = 2× ATR (filter ON for perp)
SWING_LOOKBACK = 5        # Candles to look back for swing high/low

CHECK_INTERVAL = 60 * 15  # Every 15 minutes (seconds)

# Track last signal per symbol to avoid duplicate alerts
last_signals = {}

# ── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── EXCHANGE ─────────────────────────────────────────────────────────────────
exchange = ccxt.bybit({
    "options": {"defaultType": "future"},
    "enableRateLimit": True,
})

# ── HELPERS ──────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts")


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def get_htf_bias(df4h: pd.DataFrame) -> str:
    """
    Determine 4H market structure bias.
    Bullish = recent HH + HL pattern
    Bearish = recent LH + LL pattern
    Returns: 'long', 'short', or 'neutral'
    """
    closes = df4h["close"].values
    highs = df4h["high"].values
    lows = df4h["low"].values

    # Find last 3 swing highs and lows (simple pivot detection)
    swing_highs = []
    swing_lows = []
    lb = SWING_LOOKBACK

    for i in range(lb, len(df4h) - 1):
        if highs[i] == max(highs[i - lb:i + lb + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - lb:i + lb + 1]):
            swing_lows.append(lows[i])

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "neutral"

    # Last 2 swing highs and lows
    sh1, sh2 = swing_highs[-2], swing_highs[-1]
    sl1, sl2 = swing_lows[-2], swing_lows[-1]

    bullish = sh2 > sh1 and sl2 > sl1   # HH + HL
    bearish = sh2 < sh1 and sl2 < sl1   # LH + LL

    if bullish:
        return "long"
    elif bearish:
        return "short"
    return "neutral"


def get_ltf_signal(df15m: pd.DataFrame, bias: str):
    """
    Detect 15M LTF structure break in direction of bias.
    Aggressive mode: signal on candle that closes beyond swing point.
    Returns: (signal, entry_price, swing_extreme) or (None, None, None)
    """
    if bias == "neutral":
        return None, None, None

    highs = df15m["high"].values
    lows = df15m["low"].values
    closes = df15m["close"].values
    lb = SWING_LOOKBACK

    # Use second-to-last closed candle as signal candle (last is still forming)
    signal_idx = len(df15m) - 2

    if bias == "long":
        # Find most recent swing high before signal candle
        recent_swing_high = max(highs[max(0, signal_idx - 20):signal_idx - 1])
        entry = closes[signal_idx]
        if entry > recent_swing_high:
            # Find the wick low of the broken swing (stop placement)
            swing_low = min(lows[max(0, signal_idx - 10):signal_idx])
            return "long", entry, swing_low

    elif bias == "short":
        # Find most recent swing low before signal candle
        recent_swing_low = min(lows[max(0, signal_idx - 20):signal_idx - 1])
        entry = closes[signal_idx]
        if entry < recent_swing_low:
            # Find the wick high of the broken swing (stop placement)
            swing_high = max(highs[max(0, signal_idx - 10):signal_idx])
            return "short", entry, swing_high

    return None, None, None


def validate_stop(entry: float, stop: float, current_atr: float, direction: str) -> bool:
    """Check stop distance ≤ 2× ATR"""
    stop_dist = abs(entry - stop)
    max_stop = MAX_STOP_ATR * current_atr
    return stop_dist <= max_stop


def send_telegram(message: str):
    """Send message via Telegram Bot API"""
    import urllib.request
    import urllib.parse
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }).encode()
    try:
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=10)
        log.info("Telegram message sent.")
    except Exception as e:
        log.error(f"Telegram send failed: {e}")


def format_alert(symbol: str, direction: str, entry: float, stop: float, atr_val: float, htf: str = "4h") -> str:
    stop_dist = abs(entry - stop)
    tp1 = entry + stop_dist * BE_R if direction == "long" else entry - stop_dist * BE_R
    tp2 = entry + stop_dist * TARGET_R if direction == "long" else entry - stop_dist * TARGET_R
    stop_pct = (stop_dist / entry) * 100
    emoji = "🟢 LONG" if direction == "long" else "🔴 SHORT"
    coin = symbol.split("/")[0]
    now = datetime.utcnow().strftime("%H:%M UTC")

    return (
        f"<b>⚡ MXS SIGNAL — {coin} {emoji}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🕐 <b>Time:</b> {now}\n"
        f"📍 <b>Entry:</b> {entry:.4f}\n"
        f"🛑 <b>Stop:</b> {stop:.4f} ({stop_pct:.2f}%)\n"
        f"🎯 <b>BE (1R):</b> {tp1:.4f}\n"
        f"🎯 <b>TP (1.6R):</b> {tp2:.4f}\n"
        f"📊 <b>ATR:</b> {atr_val:.4f}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<i>MXS v5 Perp · {htf.upper()} bias + 15M flip</i>"
    )


# ── MAIN LOOP ────────────────────────────────────────────────────────────────
def check_symbol(symbol: str):
    try:
        log.info(f"Checking {symbol}...")

        # Fetch candles
        htf = SYMBOL_HTF.get(symbol, "4h")
        df_htf = fetch_ohlcv(symbol, htf, limit=100)
        df15m = fetch_ohlcv(symbol, LTF, limit=60)

        # Step 1: HTF bias
        bias = get_htf_bias(df_htf)
        log.info(f"{symbol} {htf.upper()} bias: {bias}")

        if bias == "neutral":
            return

        # Step 2: LTF signal
        signal, entry, swing_extreme = get_ltf_signal(df15m, bias)

        if not signal:
            return

        # Step 3: ATR stop validation
        atr_series = atr(df15m, ATR_LENGTH)
        current_atr = atr_series.iloc[-2]

        stop = swing_extreme
        if not validate_stop(entry, stop, current_atr, signal):
            log.info(f"{symbol} signal skipped — stop too wide ({abs(entry-stop):.4f} > 2×ATR {2*current_atr:.4f})")
            return

        # Deduplicate — don't alert same direction twice in a row
        sig_key = f"{symbol}_{signal}_{round(entry, 2)}"
        if last_signals.get(symbol) == sig_key:
            log.info(f"{symbol} duplicate signal, skipping.")
            return

        last_signals[symbol] = sig_key

        # Send alert
        msg = format_alert(symbol, signal, entry, stop, current_atr, htf)
        send_telegram(msg)
        log.info(f"Alert sent for {symbol} {signal}")

    except Exception as e:
        log.error(f"Error checking {symbol}: {e}")


def run():
    log.info("MXS Alert Bot started. Monitoring: " + ", ".join(SYMBOLS))
    send_telegram("🤖 <b>MXS Alert Bot started</b>\nMonitoring: BTC, ETH, SOL, LINK\nSettings: 4H bias + 15M flip · Perp (1.6R target)")

    while True:
        for symbol in SYMBOLS:
            check_symbol(symbol)
            time.sleep(2)  # Small delay between symbols

        next_check = datetime.utcnow().strftime("%H:%M UTC")
        log.info(f"Cycle complete. Next check in 15 min. [{next_check}]")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run()
