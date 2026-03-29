"""
MXS v5 Perp Alert Bot — v9
Monitors BTC, ETH, SOL, LINK on KuCoin Futures

NEW in v9: Active Trade Tracker
  - Once a signal fires, the trade is tracked until closed
  - No new signals for that coin while a trade is active (reduces noise)
  - Configurable exit conditions (toggle ON/OFF in TRADE_CONDITIONS below):
      C1_STOP_LOSS    — notify and close when price hits stop
      C2_TARGET       — notify and close when price hits TP (1.6R)
      C3_HTF_BIAS     — notify and close when HTF bias flips against trade
      C4_BREAKEVEN    — move stop to BE when price hits 1R, close if BE hit
  - Each exit notification includes R achieved

HTF Bias: Hybrid 2-layer BOS
  INTERNAL: direction-aware LH/HL tracking → fast flip
  EXTERNAL: confirmed pivot with forward confirmation → establishes bias

LTF Signal: Multi-pivot walk (v7 fix)
  Walks pivots newest→oldest to find first broken level

Stop: Deviation extreme | Target: 1.6R | BE: 1R
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
HTF_PIVOT_LB   = 5
LTF_PIVOT_LB   = 3
LTF_SCAN       = 4
CHECK_INTERVAL = 60 * 15
FETCH_RETRIES  = 3
RETRY_DELAY    = 10

# ── TRADE EXIT CONDITIONS (set to False to disable) ───────────────────────────
TRADE_CONDITIONS = {
    "C1_STOP_LOSS": True,   # Close + notify when price hits stop loss
    "C2_TARGET":    True,   # Close + notify when price hits TP (1.6R)
    "C3_HTF_BIAS":  True,   # Close + notify when HTF bias flips against trade
    "C4_BREAKEVEN": True,   # Move stop to BE at 1R, close + notify if BE hit
}

# ── STATE ────────────────────────────────────────────────────────────────────
last_signals  = {}   # symbol -> sig_key (dedup)
active_trades = {}   # symbol -> trade dict

# Trade dict structure:
# {
#   "direction": "long" | "short",
#   "entry":     float,
#   "stop":      float,       # current stop (moves to BE after 1R)
#   "orig_stop": float,       # original stop (for R calculation)
#   "be":        float,       # breakeven price
#   "tp":        float,       # target price
#   "stop_dist": float,       # original stop distance (1R)
#   "htf":       str,         # HTF timeframe
#   "be_moved":  bool,        # whether stop has been moved to BE
#   "time":      str,         # entry time UTC
# }

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
                log.warning(f"Fetch attempt {attempt} [{symbol} {timeframe}]: {e}. Retry in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                raise

def fetch_ticker(symbol: str) -> float:
    """Get current mid price."""
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            ticker = exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            if attempt < FETCH_RETRIES:
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
    ph, pl = [], []
    n = len(highs)
    for i in range(lb, n - lb):
        if highs[i] == max(highs[i - lb: i + lb + 1]):
            ph.append((i, float(highs[i])))
        if lows[i] == min(lows[i - lb: i + lb + 1]):
            pl.append((i, float(lows[i])))
    return ph, pl

# ── HTF BIAS ─────────────────────────────────────────────────────────────────
def get_htf_bias(df: pd.DataFrame) -> str:
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(df)

    ext_ph, ext_pl = find_pivots(highs, lows, HTF_PIVOT_LB)
    if not ext_ph or not ext_pl:
        return "neutral"

    ph_at = {i: p for i, p in ext_ph}
    pl_at = {i: p for i, p in ext_pl}

    bias = "neutral"
    key_high = key_low = None
    last_valid_ph = last_valid_pl = None
    int_key_high = int_key_low = None

    def confirms_down(idx):
        f = lows[idx+1:min(idx+5,n)]
        return len(f) > 0 and min(f) < lows[idx]

    def confirms_up(idx):
        f = highs[idx+1:min(idx+5,n)]
        return len(f) > 0 and max(f) > highs[idx]

    start = min(ext_ph[0][0], ext_pl[0][0])

    for i in range(max(start, 2), n - 1):
        c = closes[i]

        if lows[i-1] < lows[i-2] and lows[i-1] < lows[i]:
            if bias == "long":
                int_key_low = lows[i-1]

        if highs[i-1] > highs[i-2] and highs[i-1] > highs[i]:
            if bias == "short":
                int_key_high = highs[i-1]

        if i in ph_at:
            if confirms_down(i):
                last_valid_ph = ph_at[i]
                if bias in ("short", "neutral"):
                    key_high = last_valid_ph

        if i in pl_at:
            if confirms_up(i):
                last_valid_pl = pl_at[i]
                if bias in ("long", "neutral"):
                    key_low = last_valid_pl

        if bias == "short" and int_key_high is not None and c > int_key_high:
            bias = "long"
            key_low = last_valid_pl; key_high = None
            int_key_high = int_key_low = None
            continue

        if bias == "long" and int_key_low is not None and c < int_key_low:
            bias = "short"
            key_high = last_valid_ph; key_low = None
            int_key_low = int_key_high = None
            continue

        if key_high is not None and c > key_high:
            bias = "long"
            key_low = last_valid_pl; key_high = None
            int_key_high = int_key_low = None

        elif key_low is not None and c < key_low:
            bias = "short"
            key_high = last_valid_ph; key_low = None
            int_key_low = int_key_high = None

    return bias

# ── LTF SIGNAL ───────────────────────────────────────────────────────────────
def get_ltf_signal(df: pd.DataFrame, bias: str):
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
            for sh_idx, sh_price in reversed(prior_highs):
                if entry > sh_price:
                    stop = float(min(lows[sh_idx: signal_idx + 1]))
                    return "long", float(entry), stop

        elif bias == "short":
            prior_lows = [pl for pl in pivot_lows if pl[0] < signal_idx]
            for sl_idx, sl_price in reversed(prior_lows):
                if entry < sl_price:
                    stop = float(max(highs[sl_idx: signal_idx + 1]))
                    return "short", float(entry), stop

    return None, None, None

# ── HELPERS ──────────────────────────────────────────────────────────────────
def calc_r(trade: dict, price: float) -> float:
    """Calculate R multiple achieved at a given price."""
    dist = price - trade["entry"] if trade["direction"] == "long" else trade["entry"] - price
    return dist / trade["stop_dist"]

def fmt_price(price: float) -> str:
    d = 2 if price > 100 else (4 if price > 1 else 6)
    return f"{price:.{d}f}"

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
                log.error(f"Telegram failed: {e}")

# ── FORMAT MESSAGES ───────────────────────────────────────────────────────────
def fmt_signal(symbol: str, trade: dict) -> str:
    coin  = symbol.split("/")[0]
    emoji = "🟢 LONG" if trade["direction"] == "long" else "🔴 SHORT"
    pct   = (trade["stop_dist"] / trade["entry"]) * 100
    return (
        f"<b>⚡ MXS SIGNAL — {coin} {emoji}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🕐 <b>Time:</b> {trade['time']}\n"
        f"📍 <b>Entry:</b> {fmt_price(trade['entry'])}\n"
        f"🛑 <b>Stop:</b> {fmt_price(trade['orig_stop'])} ({pct:.2f}%)\n"
        f"🎯 <b>BE (1R):</b> {fmt_price(trade['be'])}\n"
        f"🎯 <b>TP (1.6R):</b> {fmt_price(trade['tp'])}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<i>MXS v5 Perp · {trade['htf'].upper()} bias + 15M flip</i>"
    )

def fmt_exit(symbol: str, trade: dict, reason: str, price: float) -> str:
    coin    = symbol.split("/")[0]
    r_val   = calc_r(trade, price)
    r_str   = f"{r_val:+.2f}R"
    now     = datetime.now(timezone.utc).strftime("%H:%M UTC")
    icons   = {
        "STOP LOSS":    "🔴",
        "TARGET HIT":   "🟢",
        "HTF BIAS FLIP":"🔄",
        "BREAKEVEN HIT":"⚖️",
    }
    icon = icons.get(reason, "⚠️")
    return (
        f"<b>{icon} {reason} — {coin}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🕐 <b>Time:</b> {now}\n"
        f"📍 <b>Entry:</b> {fmt_price(trade['entry'])}\n"
        f"💰 <b>Exit:</b> {fmt_price(price)}\n"
        f"📊 <b>Result:</b> <b>{r_str}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<i>Trade closed</i>"
    )

def fmt_be_moved(symbol: str, trade: dict) -> str:
    coin = symbol.split("/")[0]
    now  = datetime.now(timezone.utc).strftime("%H:%M UTC")
    return (
        f"<b>⚖️ BREAKEVEN MOVED — {coin}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🕐 <b>Time:</b> {now}\n"
        f"📍 <b>Entry:</b> {fmt_price(trade['entry'])}\n"
        f"🛡 <b>Stop → BE:</b> {fmt_price(trade['stop'])}\n"
        f"🎯 <b>TP still active:</b> {fmt_price(trade['tp'])}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<i>Risk eliminated</i>"
    )

# ── TRADE MANAGEMENT ──────────────────────────────────────────────────────────
def open_trade(symbol: str, direction: str, entry: float, stop: float, htf: str):
    stop_dist = abs(entry - stop)
    be  = entry + stop_dist * BE_R     if direction == "long" else entry - stop_dist * BE_R
    tp  = entry + stop_dist * TARGET_R if direction == "long" else entry - stop_dist * TARGET_R
    trade = {
        "direction": direction,
        "entry":     entry,
        "stop":      stop,
        "orig_stop": stop,
        "be":        be,
        "tp":        tp,
        "stop_dist": stop_dist,
        "htf":       htf,
        "be_moved":  False,
        "time":      datetime.now(timezone.utc).strftime("%H:%M UTC"),
    }
    active_trades[symbol] = trade
    log.info(f"  📂 Trade opened: {symbol} {direction} entry={fmt_price(entry)} stop={fmt_price(stop)}")
    return trade

def check_active_trade(symbol: str, current_price: float, current_bias: str) -> bool:
    """
    Check exit conditions for an active trade.
    Returns True if trade was closed (so caller skips signal scanning).
    Returns False if trade is still open or no active trade.
    """
    trade = active_trades.get(symbol)
    if not trade:
        return False

    direction = trade["direction"]
    entry     = trade["entry"]
    stop      = trade["stop"]
    be        = trade["be"]
    tp        = trade["tp"]
    be_moved  = trade["be_moved"]

    log.info(f"  📊 Tracking {symbol} {direction} | price={fmt_price(current_price)} stop={fmt_price(stop)} tp={fmt_price(tp)}")

    # C4: Breakeven — move stop to BE when price hits 1R (check before stop)
    if TRADE_CONDITIONS["C4_BREAKEVEN"] and not be_moved:
        be_triggered = (direction == "long"  and current_price >= be) or \
                       (direction == "short" and current_price <= be)
        if be_triggered:
            active_trades[symbol]["stop"]     = entry  # move stop to entry
            active_trades[symbol]["be_moved"] = True
            log.info(f"  ⚖️ BE moved: {symbol}")
            send_telegram(fmt_be_moved(symbol, active_trades[symbol]))
            # Don't close yet — let trade run to TP
            stop = entry  # update local ref for subsequent checks this cycle

    # C1: Stop loss
    if TRADE_CONDITIONS["C1_STOP_LOSS"]:
        stopped = (direction == "long"  and current_price <= stop) or \
                  (direction == "short" and current_price >= stop)
        if stopped:
            reason = "BREAKEVEN HIT" if be_moved else "STOP LOSS"
            send_telegram(fmt_exit(symbol, trade, reason, current_price))
            del active_trades[symbol]
            log.info(f"  ❌ Trade closed ({reason}): {symbol}")
            return True

    # C2: Target hit
    if TRADE_CONDITIONS["C2_TARGET"]:
        targeted = (direction == "long"  and current_price >= tp) or \
                   (direction == "short" and current_price <= tp)
        if targeted:
            send_telegram(fmt_exit(symbol, trade, "TARGET HIT", current_price))
            del active_trades[symbol]
            log.info(f"  ✅ Trade closed (TARGET): {symbol}")
            return True

    # C3: HTF bias flip against trade
    if TRADE_CONDITIONS["C3_HTF_BIAS"]:
        bias_flipped = (direction == "long"  and current_bias == "short") or \
                       (direction == "short" and current_bias == "long")
        if bias_flipped:
            send_telegram(fmt_exit(symbol, trade, "HTF BIAS FLIP", current_price))
            del active_trades[symbol]
            log.info(f"  🔄 Trade closed (BIAS FLIP): {symbol}")
            return True

    return False  # trade still active

# ── CHECK SYMBOL ─────────────────────────────────────────────────────────────
def check_symbol(symbol: str):
    htf = SYMBOL_HTF.get(symbol, "1h")
    try:
        log.info(f"Checking {symbol}...")

        df_htf = fetch_ohlcv(symbol, htf, limit=150)
        df_ltf = fetch_ohlcv(symbol, LTF, limit=100)

        bias = get_htf_bias(df_htf)
        log.info(f"  {symbol} {htf.upper()} bias: {bias}")

        # Get current price for trade tracking
        current_price = float(df_ltf["close"].iloc[-1])

        # Check active trade first — if trade is still open, skip signal scanning
        if symbol in active_trades:
            trade_closed = check_active_trade(symbol, current_price, bias)
            if not trade_closed:
                log.info(f"  {symbol} trade active, skipping signal scan")
                return
            # Trade was closed — fall through to scan for new signal

        if bias == "neutral":
            return

        # Scan for new LTF signal
        signal, entry, stop = get_ltf_signal(df_ltf, bias)
        if not signal:
            log.info(f"  {symbol} no LTF signal")
            return

        atr_series  = calc_atr(df_ltf, ATR_LENGTH)
        current_atr = float(atr_series.iloc[-2])

        # Dedup
        key = make_sig_key(signal, entry, current_atr)
        if last_signals.get(symbol) == key:
            log.info(f"  {symbol} duplicate signal, skipping")
            return
        last_signals[symbol] = key

        # Open trade and send alert
        trade = open_trade(symbol, signal, entry, stop, htf)
        send_telegram(fmt_signal(symbol, trade))
        log.info(f"  ✅ Signal + trade opened: {symbol} {signal} @ {fmt_price(entry)}")

    except Exception as e:
        log.error(f"Error [{symbol}]: {e}")

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
def run():
    symbols = list(SYMBOL_HTF.keys())
    coins   = " · ".join(s.split("/")[0] for s in symbols)

    # Build condition summary for startup message
    cond_lines = []
    labels = {
        "C1_STOP_LOSS": "C1 Stop Loss",
        "C2_TARGET":    "C2 Target (1.6R)",
        "C3_HTF_BIAS":  "C3 HTF Bias Flip",
        "C4_BREAKEVEN": "C4 Breakeven (1R)",
    }
    for k, label in labels.items():
        status = "✅" if TRADE_CONDITIONS[k] else "❌"
        cond_lines.append(f"{status} {label}")

    log.info(f"MXS Alert Bot v9 started. Monitoring: {coins}")
    send_telegram(
        "🤖 <b>MXS Alert Bot v9 started</b>\n"
        f"Monitoring: {coins}\n"
        "HTF: 1H (BTC/ETH/LINK) · 1D (SOL)\n"
        "━━━━━━━━━━━━━━━━\n"
        "<b>Exit Conditions:</b>\n" +
        "\n".join(cond_lines)
    )

    while True:
        for symbol in symbols:
            check_symbol(symbol)
            time.sleep(5)
        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        active = [s.split("/")[0] for s in active_trades]
        log.info(f"Cycle done. Next in 15 min [{now}] | Active trades: {active or 'none'}")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run()
