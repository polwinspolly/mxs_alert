# MXS Alert Bot — Setup Guide

## Files Needed
- mxs_alert_bot.py
- requirements.txt
- Procfile

---

## Step 1: Create Your Telegram Bot (5 min)

1. Open Telegram → search **@BotFather**
2. Send `/newbot`
3. Give it any name (e.g. "MXS Alerts") and username (e.g. "mxs_alerts_bot")
4. BotFather gives you a **token** like: `7123456789:AAF-xxxxxxxxxxxxxxxxxxx`
5. Save this — it's your TELEGRAM_TOKEN

### Get your Chat ID:
1. Start a chat with your new bot (click Start)
2. Open this URL in browser (replace YOUR_TOKEN):
   `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
3. Look for `"chat":{"id":XXXXXXXXX}` — that number is your TELEGRAM_CHAT_ID

---

## Step 2: Deploy to Railway (free, no PC needed)

1. Go to **railway.app** → sign up with GitHub
2. Click **New Project → Deploy from GitHub repo**
   - OR: click **New Project → Empty Project**, then drag-upload the 3 files
3. Once deployed, go to your service → **Variables** tab
4. Add these 2 environment variables:
   ```
   TELEGRAM_TOKEN = your_token_here
   TELEGRAM_CHAT_ID = your_chat_id_here
   ```
5. Railway will auto-detect the Procfile and start the worker
6. Check **Logs** tab — you should see "MXS Alert Bot started"
7. You'll get a Telegram message confirming it's live

---

## What Alerts Look Like

```
⚡ MXS SIGNAL — BTC 🟢 LONG
━━━━━━━━━━━━━━━━
🕐 Time: 14:15 UTC
📍 Entry: 84250.0000
🛑 Stop: 83800.0000 (0.53%)
🎯 BE (1R): 84700.0000
🎯 TP (1.6R): 84970.0000
📊 ATR: 420.0000
━━━━━━━━━━━━━━━━
MXS v5 Perp · 4H bias + 15M flip
```

---

## Signal Logic (matches MXS v5 Perp settings)

| Setting | Value |
|---|---|
| HTF Timeframe | 4H |
| LTF Timeframe | 15M |
| Trade Direction | Both |
| LTF Aggressive Signals | ON |
| Stop Placement | Deviation Extreme (swing wick) |
| Max Stop Distance | 2× ATR(14) |
| Target | 1.6R |
| Breakeven Trigger | 1R |

---

## Troubleshooting

**Bot not sending messages?**
- Double-check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in Railway variables
- Make sure you clicked Start on your bot in Telegram first

**"Symbol not found" error?**
- Bybit perp symbols occasionally change naming — check ccxt docs

**Too many signals / noise?**
- Increase SWING_LOOKBACK from 5 to 7 in the script

---

## Running Locally Instead (optional)

If you want to run on your PC instead of Railway:

```bash
pip install ccxt pandas

# Set your tokens in the script directly (lines 20-21)
# OR set environment variables:
set TELEGRAM_TOKEN=your_token      # Windows
set TELEGRAM_CHAT_ID=your_chat_id

python mxs_alert_bot.py
```

Keep terminal open while trading.
