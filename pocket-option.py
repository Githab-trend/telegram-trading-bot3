# -*- coding: utf-8 -*-
"""
Full PRO Telegram Signal Bot (Binance spot klines)
Features:
- Dynamic Volume & Volatility filters (not hard-coded; rolling stats–based)
- Multi-confirmation: at least N of {EMA trend, MACD side, RSI(50)} must agree
- Confidence score + per-indicator breakdown
- Persist bars filter
- Cooldown + min-price-move gating (per symbol)
- First-pass memory (doesn't send on very first check)
- Per-symbol signal history (last 5)
- Self-healing polling (auto-reconnect on network errors)
- Admin commands: /set, /status, /ping
- Optional chart sending with EMAs (toggleable)
"""

import time
import pandas as pd
import numpy as np
import telebot
import threading
import requests
import io
import traceback
import os
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========= Telegram Bot Setup =========
bot_token = '8100566090:AAGky2qeO6yif0vDjnP7NX-AFZ07FEZgs6w'
chat_id = '-1002794962661'                                  # <- Քո channel/group/chat id-ը (կամ քո user id)
bot = telebot.TeleBot(bot_token, parse_mode="HTML")

# ========= Runtime Config (editable via /set) =========
CFG = {
    # Markets
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "RPLUSDT"],   # /set symbols=BTCUSDT,ETHUSDT
    "intervals": ["1m", "5m", "15m", "30m", "1h"],

    # Signal gating
    "conf_min": 80,             # minimal confidence to allow
    "persist_bars": 2,          # last N bars must keep direction
    "agreement_needed": 2,      # need N of 3 indicators to align (EMA/MACD/RSI)

    # Cooldown / resend rules
    "cooldown_sec": 10*60,      # minimal time between same key signals
    "min_price_move_pct": 0.003, # minimal relative move to resend same key

    # Check cadence
    "check_interval_sec": 15*60,  # main looping sleep

    # Volume (dynamic): current vol z-score vs last N bars mean/std
    "vol_window": 30,           # rolling window
    "vol_z": 0.8,               # require z-score >= vol_z  (change via /set vol_z=1.0)

    # Volatility (dynamic): current TR vs rolling mean TR * multiplier
    "tr_window": 30,
    "tr_mult": 1.10,            # require TR_now >= meanTR * tr_mult

    # Indicators params
    "ema_fast": 5,
    "ema_slow": 20,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # Chart
    "send_chart": True,         # also send a small chart with EMAs
    "chart_bars": 120,
}

# ========= State =========
last_signal_info = {}  # symbol -> {"key": (dir, tf), "ts": float, "price": float}
first_check_done = {}  # symbol -> bool
signal_history   = {}  # symbol -> list of dicts (last 5 signals)
start_time       = time.time()

def _init_state():
    for s in CFG["symbols"]:
        last_signal_info[s] = {"key": None, "ts": 0.0, "price": None}
        first_check_done[s] = False
        signal_history[s]   = []
_init_state()

# ========= Binance fetch =========
def fetch_binance_klines(symbol, interval, limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            print(f"[BINANCE] {symbol} {interval} -> {r.text}")
            return pd.DataFrame()
        data = r.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'timestamp','open','high','low','close','volume',
            'close_time','quote_asset_volume','number_of_trades',
            'taker_buy_base_volume','taker_buy_quote_volume','ignore'
        ])
        for col in ['open','high','low','close','volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df[['timestamp','open','high','low','close','volume']]
    except Exception as e:
        print(f"[BINANCE ERR] {symbol} {interval}: {e}")
        return pd.DataFrame()

# ========= Indicators =========
def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_indicators(df):
    close = df['close']
    ema_fast = close.ewm(span=CFG["ema_fast"], adjust=False).mean()
    ema_slow = close.ewm(span=CFG["ema_slow"], adjust=False).mean()
    rsi = compute_rsi(close, CFG["rsi_period"])
    macd_line, signal_line = compute_macd(close, CFG["macd_fast"], CFG["macd_slow"], CFG["macd_signal"])
    return ema_fast, ema_slow, rsi, macd_line, signal_line

# ========= Dynamic Filters =========
def volume_ok(df):
    w = CFG["vol_window"]
    if len(df) < w + 5:  # need enough data
        return True  # don’t block if not enough data
    vol = df['volume']
    mu = vol.rolling(w).mean().iloc[-1]
    sd = vol.rolling(w).std().iloc[-1]
    if sd is None or sd == 0 or np.isnan(sd):
        return True
    z = (vol.iloc[-1] - mu) / sd
    return z >= CFG["vol_z"], z

def tr_ok(df):
    w = CFG["tr_window"]
    high = df['high']; low = df['low']; close = df['close']
    tr = (high - low).abs()
    if len(tr) < w + 5:
        return True, np.nan, np.nan
    mean_tr = tr.rolling(w).mean().iloc[-1]
    now_tr  = tr.iloc[-1]
    if np.isnan(mean_tr) or mean_tr == 0:
        return True, now_tr, mean_tr
    return (now_tr >= mean_tr * CFG["tr_mult"]), now_tr, mean_tr

# ========= Confidence =========
def confidence_from_indicators(ema_fast, ema_slow, rsi, macd_line, signal_line, direction):
    w_trend, w_mom, w_rsi, w_cross = 0.40, 0.30, 0.15, 0.15
    trend_ok = (
        (ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-1] > ema_slow.iloc[-2] and direction == "Buy") or
        (ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-1] < ema_slow.iloc[-2] and direction == "Sell")
    )
    trend_score = 1.0 if trend_ok else 0.0

    hist_now  = macd_line.iloc[-1] - signal_line.iloc[-1]
    hist_prev = macd_line.iloc[-2] - signal_line.iloc[-2]
    mom_up    = hist_now > hist_prev
    mom_down  = hist_now < hist_prev
    mom_score = 1.0 if (mom_up and direction=="Buy") or (mom_down and direction=="Sell") else 0.0

    rsi_now   = float(rsi.iloc[-1])
    rsi_score = 0.0
    if direction == "Buy" and 35 <= rsi_now <= 75:
        rsi_score = 1.0 - abs(rsi_now - 55) / 20.0
    elif direction == "Sell" and 25 <= rsi_now <= 65:
        rsi_score = 1.0 - abs(rsi_now - 45) / 20.0
    rsi_score = max(0.0, min(1.0, rsi_score))

    crossed_up   = (macd_line.iloc[-2] <= signal_line.iloc[-2]) and (macd_line.iloc[-1] > signal_line.iloc[-1])
    crossed_down = (macd_line.iloc[-2] >= signal_line.iloc[-2]) and (macd_line.iloc[-1] < signal_line.iloc[-1])
    cross_ok = (crossed_up and direction=="Buy") or (crossed_down and direction=="Sell")
    cross_score = 1.0 if cross_ok else 0.0

    score = (w_trend*trend_score + w_mom*mom_score + w_rsi*rsi_score + w_cross*cross_score) * 100.0
    return int(round(score)), (trend_score, mom_score, rsi_score, cross_score)

# ========= Analyzer per symbol =========
def analyze_symbol(symbol):
    best = {"dir": None, "price": None, "score": -1, "tf": None, "breakdown": None, "extras": {}}

    for tf in CFG["intervals"]:
        df = fetch_binance_klines(symbol, tf, limit=max(200, CFG["chart_bars"]))
        if df.empty or len(df) < 50:
            continue

        ema_f, ema_s, rsi, macd_l, macd_s = compute_indicators(df)
        price = df['close'].iloc[-1]

        # direction
        buy  = (ema_f.iloc[-1] > ema_s.iloc[-1]) and (macd_l.iloc[-1] > macd_s.iloc[-1])
        sell = (ema_f.iloc[-1] < ema_s.iloc[-1]) and (macd_l.iloc[-1] < macd_s.iloc[-1])
        direction = "Buy" if buy else ("Sell" if sell else None)
        if not direction:
            continue

        # persist bars
        ok_persist = all(
            (ema_f.iloc[-i] > ema_s.iloc[-i]) if direction=="Buy" else (ema_f.iloc[-i] < ema_s.iloc[-i])
            for i in range(1, CFG["persist_bars"]+1)
        )
        if not ok_persist:
            continue

        # agreement: N of {EMA trend, MACD side, RSI side}
        agree = 0
        if direction == "Buy":
            if ema_f.iloc[-1] > ema_s.iloc[-1]: agree += 1
            if macd_l.iloc[-1] > macd_s.iloc[-1]: agree += 1
            if rsi.iloc[-1] > 50: agree += 1
        else:
            if ema_f.iloc[-1] < ema_s.iloc[-1]: agree += 1
            if macd_l.iloc[-1] < macd_s.iloc[-1]: agree += 1
            if rsi.iloc[-1] < 50: agree += 1
        if agree < CFG["agreement_needed"]:
            continue

        # dynamic volume
        vol_ok, vol_z = volume_ok(df)
        if not vol_ok:
            continue

        # dynamic volatility (true range vs rolling mean)
        tr_ok_flag, now_tr, mean_tr = tr_ok(df)
        if not tr_ok_flag:
            continue

        # confidence
        score, breakdown = confidence_from_indicators(ema_f, ema_s, rsi, macd_l, macd_s, direction)
        if score > best["score"]:
            best.update({
                "dir": direction, "price": float(price), "score": score, "tf": tf,
                "breakdown": breakdown,
                "extras": {
                    "vol_z": None if isinstance(vol_z, bool) else float(vol_z),
                    "now_tr": None if now_tr is None or np.isnan(now_tr) else float(now_tr),
                    "mean_tr": None if mean_tr is None or np.isnan(mean_tr) else float(mean_tr),
                }
            })
            # attach df if best for chart
            best["df"] = df.copy()
            best["ema_f"] = ema_f
            best["ema_s"] = ema_s

    return best

# ========= Text & Chart =========
def fmt_pct(x, digits=2):
    try:
        return f"{x*100:.{digits}f}%"
    except:
        return "n/a"

def generate_signal_text(symbol, res):
    trend_s, mom_s, rsi_s, cross_s = res["breakdown"] if res["breakdown"] else (0,0,0,0)
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    extra = res.get("extras", {})
    volz  = extra.get("vol_z", None)
    tr_now= extra.get("now_tr", None); tr_mean = extra.get("mean_tr", None)

    parts = [
        "📊 Վերջին ազդանշան՝",
        "🧠 <b>Pocket Opinion</b>",
        f"📉 Զույգը: <b>{symbol}</b>",
        f"📈 Ուղղություն: <b>{'🔼 Գնել 🟢' if res['dir']=='Buy' else '🔽 Վաճառել 🔴'}</b>",
        f"⏰ Ժամանակը: <b>{now_utc}</b>",
        f"💰 Գին: <b>{res['price']}</b>",
        f"📌 Տայմֆրեյմ: <b>{res['tf']}</b>",
        f"📊 Վստահություն: <b>{res['score']}%</b>",
        "— — —",
        f"🧩 Բաշխում → Trend: {int(trend_s*100)} | Momentum: {int(mom_s*100)} | RSI: {int(rsi_s*100)} | Cross: {int(cross_s*100)}",
        f"📦 Volume z-score ≥ {CFG['vol_z']} → <b>{'OK' if (volz is None or volz>=CFG['vol_z']) else 'NO'}</b>" + (f" (z={volz:.2f})" if volz is not None else ""),
        f"🌪 Volatility TR_now ≥ {CFG['tr_mult']}×mean(TR) → <b>{'OK' if (tr_now is None or tr_mean is None or tr_now >= tr_mean*CFG['tr_mult']) else 'NO'}</b>",
    ]
    return "\n\n".join(parts)

def render_chart_and_get_bytes(df, ema_f, ema_s, symbol, price):
    try:
        fig = plt.figure(figsize=(8, 3.5), dpi=150)
        sub = df.tail(CFG["chart_bars"])
        plt.plot(sub['timestamp'], sub['close'], label="Close")
        plt.plot(sub['timestamp'], ema_f.tail(len(sub)), label=f"EMA{CFG['ema_fast']}")
        plt.plot(sub['timestamp'], ema_s.tail(len(sub)), label=f"EMA{CFG['ema_slow']}")
        plt.title(f"{symbol} — last {CFG['chart_bars']} bars")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        bio = io.BytesIO()
        plt.savefig(bio, format='png')
        plt.close(fig)
        bio.seek(0)
        return bio
    except Exception:
        return None

# ========= Send logic =========
def maybe_send_signal(symbol, res):
    """Apply cooldown/min-move rules and send"""
    key = (res["dir"], res["tf"])
    now = time.time()
    info = last_signal_info[symbol]

    # first pass never sends
    if not first_check_done[symbol]:
        last_signal_info[symbol] = {"key": key, "ts": now, "price": float(res["price"])}
        first_check_done[symbol] = True
        print(f"{symbol} ✅ Առաջին ստուգում — պահվեց, բայց չուղարկվեց։")
        return

    # same key? check cooldown + min price move
    if info["key"] == key:
        within_cd = (now - info["ts"] < CFG["cooldown_sec"])
        small_move = (info["price"] is not None) and (abs(res["price"] - info["price"]) / info["price"] < CFG["min_price_move_pct"])
        if within_cd and small_move:
            # skip
            return

    # update and send
    last_signal_info[symbol] = {"key": key, "ts": now, "price": float(res["price"])}
    text = generate_signal_text(symbol, res)
    bot.send_message(chat_id, text)

    # store to history
    hist = signal_history[symbol]
    hist.append({
        "ts": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        "dir": res["dir"], "tf": res["tf"], "price": res["price"], "score": res["score"]
    })
    if len(hist) > 5:
        del hist[0]

    if CFG["send_chart"] and "df" in res:
        img = render_chart_and_get_bytes(res["df"], res["ema_f"], res["ema_s"], symbol, res["price"])
        if img:
            bot.send_photo(chat_id, img)

# ========= Main checker =========
def check_loop():
    while True:
        try:
            for symbol in CFG["symbols"]:
                res = analyze_symbol(symbol)
                if not res["dir"] or res["score"] < CFG["conf_min"]:
                    print(f"{symbol} ❌ Վստահությունը ցածր է կամ սիգնալ չկա։")
                    continue
                maybe_send_signal(symbol, res)
            time.sleep(CFG["check_interval_sec"])
        except Exception as e:
            print(f"❌ Loop error: {e}\n{traceback.format_exc()}")
            time.sleep(5)

# ========= Commands =========
def _uptime():
    s = int(time.time() - start_time)
    h = s // 3600; m = (s % 3600)//60; sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

@bot.message_handler(commands=['ping'])
def cmd_ping(msg):
    bot.reply_to(msg, f"🏓 Pong! Uptime: <b>{_uptime()}</b>")

@bot.message_handler(commands=['status'])
def cmd_status(msg):
    lines = [
        "🧰 <b>Bot status</b>",
        f"Symbols: {', '.join(CFG['symbols'])}",
        f"Intervals: {', '.join(CFG['intervals'])}",
        f"conf_min={CFG['conf_min']}  persist_bars={CFG['persist_bars']}  agree>={CFG['agreement_needed']}",
        f"cooldown={CFG['cooldown_sec']}s  min_move={CFG['min_price_move_pct']*100:.2f}%",
        f"vol_z>={CFG['vol_z']} (win={CFG['vol_window']})  TR_mult={CFG['tr_mult']} (win={CFG['tr_window']})",
        f"send_chart={CFG['send_chart']} bars={CFG['chart_bars']}",
        f"Uptime: {_uptime()}",
        "",
        "<b>Last signals (per symbol):</b>"
    ]
    for s in CFG["symbols"]:
        info = last_signal_info.get(s, {})
        key = info.get("key")
        if key:
            direction, tf = key
            ts = info.get("ts", 0)
            tss = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts))
            price = info.get("price")
            lines.append(f"• {s}: {direction} {tf} @ {price} ({tss} UTC)")
        else:
            lines.append(f"• {s}: —")
        hist = signal_history.get(s, [])
        if hist:
            for h in hist[-5:]:
                lines.append(f"   ↳ {h['ts']} UTC {h['dir']} {h['tf']} @ {h['price']} [{h['score']}%]")
    bot.reply_to(msg, "\n".join(lines))

@bot.message_handler(commands=['set'])
def cmd_set(msg):
    """
    /set key=value [key=value ...]
    keys:
      symbols=BTCUSDT,ETHUSDT
      conf_min=80
      persist_bars=3
      agreement_needed=2
      cooldown_sec=900
      min_price_move_pct=0.004
      check_interval_sec=1200
      vol_z=1.0
      tr_mult=1.15
      vol_window=40
      tr_window=40
      send_chart=true/false
      chart_bars=150
    """
    txt = msg.text.replace("\n"," ").strip()
    parts = txt.split()
    changed = []
    for p in parts[1:]:
        if '=' not in p: continue
        k, v = p.split('=', 1)
        k = k.strip()
        v = v.strip()
        try:
            if k == "symbols":
                syms = [x.strip().upper() for x in v.split(',') if x.strip()]
                if syms:
                    CFG["symbols"] = syms
                    # re-init state for new symbols
                    for s in syms:
                        if s not in last_signal_info:
                            last_signal_info[s] = {"key": None, "ts": 0.0, "price": None}
                            first_check_done[s] = False
                            signal_history[s]   = []
                    changed.append((k, ",".join(syms)))
            elif k in {"conf_min","persist_bars","agreement_needed","cooldown_sec","check_interval_sec",
                       "vol_window","tr_window","chart_bars"}:
                CFG[k] = int(float(v))
                changed.append((k, str(CFG[k])))
            elif k in {"min_price_move_pct","vol_z","tr_mult"}:
                CFG[k] = float(v)
                changed.append((k, str(CFG[k])))
            elif k == "send_chart":
                CFG[k] = (v.lower() in {"1","true","yes","on"})
                changed.append((k, str(CFG[k])))
            else:
                continue
        except Exception as e:
            bot.reply_to(msg, f"⚠️ Չհաջողվեց կիրառել `{k}={v}` → {e}")
    if changed:
        pretty = " | ".join([f"{k}={val}" for k,val in changed])
        bot.reply_to(msg, f"✅ Կիրառվեց: {pretty}")
    else:
        bot.reply_to(msg, "ℹ️ Օգտագործում՝ /set key=value ... (տես `/set` doc-ը կոդում)")

@bot.message_handler(commands=['start','help'])
def welcome(msg):
    bot.reply_to(msg, "Բարև 👋\nՀասանելի հրամաններ՝\n"
                      "• /ping — ping/uptime\n"
                      "• /status — կոնֆիգ + վերջին սիգնալներ\n"
                      "• /set key=value [...] — փոխել կարգավորումներ\n"
                      "Օրինակ՝ /set conf_min=80 vol_z=1.2 tr_mult=1.15 symbols=BTCUSDT,ETHUSDT,SOLUSDT")

# ========= Worker thread =========
def start_checker_thread():
    t = threading.Thread(target=check_loop, daemon=True)
    t.start()

# ========= Robust polling =========
def start_polling_forever():
    # long-polling self-healing loop
    while True:
        try:
            bot.polling(non_stop=True, interval=1, timeout=25, long_polling_timeout=30)
        except Exception as e:
            print(f"⚠️ polling error: {e}. Restarting in 5s…")
            time.sleep(5)

# ========= Main =========
if __name__ == "__main__":
    start_checker_thread()
    start_polling_forever()