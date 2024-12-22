import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import requests
import hmac
import hashlib
from urllib.parse import urlencode
import traceback
import pandas_ta as ta
import os
import json
import math
import csv
import sys
import logging
from dotenv import load_dotenv
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

matplotlib.use("TkAgg")

script_path = Path(__file__).resolve()
script_dir = script_path.parent
env_path = script_dir / 've.env'
load_dotenv(dotenv_path=env_path)

logging.basicConfig(
    filename='bot.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def load_configuration():
    config = {
        'API_KEY': os.getenv('BINANCE_API_KEY'),
        'API_SECRET': os.getenv('BINANCE_API_SECRET'),
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
        'SYMBOL': os.getenv('SIMBOL'),
        'INTERVAL': '1m',
        'START_PERCENTAGE': float(os.getenv('START_PERCENTAGE', 1.0)),
        'INCREMENT_PERCENTAGE': float(os.getenv('INCREMENT_PERCENTAGE', 1.0)),
        'MAX_PERCENTAGE': float(os.getenv('MAX_PERCENTAGE', 5.0)),
        'STOP_LOSS_PERCENT': float(os.getenv('STOP_LOSS_PERCENT', 1.0)),
        'DCA_PRICE_DROP_START': float(os.getenv('DCA_PRICE_DROP_START', 1.0)),
        'DCA_PRICE_DROP_INCREMENT': float(os.getenv('DCA_PRICE_DROP_INCREMENT', 0.5)),
        'DCA_PRICE_DROP_MAX': float(os.getenv('DCA_PRICE_DROP_MAX', 5.0)),
        'BNB_SYMBOL': 'BNBUSDT',
        'BNB_MIN_USD': float(os.getenv('BNB_MIN_USD', 20.0)),
        'BNB_REPLENISH_USD': float(os.getenv('BNB_REPLENISH_USD', 100.0)),
        'BNB_COOLDOWN_SECONDS': int(os.getenv('BNB_COOLDOWN_SECONDS', 60)),
        'BNB_BUY_LIMIT_PER_DAY': int(os.getenv('BNB_BUY_LIMIT_PER_DAY', 10)),
        'BNB_CHECK_DELAY_SECONDS': 300,
        'DROP_PERCENT': float(os.getenv('DROP_PERCENT', 2)),
        'MIN_PROFIT_PERCENT' : float(os.getenv('MIN_PROFIT_PERCENT', 2)),
    }
    return config

config = load_configuration()

if not all([config['API_KEY'], config['API_SECRET'], config['TELEGRAM_BOT_TOKEN'], config['TELEGRAM_CHAT_ID'], config['SYMBOL']]):
    logging.error("Faltan variables de entorno. Asegúrate de configurarlas en ve.env.")
    sys.exit("Faltan variables de entorno.")

EMA_FAST_PERIOD = 3
EMA_MID_PERIOD = 7
EMA_SLOW_PERIOD = 18
BB_LENGTH = 20
BB_STDDEV = 2.0
VOL_INCREASE_PCT = 25.0
VOL_PERIODS = 5
VOL_STRENGTH_THRESHOLD = 150.0
RSI_LENGTH = 14
RSI_LENGTH_1M = 14
RSI_LENGTH_MULTI = 21
RSI_OVERSOLD_THRESHOLD = 25
rsiDivergenceCheck = True
MACD_FAST = 52
MACD_SLOW = 200
MACD_SIGNAL = 3

ENTRY_PRICE_FILE = 'entry_price.json'
TOTAL_PROFIT_FILE = 'total_profit.json'
TRADE_LOG_FILE = 'trade_log.csv'
BUY_PRICES_FILE = 'buy_prices.json'
BUY_AMOUNTS_FILE = 'buy_amounts.json'
DCA_STEP_FILE = 'dca_step.json'
INITIAL_USDT_FILE = 'initial_usdt.json'

total_profit_usdt = 0.0
last_operation_profit = 0.0
last_buy_time = None
just_sold = False
initial_usdt = None

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{config['TELEGRAM_BOT_TOKEN']}/sendMessage"
    payload = {'chat_id': config['TELEGRAM_CHAT_ID'],'text': message,'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code != 200:
            logging.error(f"Error Telegram: {response.text}")
    except Exception as e:
        logging.exception(f"Excepción Telegram: {e}")

def sign_request(params, secret=config['API_SECRET']):
    query_string = urlencode(params)
    signature = hmac.new(secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

def save_total_profit(total_profit):
    try:
        data = {'total_profit_usdt': total_profit}
        with open(TOTAL_PROFIT_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logging.exception(f"Error al guardar total_profit: {e}")

def load_total_profit():
    if os.path.exists(TOTAL_PROFIT_FILE):
        try:
            with open(TOTAL_PROFIT_FILE, 'r') as f:
                data = json.load(f)
                return data.get('total_profit_usdt', 0.0)
        except:
            return 0.0
    return 0.0

def save_buy_prices(buy_prices):
    try:
        with open(BUY_PRICES_FILE, 'w') as f:
            json.dump(buy_prices, f)
    except:
        pass

def load_buy_prices():
    if os.path.exists(BUY_PRICES_FILE):
        try:
            with open(BUY_PRICES_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_buy_amounts(buy_amounts):
    try:
        with open(BUY_AMOUNTS_FILE, 'w') as f:
            json.dump(buy_amounts, f)
    except:
        pass

def load_buy_amounts():
    if os.path.exists(BUY_AMOUNTS_FILE):
        try:
            with open(BUY_AMOUNTS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_dca_step(dca_step):
    try:
        data = {'dca_step': dca_step}
        with open(DCA_STEP_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass

def load_dca_step():
    if os.path.exists(DCA_STEP_FILE):
        try:
            with open(DCA_STEP_FILE, 'r') as f:
                data = json.load(f)
                return data.get('dca_step', 0)
        except:
            return 0
    return 0

def save_initial_usdt(usdt_balance):
    try:
        data = {'initial_usdt': usdt_balance}
        with open(INITIAL_USDT_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass

def load_initial_usdt():
    if os.path.exists(INITIAL_USDT_FILE):
        try:
            with open(INITIAL_USDT_FILE, 'r') as f:
                data = json.load(f)
                return data.get('initial_usdt', None)
        except:
            return None
    return None

def log_trade(timestamp, side, quantity, price, profit_usdt, total_profit_usdt, symbol=config['SYMBOL'], bot=None):
    try:
        file_exists = os.path.isfile(TRADE_LOG_FILE)
        with open(TRADE_LOG_FILE, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'symbol', 'side', 'quantity', 'price', 'profit_usdt', 'total_profit_usdt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': timestamp,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'profit_usdt': profit_usdt,
                'total_profit_usdt': total_profit_usdt
            })
        if bot:
            log_message = f"{side} {quantity} {symbol} a {price} USDT. Ganancia: {profit_usdt} USDT"
            bot.send_log(log_message)
            bot.send_trade_data(side, quantity, price)
    except:
        pass

def get_historical_data(symbol, interval, limit):
    klines = []
    end_time = None
    attempts = 0
    max_attempts = 5  # Intentaremos hasta 5 veces
    while limit > 0 and attempts < max_attempts:
        fetch_limit = min(limit, 1000)
        params = {'symbol': symbol,'interval': interval,'limit': fetch_limit}
        if end_time:
            params['endTime'] = end_time
        try:
            response = requests.get('https://api.binance.com/api/v3/klines', params=params, timeout=10)
            data = response.json()
            if isinstance(data, dict) and 'code' in data:
                attempts += 1
                time.sleep(2)
                continue
            if not data:
                attempts += 1
                time.sleep(2)
                continue
            klines = data + klines
            end_time = data[0][0] - 1
            limit -= fetch_limit
            # Si se obtuvieron datos reiniciamos el contador de reintentos
            attempts = 0
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Error en get_historical_data: {e}")
            attempts += 1
            time.sleep(2)

    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_indicators(df):
    df['ema_fast'] = ta.ema(df['open'], length=EMA_FAST_PERIOD)
    df['ema_mid'] = ta.ema(df['open'], length=EMA_MID_PERIOD)
    df['ema_slow'] = ta.ema(df['open'], length=EMA_SLOW_PERIOD)

    bb = ta.bbands(df['close'], length=BB_LENGTH, std=BB_STDDEV)
    df['bb_upper'] = bb[f'BBU_{BB_LENGTH}_{BB_STDDEV}']
    df['bb_middle'] = bb[f'BBM_{BB_LENGTH}_{BB_STDDEV}']
    df['bb_lower'] = bb[f'BBL_{BB_LENGTH}_{BB_STDDEV}']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
    df['bb_expanding'] = df['bb_width'] > df['bb_width'].shift(1)

    macd = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df['macd_line'] = macd[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    df['signal_line'] = macd[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]

    df['rsi'] = ta.rsi(df['close'], length=RSI_LENGTH)
    df['rsi_1m'] = ta.rsi(df['close'], length=RSI_LENGTH_1M)
    df['rsi_multi'] = ta.rsi(df['close'], length=RSI_LENGTH_MULTI)

    df['vol_sma'] = df['volume'].rolling(VOL_PERIODS).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']
    df['is_volume_trend_up'] = df['volume'] > df['volume'].shift(1) * (1 + VOL_INCREASE_PCT / 100)

    df['max_volume'] = df['volume'].cummax()
    df['vol_strength'] = (df['volume'] / df['max_volume']) * 100
    df['is_high_vol_strength'] = df['vol_strength'] >= VOL_STRENGTH_THRESHOLD

    df['highest_price'] = df['high'].cummax()
    df['price_drop_percent'] = ((df['highest_price'] - df['close']) / df['highest_price']) * 100
    df['is_price_drop_signal'] = df['price_drop_percent'] >= config['DROP_PERCENT']

    return df

def isStopLossTriggered(current_price, stopLossPrice, in_position):
    if in_position and stopLossPrice is not None:
        return current_price <= stopLossPrice
    return False

def bullish_divergence(df):
    if len(df) < 4:
        return False
    rsi_1m_1 = df['rsi_1m'].iloc[-2]
    rsi_1m_2 = df['rsi_1m'].iloc[-3]
    rsi_1m_3 = df['rsi_1m'].iloc[-4]
    close_0 = df['close'].iloc[-1]
    close_1 = df['close'].iloc[-2]
    close_2 = df['close'].iloc[-3]

    bearishRsiHigher = (rsi_1m_1 > rsi_1m_2) and (rsi_1m_2 < rsi_1m_3)
    priceRsiConfirm = (close_0 < close_2) and (close_1 < close_2)

    rsi_multi_val = df['rsi_multi'].iloc[-1]
    return (rsi_multi_val < 50) and bearishRsiHigher and priceRsiConfirm

def calculate_weighted_average_price(buy_prices, buy_amounts):
    total_cost = sum(p * a for p, a in zip(buy_prices, buy_amounts))
    total_amount = sum(buy_amounts)
    if total_amount == 0:
        return 0
    return total_cost / total_amount

def check_signals(df, buy_prices, buy_amounts, in_position):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    is_bullish = (
        latest['ema_fast'] > latest['ema_slow'] and
        latest['macd_line'] > latest['signal_line'] and
        latest['close'] > latest['bb_middle']
    )
    is_bearish = (
        latest['ema_fast'] < latest['ema_slow'] and
        latest['macd_line'] < latest['signal_line'] and
        latest['close'] < latest['bb_middle']
    )

    is_weakening = latest['rsi'] < previous['rsi'] and latest['vol_ratio'] < 1
    b_divergence = bullish_divergence(df) if rsiDivergenceCheck else False

    if buy_prices and buy_amounts:
        avg_buy_price = calculate_weighted_average_price(buy_prices, buy_amounts)
    else:
        avg_buy_price = None

    dca_step = len(buy_prices)
    required_drop = config['DCA_PRICE_DROP_START'] + (dca_step * config['INCREMENT_PERCENTAGE'])
    required_drop = min(required_drop, config['DCA_PRICE_DROP_MAX'])

    if just_sold:
        long_signal = (is_bullish and latest['macd_line'] > latest['signal_line'] and latest['rsi'] < 65)
    elif not in_position:
        long_signal = (is_bullish and latest['macd_line'] > latest['signal_line'] and latest['rsi'] < 65)
    else:
        if avg_buy_price is not None and avg_buy_price > 0:
            price_drop_percent = ((avg_buy_price - latest['close']) / avg_buy_price) * 100
            long_signal = (
                is_bearish and 
                latest['close'] < avg_buy_price and
                price_drop_percent >= required_drop
            )
        else:
            long_signal = False

    if avg_buy_price and sum(buy_amounts) > 0:
        current_profit = ((latest['close'] - avg_buy_price) / avg_buy_price) * 100
    else:
        current_profit = 0

    has_min_profit = current_profit >= config['MIN_PROFIT_PERCENT']
    stopLossPrice = None
    if avg_buy_price and sum(buy_amounts) > 0:
        stopLossPrice = avg_buy_price * (1 - config['STOP_LOSS_PERCENT'] / 100.0)

    stopLossTriggered = isStopLossTriggered(latest['close'], stopLossPrice, in_position)
    short_signal = False
    if in_position:
        if has_min_profit and ((is_bearish and is_weakening) or latest['is_price_drop_signal'] or stopLossTriggered or (latest['macd_line'] < latest['signal_line'])):
            short_signal = True

    return long_signal, short_signal, current_profit, is_bullish, stopLossPrice, required_drop, avg_buy_price

def get_position():
    url = 'https://api.binance.com/api/v3/account'
    params = {'timestamp': int(time.time() * 1000)}
    signature = sign_request(params)
    params['signature'] = signature
    headers = {'X-MBX-APIKEY': config['API_KEY']}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        account_info = response.json()
        if 'balances' not in account_info:
            return 0
        asset = config['SYMBOL'].replace('USDT', '')
        for balance in account_info['balances']:
            if balance['asset'] == asset:
                return float(balance['free']) + float(balance['locked'])
    except:
        pass
    return 0

def get_usdt_balance():
    url = 'https://api.binance.com/api/v3/account'
    params = {'timestamp': int(time.time() * 1000)}
    signature = sign_request(params)
    params['signature'] = signature
    headers = {'X-MBX-APIKEY': config['API_KEY']}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        account_info = response.json()
        if 'balances' not in account_info:
            return 0
        for bal in account_info['balances']:
            if bal['asset'] == 'USDT':
                return float(bal['free'])
    except:
        pass
    return 0

def get_bnb_balance():
    url = 'https://api.binance.com/api/v3/account'
    params = {'timestamp': int(time.time() * 1000)}
    signature = sign_request(params)
    params['signature'] = signature
    headers = {'X-MBX-APIKEY': config['API_KEY']}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        account_info = response.json()
        if 'balances' not in account_info:
            return 0.0
        for bal in account_info['balances']:
            if bal['asset'] == 'BNB':
                return float(bal['free']) + float(bal['locked'])
    except:
        pass
    return 0.0

def get_bnb_price():
    try:
        response = requests.get('https://api.binance.com/api/v3/ticker/price', params={'symbol': config['BNB_SYMBOL']}, timeout=10)
        data = response.json()
        return float(data['price'])
    except:
        return 0.0

def get_symbol_info(symbol):
    url = 'https://api.binance.com/api/v3/exchangeInfo'
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        for s in data['symbols']:
            if s['symbol'] == symbol:
                return s
        return None
    except:
        return None

def adjust_quantity(symbol_info, quantity):
    try:
        step_size = 0.0
        min_qty = 0.0
        max_qty = 0.0
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])
                min_qty = float(f['minQty'])
                max_qty = float(f['maxQty'])
                break
        if step_size == 0:
            precision = 8
        else:
            precision = int(round(-math.log(step_size, 10), 0))
        quantity = max(min_qty, min(max_qty, quantity))
        quantity = float(round(quantity - (quantity % step_size), precision))
        return quantity
    except:
        return 0.0

def place_order(side, quantity, symbol=config['SYMBOL']):
    url = 'https://api.binance.com/api/v3/order'
    quantity_str = "{:0.{}f}".format(quantity, 8).rstrip('0').rstrip('.')
    data = {
        'symbol': symbol,
        'side': side,
        'type': 'MARKET',
        'quantity': quantity_str,
        'timestamp': int(time.time() * 1000)
    }
    signature = sign_request(data)
    data['signature'] = signature
    headers = {'X-MBX-APIKEY': config['API_KEY']}
    try:
        response = requests.post(url, params=data, headers=headers, timeout=10)
        return response.json()
    except:
        return {}

def buy_bnb(amount_usd=100.0, bot=None):
    bnb_symbol = config['BNB_SYMBOL']
    usdt_balance = get_usdt_balance()
    if usdt_balance < amount_usd:
        if bot:
            bot.send_log(f"No hay suficiente USDT ({usdt_balance}) para comprar BNB.")
        return False

    bnb_price = get_bnb_price()
    if bnb_price == 0.0:
        if bot:
            bot.send_log("No se pudo obtener el precio de BNB.")
        return False

    quantity = amount_usd / bnb_price
    symbol_info = get_symbol_info(bnb_symbol)
    if not symbol_info:
        if bot:
            bot.send_log(f"No se pudo obtener información del símbolo {bnb_symbol}.")
        return False

    quantity = adjust_quantity(symbol_info, quantity)
    result = place_order('BUY', quantity, symbol=bnb_symbol)
    if 'orderId' in result:
        fills = result.get('fills', [])
        if fills:
            total_qty = sum(float(fill['qty']) for fill in fills)
            total_price = sum(float(fill['price']) * float(fill['qty']) for fill in fills)
            avg_price = total_price / total_qty
            total_fee = sum(float(fill['commission']) for fill in fills)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_trade(timestamp, 'BUY_BNB', total_qty, avg_price, 0.0, total_profit_usdt, symbol=bnb_symbol, bot=bot)
            if bot:
                bot.send_log(f"Compra BNB: {total_qty:.6f} a {avg_price:.5f} USDT. Fee: {total_fee:.6f} BNB.")
                message = (
                    f"🚀 *MASTER TRADING BOT - Compra BNB*\n"
                    f"**Símbolo:** {bnb_symbol}\n"
                    f"**Cantidad:** {total_qty:.6f} BNB\n"
                    f"**Precio Promedio:** {avg_price:.5f} USDT\n"
                    f"**Fee Total:** {total_fee:.6f} BNB\n"
                    f"**Balance BNB:** {get_bnb_balance():.6f} BNB"
                )
                send_telegram_message(message)
            return True
    if bot:
        bot.send_log(f"Error al comprar BNB: {result}")
    return False

class TradingBot:
    def __init__(self, gui_queue):
        self.gui_queue = gui_queue
        self.running = False
        self.thread = None
        self.avg_buy_price = None
        self.last_bnb_buy_time = None
        self.bnb_cooldown = config['BNB_COOLDOWN_SECONDS']
        self.bnb_buy_count = 0
        self.bnb_buy_limit = config['BNB_BUY_LIMIT_PER_DAY']
        self.bnb_buy_reset_time = datetime.now() + timedelta(days=1)
        self.bnb_under_min_detect_time = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_bot, daemon=True)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join()

    def send_gui_update(self, **kwargs):
        self.gui_queue.put(kwargs)

    def send_log(self, message):
        self.gui_queue.put({'log': message})

    def send_trade_data(self, side, quantity, price):
        trade_data = {
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now()
        }
        logging.info(f"Trade data: {trade_data}")
        self.gui_queue.put({'trade': trade_data})

    def run_bot(self):
        global total_profit_usdt, last_operation_profit, last_buy_time, just_sold, initial_usdt
        total_profit_usdt = load_total_profit()
        buy_prices = load_buy_prices()
        buy_amounts = load_buy_amounts()
        dca_step = load_dca_step()
        initial_usdt = load_initial_usdt()

        if initial_usdt is None:
            initial_usdt = get_usdt_balance()
            save_initial_usdt(initial_usdt)
            self.send_log(f"Balance inicial de USDT: {initial_usdt:.2f}")

        in_position = (len(buy_prices) > 0)
        last_operation_profit = 0.0
        stopLossPrice = None

        no_data_attempts = 0  # Contador de intentos fallidos de obtener datos

        while self.running:
            try:
                max_period = max(EMA_SLOW_PERIOD, BB_LENGTH, RSI_LENGTH_MULTI, MACD_SLOW) * 5

                df = get_historical_data(config['SYMBOL'], config['INTERVAL'], max_period)
                if df.empty:
                    no_data_attempts += 1
                    self.send_log(f"No se obtuvieron datos. Intento {no_data_attempts} de 10.")
                    if no_data_attempts >= 10:
                        self.send_log("No se pudieron obtener datos tras 10 intentos. Esperando 10 segundos antes de reintentar.")
                        time.sleep(10)
                        no_data_attempts = 0
                    else:
                        time.sleep(1)
                    continue
                else:
                    no_data_attempts = 0

                df = calculate_indicators(df)
                df.dropna(inplace=True)
                if df.empty or len(df) < 4:
                    self.send_log("Datos insuficientes (mín. 4 velas). Reintentando...")
                    time.sleep(1)
                    continue

                current_price = df['close'].iloc[-1]
                symbol_info = get_symbol_info(config['SYMBOL'])
                if not symbol_info:
                    self.send_log(f"No se pudo obtener info del símbolo {config['SYMBOL']}. Reintento...")
                    time.sleep(1)
                    continue

                in_position = (len(buy_prices) > 0)
                position_amt = get_position()
                usdt_balance = get_usdt_balance()
                bnb_balance = get_bnb_balance()
                bnb_price = get_bnb_price()
                bnb_balance_usdt = bnb_balance * bnb_price

                long_signal, short_signal, current_profit, is_bullish, stopLossPrice, required_drop, avg_buy_price = check_signals(
                    df, buy_prices, buy_amounts, in_position
                )

                self.avg_buy_price = avg_buy_price
                self.send_gui_update(avg_buy_price=self.avg_buy_price)

                if just_sold and long_signal and not is_bullish:
                    long_signal = False

                if long_signal:
                    if not in_position:
                        current_percentage = config['START_PERCENTAGE']
                    else:
                        dca_step = len(buy_prices)
                        current_percentage = config['START_PERCENTAGE'] + (dca_step * config['INCREMENT_PERCENTAGE'])
                        if current_percentage > config['MAX_PERCENTAGE']:
                            current_percentage = config['MAX_PERCENTAGE']

                    self.send_log(f"DCA: {current_percentage:.2f}%")
                    position_size_usdt = usdt_balance * current_percentage / 100
                    position_size = position_size_usdt / current_price
                    position_size = adjust_quantity(symbol_info, position_size)
                    min_qty = float(next(f['minQty'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'))
                    if position_size >= min_qty:
                        result = place_order('BUY', position_size)
                        if 'orderId' in result:
                            fills = result.get('fills', [])
                            if fills:
                                total_qty = sum(float(fill['qty']) for fill in fills)
                                total_price = sum(float(fill['price']) * float(fill['qty']) for fill in fills)
                                avg_price = total_price / total_qty
                                total_fee = sum(float(fill['commission']) for fill in fills)
                                buy_prices.append(avg_price)
                                buy_amounts.append(total_qty)
                                save_buy_prices(buy_prices)
                                save_buy_amounts(buy_amounts)
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                log_trade(timestamp, 'BUY', total_qty, avg_price, 0.0, total_profit_usdt, symbol=config['SYMBOL'], bot=self)
                                self.send_log(f"Compra: {total_qty:.6f} {config['SYMBOL']} a {avg_price:.5f} USDT. Fee: {total_fee:.6f}")
                                in_position = True
                                last_buy_time = datetime.now()
                                stopLossPrice = avg_price * (1 - config['STOP_LOSS_PERCENT'] / 100)
                                just_sold = False
                                dca_step += 1
                                save_dca_step(dca_step)
                                message = (
                                    f"🚀 *MASTER TRADING BOT - Compra*\n"
                                    f"**Símbolo:** {config['SYMBOL']}\n"
                                    f"**Cantidad:** {total_qty:.6f}\n"
                                    f"**Precio:** {avg_price:.5f} USDT\n"
                                    f"**Fee Total:** {total_fee:.6f}\n"
                                    f"**Balance USDT:** {usdt_balance:.2f} USDT\n"
                                    f"**Ganancia Total:** {total_profit_usdt:.2f} USDT"
                                )
                                send_telegram_message(message)

                stopLossTriggered = isStopLossTriggered(current_price, stopLossPrice, in_position)
                if in_position and short_signal:
                    sell_amount = adjust_quantity(symbol_info, position_amt)
                    if sell_amount > 0:
                        result = place_order('SELL', sell_amount)
                        if 'orderId' in result:
                            sell_df = get_historical_data(config['SYMBOL'], config['INTERVAL'], 1)
                            if not sell_df.empty:
                                current_price_sell = sell_df['close'].iloc[-1]
                            else:
                                current_price_sell = current_price
                            total_sell = sell_amount * current_price_sell
                            total_cost = sum(p * a for p, a in zip(buy_prices, buy_amounts))
                            total_amount = sum(buy_amounts)
                            avg_buy_price = total_cost / total_amount if total_amount > 0 else 0
                            profit_usdt = total_sell - (avg_buy_price * sell_amount)
                            fills = result.get('fills', [])
                            total_fee = sum(float(fill['commission']) for fill in fills)
                            commission_pct = 0.2
                            profit_usdt = profit_usdt - (total_sell * commission_pct / 100)
                            profit_usdt += total_fee * avg_buy_price
                            total_profit_usdt += profit_usdt
                            save_total_profit(total_profit_usdt)
                            last_operation_profit = profit_usdt
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            log_trade(timestamp, 'SELL_ALL', sell_amount, current_price_sell, profit_usdt, total_profit_usdt, symbol=config['SYMBOL'], bot=self)
                            self.send_log(f"Venta: {sell_amount:.6f} a {current_price_sell:.5f} USDT. +{profit_usdt:.2f} USDT")
                            buy_prices = []
                            buy_amounts = []
                            save_buy_prices(buy_prices)
                            save_buy_amounts(buy_amounts)
                            in_position = False
                            stopLossPrice = None
                            just_sold = True
                            dca_step = 0
                            save_dca_step(dca_step)
                            self.avg_buy_price = None
                            self.send_gui_update(avg_buy_price=self.avg_buy_price)
                            message = (
                                f"📉 *MASTER TRADING BOT - Venta*\n"
                                f"**Símbolo:** {config['SYMBOL']}\n"
                                f"**Cantidad Vendida:** {sell_amount:.6f}\n"
                                f"**Precio Venta:** {current_price_sell:.5f} USDT\n"
                                f"**Ganancia:** {profit_usdt:.2f} USDT\n"
                                f"**Fee:** {total_fee:.6f}\n"
                                f"**Ganancia Total:** {total_profit_usdt:.2f} USDT\n"
                                f"**Balance USDT:** {usdt_balance:.2f} USDT"
                            )
                            send_telegram_message(message)

                current_time = datetime.now()
                can_buy_bnb = False
                if self.last_bnb_buy_time is None:
                    can_buy_bnb = True
                else:
                    elapsed_time = (current_time - self.last_bnb_buy_time).total_seconds()
                    if elapsed_time >= self.bnb_cooldown:
                        can_buy_bnb = True

                if current_time >= self.bnb_buy_reset_time:
                    self.bnb_buy_count = 0
                    self.bnb_buy_reset_time = current_time + timedelta(days=1)

                if bnb_balance_usdt < config['BNB_MIN_USD']:
                    if self.bnb_under_min_detect_time is None:
                        self.bnb_under_min_detect_time = current_time
                        self.send_log("BNB bajo el mínimo. Esperando 5 minutos.")
                    else:
                        time_since_detection = (current_time - self.bnb_under_min_detect_time).total_seconds()
                        if time_since_detection >= config['BNB_CHECK_DELAY_SECONDS']:
                            if can_buy_bnb and self.bnb_buy_count < self.bnb_buy_limit:
                                self.send_log("BNB bajo mínimo por >5 min. Comprando BNB.")
                                buy_success = buy_bnb(config['BNB_REPLENISH_USD'], bot=self)
                                if buy_success:
                                    self.last_bnb_buy_time = current_time
                                    self.bnb_buy_count += 1
                else:
                    if self.bnb_under_min_detect_time is not None:
                        self.send_log("BNB recuperó el mínimo. Reseteando temporizador.")
                    self.bnb_under_min_detect_time = None

                dca_percentage = config['START_PERCENTAGE'] + (dca_step * config['INCREMENT_PERCENTAGE'])
                if dca_percentage > config['MAX_PERCENTAGE']:
                    dca_percentage = config['MAX_PERCENTAGE']

                self.send_gui_update(
                    current_price=current_price,
                    usdt_balance=usdt_balance,
                    bnb_balance=bnb_balance,
                    bnb_balance_usdt=bnb_balance_usdt,
                    total_profit_usdt=total_profit_usdt,
                    last_operation_profit=last_operation_profit,
                    long_signal=long_signal,
                    short_signal=short_signal,
                    dca_percentage=dca_percentage if in_position else config['START_PERCENTAGE'],
                    initial_usdt=initial_usdt
                )
                time.sleep(1)
            except Exception as e:
                self.send_log(f"Error en el bot: {e}")
                time.sleep(1)

class BotGUI:
    def __init__(self, master, bot):
        self.master = master
        self.bot = bot
        self.queue = queue.Queue()
        self.bot.gui_queue = self.queue

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('Helvetica', 9))
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 9))
        style.configure('TButton', font=('Helvetica', 9, 'bold'), background='#e0e0e0', foreground='#000')

        master.title("MASTER TRADING BOT by Neurodoc")

        # Establecer el icono
        try:
            master.iconbitmap("icono.ico")
        except:
            pass

        menubar = tk.Menu(master)
        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="Acerca de", command=self.show_about_tab)
        about_menu.add_command(label="Configuración", command=self.show_config_tab)
        menubar.add_cascade(label="Opciones", menu=about_menu)
        master.config(menu=menubar)

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.status_frame = ttk.Frame(self.notebook, padding=5)
        self.trades_frame = ttk.Frame(self.notebook, padding=5)
        self.config_frame = ttk.Frame(self.notebook, padding=5)
        self.about_frame = ttk.Frame(self.notebook, padding=5)

        self.notebook.add(self.status_frame, text='Estado del Bot')
        self.notebook.add(self.trades_frame, text='Gráficos')
        self.notebook.add(self.config_frame, text='Config')
        self.notebook.add(self.about_frame, text='Acerca de')

        self.create_status_tab()
        self.create_trades_tab()
        self.create_config_tab()
        self.create_about_tab()

        self.bottom_frame = ttk.Frame(master, padding=5)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.start_button = ttk.Button(self.bottom_frame, text="Iniciar Bot", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = ttk.Button(self.bottom_frame, text="Detener Bot", command=self.stop_bot, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.trade_times = []
        self.trade_prices = []
        self.trade_types = []
        self.price_data = pd.DataFrame(columns=['timestamp', 'price'])

        self.env_text = scrolledtext.ScrolledText(self.config_frame, width=80, height=20)
        self.env_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.load_env_content()

        self.save_env_button = ttk.Button(self.config_frame, text="Guardar", command=self.save_env)
        self.save_env_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.reload_env_button = ttk.Button(self.config_frame, text="Recargar", command=self.reload_env)
        self.reload_env_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.update_gui()

    def show_about_tab(self):
        self.notebook.select(self.about_frame)

    def show_config_tab(self):
        self.notebook.select(self.config_frame)

    def create_status_tab(self):
        container = ttk.Frame(self.status_frame)
        container.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(container)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        right_frame = ttk.Frame(container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        header_label = ttk.Label(left_frame, text="Estado del Bot", font=('Helvetica', 12, 'bold'))
        header_label.grid(row=0, column=0, columnspan=2, pady=5, sticky='w')

        ttk.Label(left_frame, text="Símbolo:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.symbol_label = ttk.Label(left_frame, text=config['SYMBOL'])
        self.symbol_label.grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Intervalo:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.interval_label = ttk.Label(left_frame, text=config['INTERVAL'])
        self.interval_label.grid(row=2, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Precio Actual:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
        self.price_var = tk.StringVar(value="---")
        self.price_label = ttk.Label(left_frame, textvariable=self.price_var)
        self.price_label.grid(row=3, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Balance USDT:").grid(row=4, column=0, sticky=tk.W, padx=2, pady=2)
        self.usdt_balance_var = tk.StringVar(value="---")
        self.usdt_balance_label = ttk.Label(left_frame, textvariable=self.usdt_balance_var)
        self.usdt_balance_label.grid(row=4, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Balance BNB:").grid(row=5, column=0, sticky=tk.W, padx=2, pady=2)
        self.bnb_balance_var = tk.StringVar(value="---")
        self.bnb_balance_label = ttk.Label(left_frame, textvariable=self.bnb_balance_var)
        self.bnb_balance_label.grid(row=5, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Balance USDT Inicial:").grid(row=6, column=0, sticky=tk.W, padx=2, pady=2)
        self.initial_usdt_var = tk.StringVar(value="---")
        self.initial_usdt_label = ttk.Label(left_frame, textvariable=self.initial_usdt_var)
        self.initial_usdt_label.grid(row=6, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Ganancia Total:").grid(row=7, column=0, sticky=tk.W, padx=2, pady=2)
        self.total_profit_var = tk.StringVar(value="---")
        self.total_profit_label = ttk.Label(left_frame, textvariable=self.total_profit_var)
        self.total_profit_label.grid(row=7, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Ganancia Últ. Oper.:").grid(row=8, column=0, sticky=tk.W, padx=2, pady=2)
        self.last_profit_var = tk.StringVar(value="---")
        self.last_profit_label = ttk.Label(left_frame, textvariable=self.last_profit_var)
        self.last_profit_label.grid(row=8, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Señal de Compra:").grid(row=9, column=0, sticky=tk.W, padx=2, pady=2)
        self.buy_signal_var = tk.StringVar(value="❌")
        self.buy_signal_label = ttk.Label(left_frame, textvariable=self.buy_signal_var, foreground="red")
        self.buy_signal_label.grid(row=9, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Señal de Venta:").grid(row=10, column=0, sticky=tk.W, padx=2, pady=2)
        self.sell_signal_var = tk.StringVar(value="❌")
        self.sell_signal_label = ttk.Label(left_frame, textvariable=self.sell_signal_var, foreground="red")
        self.sell_signal_label.grid(row=10, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Próxima Compra (DCA)%:").grid(row=11, column=0, sticky=tk.W, padx=2, pady=2)
        self.dca_percentage_var = tk.StringVar(value="---")
        self.dca_percentage_label = ttk.Label(left_frame, textvariable=self.dca_percentage_var)
        self.dca_percentage_label.grid(row=11, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(left_frame, text="Precio Prom. Compra:").grid(row=12, column=0, sticky=tk.W, padx=2, pady=2)
        self.avg_buy_price_var = tk.StringVar(value="---")
        self.avg_buy_price_label = ttk.Label(left_frame, textvariable=self.avg_buy_price_var)
        self.avg_buy_price_label.grid(row=12, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(right_frame, text="Log de Operaciones:", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=2)
        self.log_text = scrolledtext.ScrolledText(right_frame, width=80, height=25, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    def create_trades_tab(self):
        header_label = ttk.Label(self.trades_frame, text="Gráficos de Operaciones", font=('Helvetica', 12, 'bold'))
        header_label.pack(pady=5)

        self.figure, self.ax = plt.subplots(figsize=(6,4))
        self.ax.set_title(f"Operaciones de {config['SYMBOL']}", fontweight='bold')
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Precio (USDT)")
        self.trade_plot = None

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.trades_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_config_tab(self):
        header_label = ttk.Label(self.config_frame, text="Editar Configuración (ve.env)", font=('Helvetica', 12, 'bold'))
        header_label.pack(pady=5)

    def load_env_content(self):
        try:
            with open(env_path, 'r') as f:
                content = f.read()
                self.env_text.delete(1.0, tk.END)
                self.env_text.insert(tk.END, content)
        except Exception as e:
            self.env_text.delete(1.0, tk.END)
            self.env_text.insert(tk.END, f"Error al cargar ve.env: {e}")

    def save_env(self):
        try:
            content = self.env_text.get(1.0, tk.END)
            with open(env_path, 'w') as f:
                f.write(content)
            messagebox.showinfo("Éxito", "Configuración guardada exitosamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar ve.env: {e}")

    def reload_env(self):
        try:
            response = False
            if self.bot.running:
                response = messagebox.askyesno(
                    "Recargar Configuración",
                    "El bot está corriendo. ¿Detenerlo para recargar configuración?"
                )
                if response:
                    self.stop_bot()
                else:
                    return

            load_dotenv(dotenv_path=env_path, override=True)
            global config
            config = load_configuration()
            messagebox.showinfo("Éxito", "Configuración recargada exitosamente.")
            self.symbol_label.config(text=config['SYMBOL'])
            self.interval_label.config(text=config['INTERVAL'])

            if response:
                self.start_bot()

        except Exception as e:
            messagebox.showerror("Error", f"Error al recargar configuración: {e}")

    def create_about_tab(self):
        header_label = ttk.Label(self.about_frame, text="Acerca de", font=('Helvetica', 12, 'bold'))
        header_label.pack(pady=5)
        about_text = (
            "Este Bot de Trading con DCA fue creado por:\n"
            "Jesús Nicolás Astorga\n"
            "Programador y Trader\n"
            "Junín, Mendoza, Argentina\n\n"
            "Correo: mza.nicolas.astorga@gmail.com\n\n"
            "Comparte el bot si te va bien.\n\n"
            "Donaciones USDT (TRC20): TP2DxAk2idhxzrb7mKMjt6gTdBeDjjt1zA\n"
        )
        label_about = ttk.Label(self.about_frame, text=about_text, justify=tk.LEFT)
        label_about.pack(padx=5, pady=5, anchor='w')

        copy_button = ttk.Button(self.about_frame, text="Copiar Wallet", command=self.copy_wallet)
        copy_button.pack(padx=5, pady=5, anchor='w')

    def copy_wallet(self):
        wallet = "TP2DxAk2idhxzrb7mKMjt6gTdBeDjjt1zA"
        try:
            self.master.clipboard_clear()
            self.master.clipboard_append(wallet)
            messagebox.showinfo("Éxito", "Wallet copiada al portapapeles.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo copiar la wallet: {e}")

    def start_bot(self):
        self.bot.start()
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.log("Bot iniciado.")

    def stop_bot(self):
        if self.bot.running and (len(load_buy_prices()) > 0):
            response = messagebox.askyesno(
                "Confirmación de Venta",
                "Hay una posición abierta. ¿Vender y convertir a USDT?\n\n"
                "Sí: Vender y convertir.\nNo: Conservar."
            )
            if response:
                self.bot.stop()
                self.log("Bot detenido. Iniciando venta final.")
                self.sell_all_to_usdt()
            else:
                self.bot.stop()
                self.log("Bot detenido. Conservando posición.")
        else:
            self.bot.stop()
            self.log("Bot detenido.")
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def sell_all_to_usdt(self):
        position_amt = get_position()
        if position_amt > 0:
            symbol_info = get_symbol_info(config['SYMBOL'])
            if not symbol_info:
                self.send_log(f"No se pudo obtener info de {config['SYMBOL']}.")
                return
            sell_amount = adjust_quantity(symbol_info, position_amt)
            if sell_amount > 0:
                result = place_order('SELL', sell_amount)
                if 'orderId' in result:
                    sell_df = get_historical_data(config['SYMBOL'], config['INTERVAL'], 1)
                    if not sell_df.empty:
                        current_price_sell = sell_df['close'].iloc[-1]
                    else:
                        current_price_sell = position_amt
                    total_sell = sell_amount * current_price_sell
                    buy_prices = load_buy_prices()
                    buy_amounts = load_buy_amounts()
                    total_cost = sum(p * a for p, a in zip(buy_prices, buy_amounts))
                    total_amount = sum(buy_amounts)
                    avg_buy_price = total_cost / total_amount if total_amount > 0 else 0
                    profit_usdt = total_sell - (avg_buy_price * sell_amount)
                    fills = result.get('fills', [])
                    total_fee = sum(float(fill['commission']) for fill in fills)
                    commission_pct = 0.2
                    profit_usdt = profit_usdt - (total_sell * commission_pct / 100)
                    profit_usdt += total_fee * avg_buy_price
                    global total_profit_usdt, last_operation_profit
                    total_profit_usdt += profit_usdt
                    save_total_profit(total_profit_usdt)
                    last_operation_profit = profit_usdt
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    log_trade(timestamp, 'SELL_ALL', sell_amount, current_price_sell, profit_usdt, total_profit_usdt, symbol=config['SYMBOL'], bot=self)
                    self.send_log(f"Venta final: {sell_amount:.6f} a {current_price_sell:.5f} USDT. Ganancia: {profit_usdt:.2f} USDT.")

                    buy_prices = []
                    buy_amounts = []
                    save_buy_prices(buy_prices)
                    save_buy_amounts(buy_amounts)
                    dca_step = 0
                    save_dca_step(dca_step)
                    self.send_log("DCA Step Reiniciado.")
                    self.avg_buy_price = None
                    self.send_gui_update(avg_buy_price=self.avg_buy_price)

                    message = (
                        f"📉 *Venta Final*\n"
                        f"**Símbolo:** {config['SYMBOL']}\n"
                        f"**Cantidad:** {sell_amount:.6f}\n"
                        f"**Precio:** {current_price_sell:.5f} USDT\n"
                        f"**Ganancia:** {profit_usdt:.2f} USDT\n"
                        f"**Fee:** {total_fee:.6f}\n"
                        f"**Ganancia Total:** {total_profit_usdt:.2f} USDT\n"
                        f"**Balance USDT:** {get_usdt_balance():.2f}"
                    )
                    send_telegram_message(message)
            else:
                self.send_log("Cantidad mínima de venta no alcanzada.")
        else:
            self.send_log("No hay posiciones abiertas para vender.")

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text.config(state='disabled')
        self.log_text.yview(tk.END)

    def update_trades_plot(self):
        self.ax.clear()
        self.ax.set_title(f"Operaciones de {config['SYMBOL']}", fontweight='bold')
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Precio (USDT)")

        window_minutes = 60
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=window_minutes)
        mask = self.price_data['timestamp'] >= window_start
        visible_price_data = self.price_data[mask]

        if not visible_price_data.empty:
            self.ax.plot(visible_price_data['timestamp'], visible_price_data['price'], 
                         label='Precio', color='blue', linewidth=1)

        current_trades = [(t, p, typ) for t, p, typ in zip(self.trade_times, self.trade_prices, self.trade_types)
                          if t >= window_start]

        buys = [(t, p) for t, p, typ in current_trades if typ == 'BUY']
        sells = [(t, p) for t, p, typ in current_trades if typ == 'SELL']

        if buys:
            buy_times, buy_prices = zip(*buys)
            self.ax.scatter(buy_times, buy_prices, marker='^', color='green', 
                            label='Compra', s=80, zorder=5)

        if sells:
            sell_times, sell_prices = zip(*sells)
            self.ax.scatter(sell_times, sell_prices, marker='v', color='red', 
                            label='Venta', s=80, zorder=5)

        if self.bot.avg_buy_price is not None:
            self.ax.axhline(y=self.bot.avg_buy_price, color='orange', 
                            linestyle='--', label='Precio Promedio Compra', zorder=4)

        if not visible_price_data.empty:
            ymin = visible_price_data['price'].min()
            ymax = visible_price_data['price'].max()
            margin = (ymax - ymin) * 0.1
            self.ax.set_ylim(ymin - margin, ymax + margin)

        if buys or sells or self.bot.avg_buy_price is not None:
            self.ax.legend(loc='upper left')

        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.figure.autofmt_xdate()
        self.ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        self.figure.tight_layout()
        self.canvas.draw()

    def update_gui(self):
        while not self.queue.empty():
            msg = self.queue.get()
            if isinstance(msg, dict):
                if 'current_price' in msg:
                    self.price_var.set(f"{msg['current_price']:.5f} USDT")
                    timestamp = datetime.now()
                    new_row = pd.DataFrame({'timestamp': [timestamp], 'price': [msg['current_price']]})
                    self.price_data = pd.concat([self.price_data, new_row], ignore_index=True).tail(1000)
                if 'usdt_balance' in msg:
                    self.usdt_balance_var.set(f"{msg['usdt_balance']:.2f} USDT")
                if 'bnb_balance' in msg:
                    self.bnb_balance_var.set(f"{msg['bnb_balance']:.6f} BNB ({msg['bnb_balance_usdt']:.2f} USDT)")
                if 'total_profit_usdt' in msg:
                    self.total_profit_var.set(f"{msg['total_profit_usdt']:.2f} USDT")
                if 'last_operation_profit' in msg:
                    self.last_profit_var.set(f"{msg['last_operation_profit']:.2f} USDT")
                if 'long_signal' in msg:
                    self.buy_signal_var.set("✅" if msg['long_signal'] else "❌")
                    self.buy_signal_label.config(foreground="green" if msg['long_signal'] else "red")
                if 'short_signal' in msg:
                    self.sell_signal_var.set("✅" if msg['short_signal'] else "❌")
                    self.sell_signal_label.config(foreground="green" if msg['short_signal'] else "red")
                if 'dca_percentage' in msg:
                    self.dca_percentage_var.set(f"{msg['dca_percentage']:.2f}%")
                if 'initial_usdt' in msg:
                    self.initial_usdt_var.set(f"{msg['initial_usdt']:.2f} USDT")
                if 'avg_buy_price' in msg:
                    if msg['avg_buy_price'] is not None:
                        self.avg_buy_price_var.set(f"{msg['avg_buy_price']:.5f} USDT")
                    else:
                        self.avg_buy_price_var.set("---")
                if 'log' in msg:
                    self.log(msg['log'])
                if 'trade' in msg:
                    trade = msg['trade']
                    if trade['side'] == 'SELL_ALL':
                        self.trade_times = []
                        self.trade_prices = []
                        self.trade_types = []
                        self.log("Venta completa. Reiniciando gráfico.")
                    self.trade_times.append(trade['timestamp'])
                    self.trade_prices.append(trade['price'])
                    self.trade_types.append('SELL' if trade['side'] == 'SELL_ALL' else trade['side'])

        self.update_trades_plot()
        self.master.after(1000, self.update_gui)

def on_closing(root, bot):
    if messagebox.askokcancel("Salir", "¿Quieres salir?"):
        bot.stop()
        root.destroy()

def show_splash():
    splash = tk.Tk()
    splash.overrideredirect(True)
    # Centrar la ventana
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    width = 400
    height = 400
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    splash.geometry(f"{width}x{height}+{x}+{y}")

    try:
        img = Image.open("neurodoc.jpg")
        img = img.resize((400, 400), Image.Resampling.LANCZOS)
        splash_img = ImageTk.PhotoImage(img)
        label = tk.Label(splash, image=splash_img)
        label.pack()
    except:
        label = tk.Label(splash, text="Cargando...", font=('Helvetica', 16))
        label.pack(expand=True, fill="both")

    splash.update()
    time.sleep(3)  # Duración del splash
    splash.destroy()

def main():
    # Mostrar splash antes de la ventana principal
    show_splash()

    root = tk.Tk()
    gui_queue = queue.Queue()
    bot = TradingBot(gui_queue)
    gui = BotGUI(root, bot)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, bot))
    root.mainloop()

if __name__ == "__main__":
    main()
