import ccxt
import pandas as pd
import pytz
from pybit.unified_trading import HTTP
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys  # Added for reliable exit

# ===== Configuration =====
SYMBOLS = [
    'XRP/USDT:USDT',
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'SOL/USDT:USDT',
    'ADA/USDT:USDT'
]
TRADE_AMOUNT_USDT = 50          # Position size in USDT
STOPLOSS_PERCENT = 2            # 2% stop-loss
TAKEPROFIT_PERCENT = 7.5        # 7.5% take-profit

# Email Configuration
SENDER_EMAIL = "dahmadu071@gmail.com"
RECIPIENT_EMAILS = ["teejeedeeone@gmail.com"]
EMAIL_PASSWORD = "oase wivf hvqn lyhr"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Strategy Parameters
EMA_FAST = 38
EMA_SLOW = 62
EMA_TREND = 200
TIMEFRAME = '15m'

# API Configuration
API_KEY = "lJu52hbBTbPkg2VXZ2"
API_SECRET = "e43RV6YDZsn24Q9mucr0i4xbU7YytdL2HtuV"
DEMO_MODE = True  # Set to False for live trading

# ===== Initialize Connections =====
bybit = HTTP(
    api_key=API_KEY,
    api_secret=API_SECRET,
    demo=DEMO_MODE
)

bitget = ccxt.bitget({
    'enableRateLimit': True
})

# ===== Helper Functions =====
def send_email_notification(subject, body):
    """Send email notification about trade execution"""
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ", ".join(RECIPIENT_EMAILS)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
        
        print("Email notification sent successfully")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def get_lot_size_info(symbol):
    """Get the lot size rules for the symbol"""
    bybit_symbol = symbol.replace('/USDT:USDT', 'USDT')
    response = bybit.get_instruments_info(
        category="linear",
        symbol=bybit_symbol
    )
    if response['retCode'] == 0:
        return response['result']['list'][0]['lotSizeFilter']
    raise Exception(f"Failed to get lot size info: {response['retMsg']}")

def adjust_quantity(quantity, lot_size_info):
    """Adjust quantity to comply with exchange rules"""
    qty_step = float(lot_size_info['qtyStep'])
    min_qty = float(lot_size_info['minOrderQty'])
    max_qty = float(lot_size_info['maxOrderQty'])
    
    # Round to nearest step size
    adjusted_qty = round(quantity / qty_step) * qty_step
    # Ensure within min/max limits
    return max(min_qty, min(adjusted_qty, max_qty))

def get_current_price(symbol):
    """Get the current market price"""
    bybit_symbol = symbol.replace('/USDT:USDT', 'USDT')
    ticker = bybit.get_tickers(
        category="linear",
        symbol=bybit_symbol
    )
    if ticker['retCode'] == 0:
        return float(ticker['result']['list'][0]['lastPrice'])
    raise Exception(f"Failed to get price: {ticker['retMsg']}")

def place_trade_order(symbol, signal, price):
    """Place the trade order with stop-loss and take-profit"""
    bybit_symbol = symbol.replace('/USDT:USDT', 'USDT')
    lot_size_info = get_lot_size_info(symbol)
    
    # Calculate position size
    raw_qty = TRADE_AMOUNT_USDT / price
    quantity = adjust_quantity(raw_qty, lot_size_info)
    
    # Calculate SL and TP prices
    if signal == "buy":
        sl_price = round(price * (1 - STOPLOSS_PERCENT/100), 4)
        tp_price = round(price * (1 + TAKEPROFIT_PERCENT/100), 4)
        side = "Buy"
    else:  # sell/short
        sl_price = round(price * (1 + STOPLOSS_PERCENT/100), 4)
        tp_price = round(price * (1 - TAKEPROFIT_PERCENT/100), 4)
        side = "Sell"
    
    # Place the order
    order = bybit.place_order(
        category="linear",
        symbol=bybit_symbol,
        side=side,
        orderType="Market",
        qty=str(quantity),
        takeProfit=str(tp_price),
        stopLoss=str(sl_price),
        timeInForce="GTC"
    )
    
    if order['retCode'] == 0:
        trade_details = (
            f"Symbol: {symbol}\n"
            f"Direction: {signal.upper()}\n"
            f"Quantity: {quantity} {bybit_symbol.replace('USDT', '')}\n"
            f"Entry Price: {price}\n"
            f"Stop-Loss: {sl_price} ({STOPLOSS_PERCENT}%)\n"
            f"Take-Profit: {tp_price} ({TAKEPROFIT_PERCENT}%)\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        print(f"\nOrder executed successfully for {symbol}:")
        print(trade_details)
        
        # Send email notification
        email_subject = f"Trade Executed: {signal.upper()} {symbol}"
        send_email_notification(email_subject, trade_details)
    else:
        error_msg = f"Order failed for {symbol}: {order['retMsg']}"
        print(error_msg)
        send_email_notification(f"Trade Failed: {symbol}", error_msg)

def fetch_market_data(symbol, timeframe, limit=500):
    """Fetch OHLCV data with proper timezone handling"""
    try:
        ccxt_symbol = f"{symbol.replace('USDT', '')}/USDT:USDT"
        ohlcv = bitget.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(pytz.timezone('Africa/Lagos'))
        df.set_index('timestamp', inplace=True)
        df['EMA'] = df['close'].ewm(span=EMA_TREND, adjust=False).mean()
        return df
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None

def detect_trend(df, candle_index):
    """Determine trend for a specific candle using your exact method"""
    if candle_index < 1 or candle_index >= len(df):
        return "Sideways"
    
    ema_current = df['EMA'].iloc[candle_index]
    ema_prev = df['EMA'].iloc[candle_index-1]
    price = df['close'].iloc[candle_index]
    
    if ema_current > ema_prev and price > ema_current:
        return "Uptrend"
    elif ema_current < ema_prev and price < ema_current:
        return "Downtrend"
    return "Sideways"

def get_last_closed_trade():
    """Get details of the most recent closed trade"""
    try:
        trades = bybit.get_executions(category="linear", limit=50)
        if trades["retCode"] != 0:
            print(f"Error fetching trades: {trades['retMsg']}")
            return None
            
        trades = trades["result"]["list"]
        if not trades:
            return None

        for trade in sorted(trades, key=lambda x: int(x["execTime"]), reverse=True):
            symbol = trade["symbol"]
            
            positions = bybit.get_positions(category="linear", symbol=symbol)
            if positions["retCode"] != 0:
                continue
                
            position_open = any(float(p["size"]) > 0 for p in positions["result"]["list"])
            
            if not position_open and trade["closedSize"]:
                utc_time = datetime.fromtimestamp(int(trade["execTime"])/1000, pytz.UTC)
                lagos_time = utc_time.astimezone(pytz.timezone('Africa/Lagos'))
                
                return {
                    "symbol": symbol.replace("USDT", "") + "/USDT:USDT",
                    "close_time": lagos_time,
                    "close_price": float(trade["execPrice"]),
                    "side": "LONG" if trade["side"] == "Sell" else "SHORT",
                    "utc_close_time": utc_time
                }
        return None
    except Exception as e:
        print(f"Error fetching trade history: {e}")
        return None

def has_open_positions():
    """Check if there are any open positions"""
    try:
        positions = bybit.get_positions(category="linear", settleCoin="USDT")
        if positions['retCode'] == 0:
            return any(float(p['size']) > 0 for p in positions['result']['list'])
        print(f"Error checking positions: {positions['retMsg']}")
        return True  # Default to blocking if we can't check
    except Exception as e:
        print(f"Error checking open positions: {e}")
        return True  # Default to blocking if we can't check

def has_pending_orders():
    """Check if there are any pending orders"""
    try:
        # Check active orders
        active_orders = bybit.get_open_orders(category="linear", settleCoin="USDT")
        if active_orders['retCode'] == 0 and active_orders['result']['list']:
            return True
        
        # Check conditional orders
        conditional_orders = bybit.get_open_orders(
            category="linear",
            orderFilter='StopOrder',
            settleCoin="USDT"
        )
        if conditional_orders['retCode'] == 0 and conditional_orders['result']['list']:
            return True
        
        return False
    except Exception as e:
        print(f"Error checking pending orders: {e}")
        return True  # Default to blocking if we can't check

def has_trend_flipped(symbol, close_time, close_price, trade_side):
    """Check if trend has flipped since trade closed"""
    try:
        # Fetch market data since trade close
        df = fetch_market_data(symbol, '15m', 500)
        if df is None or len(df) < 2:
            print("Error: Not enough market data for trend analysis")
            return False  # Default to no flip if we can't check
            
        # Find the candle where trade was closed
        close_candle_idx = df.index.get_indexer([close_time], method='nearest')[0]
        if close_candle_idx < 1:  # Need at least 2 candles for trend detection
            close_candle_idx = 1
        
        # Check trend changes since closing
        for i in range(close_candle_idx + 1, len(df)):
            current_trend = detect_trend(df, i)
            
            # For long trades, we want to see flip to downtrend
            if trade_side == "LONG" and current_trend == "Downtrend":
                return True
            # For short trades, we want to see flip to uptrend
            elif trade_side == "SHORT" and current_trend == "Uptrend":
                return True
                
        return False
    except Exception as e:
        print(f"Error checking trend flip: {e}")
        return False  # Default to no flip if we can't check

def should_block_signals():
    """Determine if we should block signal checking based on all conditions"""
    # Check for open positions
    if has_open_positions():
        print("Blocking signals: Open positions exist")
        return True
        
    # Check for pending orders
    if has_pending_orders():
        print("Blocking signals: Pending orders exist")
        return True
        
    # Check trend flip on last traded symbol
    last_trade = get_last_closed_trade()
    if last_trade:
        print(f"Last trade was {last_trade['side']} on {last_trade['symbol']}")
        if not has_trend_flipped(
            last_trade['symbol'],
            last_trade['close_time'],
            last_trade['close_price'],
            last_trade['side']
        ):
            print("Blocking signals: Trend has not flipped since last trade")
            return True
            
    return False

def check_for_pullback_signal(symbol):
    """Check for first pullback signal on last closed candle"""
    # Set timezone (Africa/Lagos)
    lagos_tz = pytz.timezone('Africa/Lagos')
    
    # Fetch OHLCV data
    ohlcv_15m = bitget.fetch_ohlcv(symbol, TIMEFRAME, limit=500)
    ohlcv_1h = bitget.fetch_ohlcv(symbol, '1h', limit=500)
    
    # Create DataFrames
    df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamps to Africa/Lagos timezone
    for df in [df_15m, df_1h]:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(lagos_tz)
        df.set_index('timestamp', inplace=True)
    
    # Calculate EMAs for 15m
    df_15m['EMA_Fast'] = df_15m['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df_15m['EMA_Slow'] = df_15m['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df_15m['EMA_Trend'] = df_15m['close'].ewm(span=EMA_TREND, adjust=False).mean()
    
    # Calculate EMA for 1h (trend filter)
    df_1h['EMA_Trend'] = df_1h['close'].ewm(span=EMA_TREND, adjust=False).mean()
    
    # Resample 1h EMA to align with 15m data
    df_1h_resampled = df_1h['EMA_Trend'].resample('15min').ffill()
    df_15m['EMA_Trend_1h'] = df_1h_resampled
    
    # Generate initial signals
    df_15m['Signal'] = 0
    df_15m.loc[
        (df_15m['EMA_Fast'] > df_15m['EMA_Slow']) & 
        (df_15m['EMA_Fast'].shift(1) <= df_15m['EMA_Slow'].shift(1)) & 
        (df_15m['close'] > df_15m['EMA_Trend']) & 
        (df_15m['close'] > df_15m['EMA_Trend_1h']), 
        'Signal'] = 1
    
    df_15m.loc[
        (df_15m['EMA_Fast'] < df_15m['EMA_Slow']) & 
        (df_15m['EMA_Fast'].shift(1) >= df_15m['EMA_Slow'].shift(1)) & 
        (df_15m['close'] < df_15m['EMA_Trend']) & 
        (df_15m['close'] < df_15m['EMA_Trend_1h']), 
        'Signal'] = -1
    
    # Conservative pullback entries
    df_15m['Entry_Up'] = (
        (df_15m['EMA_Fast'] > df_15m['EMA_Slow']) & 
        (df_15m['close'].shift(1) < df_15m['EMA_Fast'].shift(1)) & 
        (df_15m['close'] > df_15m['EMA_Fast'])
    )
    
    df_15m['Entry_Down'] = (
        (df_15m['EMA_Fast'] < df_15m['EMA_Slow']) & 
        (df_15m['close'].shift(1) > df_15m['EMA_Fast'].shift(1)) & 
        (df_15m['close'] < df_15m['EMA_Fast'])
    )
    
    # Filter by trend
    df_15m['Entry_Up_Filtered'] = df_15m['Entry_Up'] & (df_15m['close'] > df_15m['EMA_Trend']) & (df_15m['close'] > df_15m['EMA_Trend_1h'])
    df_15m['Entry_Down_Filtered'] = df_15m['Entry_Down'] & (df_15m['close'] < df_15m['EMA_Trend']) & (df_15m['close'] < df_15m['EMA_Trend_1h'])
    
    # Track first conservative entry after each signal
    df_15m['First_Up_Arrow'] = False
    df_15m['First_Down_Arrow'] = False
    
    last_signal = 0
    for i in range(1, len(df_15m)):
        if df_15m['Signal'].iloc[i] == 1:
            last_signal = 1
        elif df_15m['Signal'].iloc[i] == -1:
            last_signal = -1

        if last_signal == 1 and df_15m['Entry_Up_Filtered'].iloc[i]:
            df_15m.at[df_15m.index[i], 'First_Up_Arrow'] = True
            last_signal = 0
        elif last_signal == -1 and df_15m['Entry_Down_Filtered'].iloc[i]:
            df_15m.at[df_15m.index[i], 'First_Down_Arrow'] = True
            last_signal = 0
    
    # Check the last closed candle (iloc[-2] because iloc[-1] is forming)
    last_candle = df_15m.iloc[-2]
    
    if last_candle['First_Up_Arrow']:
        return "buy"
    elif last_candle['First_Down_Arrow']:
        return "sell"
    return None

# ===== Main Execution =====
if __name__ == "__main__":
    print(f"Running multi-symbol strategy on {TIMEFRAME} timeframe")
    print(f"Trade amount: {TRADE_AMOUNT_USDT} USDT per symbol")
    print(f"Stop-loss: {STOPLOSS_PERCENT}%, Take-profit: {TAKEPROFIT_PERCENT}%")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    try:
        # Check blocking conditions first - MUST BE BEFORE ANY SYMBOL PROCESSING
        if should_block_signals():
            print("\nSignal checking blocked due to one or more conditions")
            print("1. Open positions exist OR")
            print("2. Pending orders exist OR")
            print("3. Trend hasn't flipped since last trade")
            sys.exit(0)  # Hard exit - no further processing
        
        # ONLY proceed if no blocking conditions exist
        for symbol in SYMBOLS:
            try:
                print(f"\nChecking {symbol}...")
                signal = check_for_pullback_signal(symbol)
                if signal:
                    current_price = get_current_price(symbol)
                    print(f"Signal detected on last closed candle: {signal.upper()}")
                    place_trade_order(symbol, signal, current_price)
                else:
                    print(f"No valid pullback signal for {symbol}")
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                send_email_notification(
                    f"Error processing {symbol}",
                    f"Error occurred while processing {symbol}:\n{str(e)}"
                )
                continue
            
            # Small delay between symbols to avoid rate limits
            time.sleep(1)
            
    except Exception as e:
        error_msg = f"Fatal error in main execution: {str(e)}"
        print(error_msg)
        send_email_notification("Trading Bot Crashed", error_msg)