"""
OKX 全币种扫描器 — 找出当前符合交易条件的币种

用法:
    python quant_scanner.py                     # 扫描所有永续合约
    python quant_scanner.py --top 20            # 只扫描交易量前20
    python quant_scanner.py --symbols BTC/USDT,ETH/USDT  # 指定币种
"""
import sys, os, time, argparse, requests, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_core import *
from quant_core import fetch_okx_sentiment
from quant_strategy import TrendFVGStrategy, StrategyConfig

# ---------------------------------------------------------------------------
# 本地缓存
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(symbol, timeframe):
    """缓存文件路径"""
    name = symbol.replace('/USDT:USDT','').replace('/USDT','').replace(':','_')
    return os.path.join(CACHE_DIR, f'{name}_{timeframe}.csv')

def _cache_valid(filepath, max_age_sec=14400):
    """检查缓存是否有效 (默认4小时)"""
    if not os.path.exists(filepath):
        return False
    return time.time() - os.path.getmtime(filepath) < max_age_sec

def fetch_ohlcv_cached(symbol, timeframe, limit=300, max_age=14400):
    """拉K线 — 带本地缓存, 4H以上缓存4小时"""
    cache_file = _cache_path(symbol, timeframe)
    if _cache_valid(cache_file, max_age):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if len(df) >= 50:
            return df
    # 拉新数据
    df = fetch_ohlcv(symbol, timeframe, limit)
    if df is not None and len(df) >= 50:
        df.to_csv(cache_file)
    return df
# 从 .env 加载
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()

PROXY = os.environ.get('OKX_PROXY', 'http://127.0.0.1:7890')
_proxies = {'http': PROXY, 'https': PROXY} if PROXY else None
OKX_REST = 'https://www.okx.com'


def _okx_get(path, params=None):
    for _ in range(2):
        try:
            r = requests.get(f'{OKX_REST}{path}', params=params,
                           proxies=_proxies, timeout=15)
            d = r.json()
            if d.get('code') == '0':
                return d
            time.sleep(0.5)
        except:
            time.sleep(1)
    return {}


def _to_okx_id(symbol):
    """BTC/USDT:USDT → BTC-USDT-SWAP"""
    s = symbol.replace('/USDT:USDT', '').replace('/USDT', '').replace(':USDT', '')
    return f'{s}-USDT-SWAP'


def fetch_ohlcv(symbol, timeframe, limit=300):
    """拉K线"""
    inst = _to_okx_id(symbol)
    tf = {'4h':'4H','1d':'1D','1h':'1H','15m':'15m'}.get(timeframe, timeframe)
    data = _okx_get('/api/v5/market/candles',
                    {'instId': inst, 'bar': tf, 'limit': str(limit)})
    rows = data.get('data', [])
    if len(rows) < 50:
        return None
    # OKX返回9列: ts,open,high,low,close,vol,volCcy,volCcyQuote,confirm
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ['ts','open','high','low','close','vol']
    for c in ['open','high','low','close','vol']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)
    return df

# ---------------------------------------------------------------------------
# 单币种分析
# ---------------------------------------------------------------------------
def analyze_symbol(symbol, config=None):
    """
    分析单个币种 — 返回当前是否有交易信号
    Returns: dict with signal info, or None
    """
    if config is None:
        config = StrategyConfig(swing_window=3, min_rr_ratio=0.8)

    # 拉数据
    df_4h = fetch_ohlcv_cached(symbol, '4h', limit=400)
    df_1d = fetch_ohlcv_cached(symbol, '1d', limit=200)
    if df_4h is None or df_1d is None or len(df_4h) < 100:
        return None

    # 计算指标
    o, h, l, c = df_4h['open'].values, df_4h['high'].values, df_4h['low'].values, df_4h['close'].values
    current_idx = len(c) - 1
    current_price = float(c[-1])

    sh, sl = find_swing_points(h, l, window=config.swing_window)
    trend_4h = detect_trend(sh, sl)

    # 多周期
    sh_d, sl_d = find_swing_points(df_1d['high'].values, df_1d['low'].values,
                                    window=config.swing_window)
    trend_1d = detect_trend(sh_d, sl_d)
    alignment = multi_timeframe_alignment(trend_4h, trend_1d)

    # 各种信号检测
    fvgs = detect_fvg(o, h, l, c)
    fvgs = check_fvg_mitigation(fvgs, h, l, current_idx)
    fvgs = [f for f in fvgs if not f.mitigated and current_idx - f.start_idx < 50]

    obs = detect_order_blocks(o, h, l, c, sh, sl)
    sweeps = detect_liquidity_sweeps(h, l, o, c, sh, sl)
    necklines = detect_necklines(sh, sl, h, l, c, current_idx)
    wedge = detect_wedge(h, l, c, sh, sl, current_idx)

    atr = calculate_atr(h, l, c)
    current_atr = atr[current_idx]

    # 生成信号
    signals = generate_signals(df_4h, fvgs, obs, sweeps, sh, sl, trend_4h, current_idx)

    # 多周期加分
    trend_bonus = 0.0
    if alignment != 0 and trend_4h['direction'] == alignment:
        trend_bonus = 0.15
    elif trend_4h['direction'] != 0 and trend_1d['direction'] != 0:
        trend_bonus = -0.2
    for s in signals:
        s.confidence = min(1.0, max(0.1, s.confidence + trend_bonus))

    # 过滤有效信号
    valid = [s for s in signals if s.confidence >= 0.5 and s.rr_ratio >= config.min_rr_ratio]
    if not valid:
        return None

    # 按信度排序
    valid.sort(key=lambda s: s.confidence, reverse=True)
    best = valid[0]

    return {
        'symbol': symbol,
        'price': current_price,
        'direction': 'LONG' if best.direction == 1 else 'SHORT',
        'entry': best.entry_price,
        'stop': best.stop_loss,
        'tp1': best.take_profit_1,
        'tp2': best.take_profit_2,
        'rr': best.rr_ratio,
        'confidence': best.confidence,
        'signal_type': best.signal_type,
        'reason': best.reason,
        'trend_4h': trend_4h['direction'],
        'trend_1d': trend_1d['direction'],
        'alignment': alignment,
        'atr': current_atr,
        'atr_pct': current_atr / current_price * 100,
        'volatility_rank': 0  # filled later
    }

# ---------------------------------------------------------------------------
# 主扫描
# ---------------------------------------------------------------------------
# 内置币种池 (ranked by OKX 24h volume on startup)
_SYMBOL_POOL = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'DOGE/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOT/USDT:USDT',
    'LINK/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'LTC/USDT:USDT',
    'ETC/USDT:USDT', 'FIL/USDT:USDT', 'APT/USDT:USDT', 'ARB/USDT:USDT',
    'OP/USDT:USDT', 'NEAR/USDT:USDT', 'INJ/USDT:USDT', 'SUI/USDT:USDT',
    'SEI/USDT:USDT', 'TIA/USDT:USDT', 'WIF/USDT:USDT', 'PEPE/USDT:USDT',
    'BONK/USDT:USDT', 'STX/USDT:USDT', 'IMX/USDT:USDT', 'GRT/USDT:USDT',
    'AAVE/USDT:USDT', 'CRV/USDT:USDT', 'SAND/USDT:USDT', 'MANA/USDT:USDT',
    'FET/USDT:USDT', 'RNDR/USDT:USDT', 'MKR/USDT:USDT', 'MATIC/USDT:USDT',
]


def _fetch_volume(symbol):
    """获取单币24h交易量(USD)"""
    s = symbol.replace('/USDT:USDT', '').replace('/USDT', '').replace(':USDT', '')
    data = _okx_get('/api/v5/market/ticker', {'instId': f'{s}-USDT-SWAP'})
    if data and data.get('data'):
        vol = float(data['data'][0].get('volCcy24h', 0) or 0)
        # 过滤异常值 (>100B 说明单位不是USD)
        return vol if 0 < vol < 1e11 else 0
    return 0


def get_ranked_symbols(top_n=None):
    """按24h交易量排序 — 1小时缓存"""
    rank_cache = os.path.join(CACHE_DIR, '_volume_rank.json')
    if _cache_valid(rank_cache, 3600):  # 1小时有效
        with open(rank_cache) as f:
            symbols = json.load(f)
        if top_n:
            symbols = symbols[:top_n]
        return symbols

    ranked = []
    for sym in _SYMBOL_POOL:
        vol = _fetch_volume(sym)
        ranked.append((sym, vol))
        time.sleep(0.08)
    ranked.sort(key=lambda x: x[1], reverse=True)
    result = [s for s, v in ranked if v > 0]
    for must in ['BTC/USDT:USDT', 'ETH/USDT:USDT']:
        if must not in result:
            result.insert(0, must)
    # 存缓存
    with open(rank_cache, 'w') as f:
        json.dump(result, f)
    if top_n:
        result = result[:top_n]
    return result

def scan_all(symbols=None, top_n=None, min_vol=0):
    """扫描币种 — 默认按24h交易量排序"""
    if symbols is None:
        print('获取24h交易量排名...')
        symbols = get_ranked_symbols(top_n)
        print(f'活跃币种: {len(symbols)} (已过滤下架/零交易量币)')

    print(f'扫描 {len(symbols)} 个币种...')

    results = []
    for i, sym in enumerate(symbols):
        try:
            r = analyze_symbol(sym)
            if r:
                results.append(r)
                print(f'  [{i+1}/{len(symbols)}] {sym:20s} -> {r["direction"]:5s} '
                      f'{r["signal_type"]:12s} conf={r["confidence"]:.2f} RR={r["rr"]:.1f}')
            else:
                print(f'  [{i+1}/{len(symbols)}] {sym:20s} -> (无信号)', end='\r')
        except Exception as e:
            print(f'  [{i+1}/{len(symbols)}] {sym:20s} -> ERR: {str(e)[:50]}')
        time.sleep(0.3)  # rate limit

    return results

# ---------------------------------------------------------------------------
# 展示
# ---------------------------------------------------------------------------
def display_results(results):
    if not results:
        print('\n  (无符合条件的币种)')
        return

    # 按信度 + RR 综合排序
    results.sort(key=lambda r: r['confidence'] + r['rr'] * 0.3, reverse=True)

    print(f'\n{"="*100}')
    print(f'  OKX 全币种扫描结果 — 当前可开仓信号 (共{len(results)}个)')
    print(f'{"="*100}')
    print(f'  {"币种":<18s} {"方向":6s} {"价格":>10s} {"止损":>10s} {"TP1":>10s} '
          f'{"RR":>5s} {"信度":>5s} {"信号类型":12s} {"4H":>3s} {"1D":>3s} {"波动%":>6s}')
    print(f'  {"-"*95}')

    for r in results:
        d4 = ['▼','─','▲'][r['trend_4h']+1] if -1 <= r['trend_4h'] <= 1 else '?'
        d1 = ['▼','─','▲'][r['trend_1d']+1] if -1 <= r['trend_1d'] <= 1 else '?'
        aligned = '*' if r['alignment'] != 0 else ' '
        print(f'  {r["symbol"]:<18s} {r["direction"]:6s} {r["price"]:>10.2f} '
              f'{r["stop"]:>10.2f} {r["tp1"]:>10.2f} '
              f'{r["rr"]:>4.1f} {r["confidence"]:>4.1f}  '
              f'{r["signal_type"]:<12s} {aligned}{d4} {d1} '
              f'{r["atr_pct"]:>5.2f}%  {r["reason"]}')

    print(f'  {"-"*95}')
    print(f'  * = 多周期共振   4H/1D: ▲多头 ▼空头 ─震荡')

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OKX 币种扫描器')
    parser.add_argument('--top', type=int, default=30, help='扫描交易量前N的币种')
    parser.add_argument('--symbols', type=str, help='指定币种,逗号分隔 (如 BTC/USDT,ETH/USDT)')
    parser.add_argument('--min-vol', type=float, default=1000000, help='最小24h交易量(USD)')
    args = parser.parse_args()

    print(f'\n  OKX 币种扫描器 — 基于趋势+FVG+OB+突破+斜形+颈线')
    print(f'  代理: {PROXY if PROXY else "direct"}')
    print()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = None

    t0 = time.time()
    results = scan_all(symbols=symbols, top_n=args.top, min_vol=args.min_vol)
    elapsed = time.time() - t0

    display_results(results)
    print(f'\n  扫描完成: {len(results)} 个信号 / {args.top} 个币种 / {elapsed:.0f}s')
