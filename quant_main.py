"""
量化回测引擎 + 主程序入口

使用方法:
    python quant_main.py
    stats = run_backtest('BTC-USDT-SWAP', '2022-01-01', '2024-01-01')
"""

import numpy as np
import pandas as pd
import time, os, json, requests
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from quant_core import (
    find_swing_points, detect_trend, detect_fvg, check_fvg_mitigation,
    detect_order_blocks, detect_liquidity_sweeps, detect_consolidation,
    calculate_atr, TradingSignal
)
from quant_strategy import TrendFVGStrategy, StrategyConfig, Trade


# ============================================================================
# .env + 代理 + 数据层
# ============================================================================
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


def _okx_get(path: str, params: dict = None) -> dict:
    """OKX GET 请求，带重试"""
    url = f'{OKX_REST}{path}'
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, proxies=_proxies, timeout=15)
            data = r.json()
            if data.get('code') == '0':
                return data
            if attempt < 2:
                time.sleep(1)
        except Exception:
            if attempt < 2:
                time.sleep(2)
    return {}


def _to_okx_symbol(symbol: str) -> str:
    """BTC/USDT → BTC-USDT-SWAP"""
    s = symbol.replace('/USDT:USDT', '').replace('/USDT', '').replace(':USDT', '')
    return f'{s}-USDT-SWAP'


def fetch_okx_ohlcv(symbol: str, timeframe: str,
                    since: str, limit: int = 5000) -> pd.DataFrame:
    """
    OKX K线数据 (直接REST, 无ccxt依赖)
    OKX单次最多300根, 自动翻页
    """
    inst_id = _to_okx_symbol(symbol)
    tf_map = {'4h': '4H', '1d': '1D', '1h': '1H', '15m': '15m', '30m': '30m'}
    bar = tf_map.get(timeframe, timeframe)
    after_ts = int(pd.Timestamp(since).timestamp() * 1000)
    all_rows = []
    max_pages = max(30, limit // 300 + 5)

    for _ in range(max_pages):
        data = _okx_get('/api/v5/market/candles',
                        {'instId': inst_id, 'bar': bar, 'limit': '300',
                         'after': str(after_ts)})
        rows = data.get('data', [])
        if not rows:
            break
        all_rows.extend(rows)
        after_ts = int(rows[-1][0]) + 1
        if len(rows) < 300:
            break
        time.sleep(0.15)  # rate limit

    if not all_rows:
        return pd.DataFrame()

    # OKX返回9列: ts,open,high,low,close,vol,volCcy,volCcyQuote,confirm
    df = pd.DataFrame(all_rows).iloc[:, :6]
    df.columns = ['ts','open','high','low','close','vol']
    for col in ['open','high','low','close','vol']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)
    return df[~df.index.duplicated()]


def fetch_okx_ticker(symbol: str) -> dict:
    """获取当前价格"""
    inst_id = _to_okx_symbol(symbol)
    data = _okx_get('/api/v5/market/ticker', {'instId': inst_id})
    arr = data.get('data', [])
    if arr:
        return {'last': float(arr[0]['last']), 'bid': float(arr[0]['bidPx']),
                'ask': float(arr[0]['askPx'])}
    return {}


def fetch_sample_data(symbol: str = 'BTC-USDT-SWAP',
                      start: str = '2022-01-01',
                      end: str = '2024-01-01') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """获取回测数据 — OKX REST API"""
    proxy_tag = PROXY if PROXY else 'direct'
    print(f'[OKX] {proxy_tag}')

    # 测试连接
    ticker = fetch_okx_ticker(symbol)
    if ticker:
        print(f'[OKX] {symbol} = ${ticker["last"]:,.0f}')
    else:
        print('[WARN] 连接失败, 尝试 CSV/模拟数据')
        try:
            df_4h = pd.read_csv('data_4h.csv', index_col=0, parse_dates=True)
            df_1d = pd.read_csv('data_1d.csv', index_col=0, parse_dates=True)
            if len(df_4h) > 100:
                return df_4h, df_1d
        except:
            pass
        return _generate_synthetic_data(start, end)

    print(f'[OKX] 拉取 4H + 1D K线...')
    df_4h = fetch_okx_ohlcv(symbol, '4h', start, limit=5000)
    df_1d = fetch_okx_ohlcv(symbol, '1d', start, limit=2000)

    if len(df_4h) == 0:
        print('[WARN] 4H数据为空, 使用模拟数据')
        df_4h, df_1d = _generate_synthetic_data(start, end)
    else:
        print(f'[OK] 4H={len(df_4h)}根 ({df_4h.index[0]}~{df_4h.index[-1]}), '
              f'1D={len(df_1d)}根 ({df_1d.index[0]}~{df_1d.index[-1]})')
        end_ts = pd.Timestamp(end)
        df_4h = df_4h[df_4h.index <= end_ts]
        df_1d = df_1d[df_1d.index <= end_ts]

    return df_4h, df_1d


def _generate_synthetic_data(start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成带趋势特征的模拟数据（仅用于代码测试）"""
    np.random.seed(42)
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    # 4H数据
    dates_4h = pd.date_range(start_dt, end_dt, freq='4h')
    n_4h = len(dates_4h)

    # 模拟一个带趋势和震荡的价格序列
    price = 40000.0
    trend = 0
    prices_4h = []
    for i in range(n_4h):
        if i % 200 == 0:
            trend = np.random.choice([-1, 1]) * np.random.uniform(0.0005, 0.003)
        noise = np.random.normal(0, 0.01)
        price *= (1 + trend + noise)
        prices_4h.append(price)

    prices_4h = np.array(prices_4h)

    df_4h = pd.DataFrame({
        'open': np.zeros(n_4h),
        'high': np.zeros(n_4h),
        'low': np.zeros(n_4h),
        'close': prices_4h,
        'volume': np.random.uniform(100, 1000, n_4h)
    }, index=dates_4h)

    # 按顺序生成OHLC (确保 open <= high, open >= low 等)
    for i in range(n_4h):
        c = prices_4h[i]
        o = c * (1 + np.random.normal(0, 0.003))
        h = max(o, c) * (1 + abs(np.random.normal(0, 0.004)))
        l_ = min(o, c) * (1 - abs(np.random.normal(0, 0.004)))
        df_4h.iloc[i, df_4h.columns.get_loc('open')] = o
        df_4h.iloc[i, df_4h.columns.get_loc('high')] = h
        df_4h.iloc[i, df_4h.columns.get_loc('low')] = l_

    # 日线数据：从4H降采样
    df_1d = df_4h.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return df_4h, df_1d


# ============================================================================
# 回测引擎
# ============================================================================

def run_backtest(symbol: str = 'BTCUSDT',
                 start: str = '2022-01-01',
                 end: str = '2024-01-01',
                 initial_balance: float = 10000.0,
                 config: StrategyConfig = None) -> Dict:
    """
    运行回测

    Args:
        symbol: 交易对
        start/end: 回测区间
        initial_balance: 初始资金(USDT)
        config: 策略配置

    Returns:
        回测结果字典
    """
    print(f"\n{'='*60}")
    print(f"  趋势+FVG策略 回测")
    print(f"  品种: {symbol}  期间: {start} → {end}")
    print(f"  初始资金: ${initial_balance:,.0f}")
    print(f"{'='*60}\n")

    if config is None:
        config = StrategyConfig()

    # 1. 获取数据
    df_4h, df_1d = fetch_sample_data(symbol=symbol, start=start, end=end)

    # 2. 初始化策略
    strategy = TrendFVGStrategy(config)

    # 3. 逐K线回测 (不切片, 传视图)
    min_1d_bars = 20
    min_4h_bars = max(50, config.swing_window * 3)
    start_bar = min_4h_bars
    for i in range(min_4h_bars, len(df_4h)):
        if len(df_1d[df_1d.index <= df_4h.index[i]]) >= min_1d_bars:
            start_bar = i
            break

    print(f"  开始回测: bar {start_bar}/{len(df_4h)} ({df_4h.index[start_bar]})")

    balance = initial_balance
    equity_history = [balance]
    total_bars = len(df_4h)

    # 调试统计
    skip_reasons = {'misaligned': 0, 'consolidation': 0, 'no_signal': 0,
                    'low_rr': 0, 'in_position': 0}

    for i in range(start_bar, total_bars):
        current_4h = df_4h.iloc[:i+1]          # 视图, 不复制
        current_time = df_4h.index[i]
        current_1d = df_1d[df_1d.index <= current_time]

        if len(current_1d) < 20:
            equity_history.append(balance)
            continue

        result = strategy.update(current_4h, current_1d, balance)
        reason = str(result.get('reason', ''))[:30]

        if '不一致' in reason:
            skip_reasons['misaligned'] += 1
        elif '震荡' in reason:
            skip_reasons['consolidation'] += 1
        elif '盈亏比' in reason:
            skip_reasons['low_rr'] += 1
        elif '无符合' in reason:
            skip_reasons['no_signal'] += 1

        if result.get('action') in ('close', 'reduce'):
            pnl = result.get('pnl', 0)
            balance += pnl
            if pnl != 0:
                ts = str(df_4h.index[i])[:16]
                print(f"  [{ts}] {result.get('reason','')[:45]}  PnL:{pnl:+.2f}")

        if result.get('action') == 'enter':
            ts = str(df_4h.index[i])[:16]
            print(f"  [{ts}] {'多' if 'long' in str(result.get('direction','')) else '空'} "
                  f"@{result['entry_price']:.2f}  RR:{result['rr']:.1f}")

        equity_history.append(balance)

    # 4. 平仓结算
    if strategy.position is not None:
        final_px = float(df_4h['close'].iloc[-1])
        r = strategy._close_position(final_px, df_4h.index[-1], '回测结束', balance)
        if r.get('pnl'):
            balance += r['pnl']

    # 5. 统计
    stats = strategy.get_stats()
    stats['initial_balance'] = initial_balance
    stats['final_balance'] = balance
    stats['total_return'] = (balance - initial_balance) / initial_balance * 100
    stats['equity_history'] = equity_history
    stats['trades'] = strategy.trades

    # 6. 打印结果
    print(f"\n  调试: 多周期不一致={skip_reasons['misaligned']}, "
          f"震荡={skip_reasons['consolidation']}, "
          f"无信号={skip_reasons['no_signal']}, "
          f"低盈亏比={skip_reasons['low_rr']}")
    print(f"\n{'='*60}")
    print(f"  回测结果")
    print(f"{'='*60}")
    print(f"  初始资金:   ${initial_balance:>12,.2f}")
    print(f"  最终资金:   ${balance:>12,.2f}")
    print(f"  总收益:     {stats['total_return']:>11.2f}%")
    print(f"  交易次数:   {stats['total_trades']:>12}")
    if stats['total_trades'] > 0:
        print(f"  胜率:       {stats['win_rate']:>11.1f}%")
        print(f"  平均盈利:   ${stats['avg_win']:>12,.2f}")
        print(f"  平均亏损:   ${stats['avg_loss']:>12,.2f}")
        print(f"  盈亏比:     {stats['profit_factor']:>12.2f}")
        print(f"  最佳交易:   ${stats['best_trade']:>12,.2f}")
        print(f"  最差交易:   ${stats['worst_trade']:>12,.2f}")

    return stats


# ============================================================================
# 结果可视化
# ============================================================================

def plot_results(stats: Dict):
    """画出回测结果图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过绘图 (pip install matplotlib)")
        return

    trades = stats.get('trades', [])
    equity = stats.get('equity_history', [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('趋势+FVG策略 回测结果', fontsize=14)

    # 图1: 资金曲线
    ax1 = axes[0][0]
    ax1.plot(equity, 'b-', linewidth=1, alpha=0.8)
    ax1.axhline(y=stats['initial_balance'], color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('资金曲线')
    ax1.set_ylabel('账户余额 (USDT)')
    ax1.grid(True, alpha=0.3)

    # 图2: 每笔交易PnL
    ax2 = axes[0][1]
    if trades:
        pnls = [t.pnl for t in trades]
        colors = ['g' if p > 0 else 'r' for p in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_title(f'每笔交易PnL (共{len(trades)}笔)')
        ax2.set_xlabel('交易序号')
        ax2.set_ylabel('PnL (USDT)')
        ax2.grid(True, alpha=0.3)

    # 图3: 累计PnL
    ax3 = axes[1][0]
    if trades:
        cum_pnl = np.cumsum([t.pnl for t in trades])
        ax3.plot(cum_pnl, 'b-', linewidth=1.5)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                         where=(cum_pnl >= np.array([0]*len(cum_pnl))),
                         color='g', alpha=0.15)
        ax3.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                         where=(cum_pnl < np.array([0]*len(cum_pnl))),
                         color='r', alpha=0.15)
        ax3.set_title('累计PnL')
        ax3.set_xlabel('交易序号')
        ax3.set_ylabel('累计PnL (USDT)')
        ax3.grid(True, alpha=0.3)

    # 图4: 统计信息
    ax4 = axes[1][1]
    ax4.axis('off')
    stats_text = f"""
    策略统计
    ──────────────────
    初始资金:   ${stats['initial_balance']:,.0f}
    最终资金:   ${stats['final_balance']:,.0f}
    总收益率:   {stats['total_return']:.1f}%
    交易次数:   {stats['total_trades']}
    胜率:       {stats.get('win_rate', 0):.1f}%
    盈亏比(PF): {stats.get('profit_factor', 0):.2f}
    平均盈利:   ${stats.get('avg_win', 0):,.0f}
    平均亏损:   ${stats.get('avg_loss', 0):,.0f}
    最佳交易:   ${stats.get('best_trade', 0):,.0f}
    最差交易:   ${stats.get('worst_trade', 0):,.0f}
    ──────────────────
    策略配置
    主周期:     4h
    大周期:     1d
    每笔风险:   2%
    最低盈亏比: 1.5
    """
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('backtest_result.png', dpi=150)
    print("\n[OK] 图表已保存为 backtest_result.png")
    plt.show()


# ============================================================================
# 单个信号分析 (用于实时监控)
# ============================================================================

def analyze_current_setup(df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> Dict:
    """
    分析当前盘面的信号情况（不执行交易）
    用于每日复盘或半自动交易
    """
    from quant_core import generate_signals as gen_sig

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    open_4h = df_4h['open'].values
    close_4h = df_4h['close'].values

    # 大周期趋势
    sh_d, sl_d = find_swing_points(df_1d['high'].values, df_1d['low'].values)
    trend_d = detect_trend(sh_d, sl_d)

    # 4H趋势
    sh_4, sl_4 = find_swing_points(high_4h, low_4h)
    trend_4 = detect_trend(sh_4, sl_4)

    # FVG
    fvgs = detect_fvg(open_4h, high_4h, low_4h, close_4h)
    current_idx = len(close_4h) - 1
    fvgs = check_fvg_mitigation(fvgs, high_4h, low_4h, current_idx)
    fresh_fvgs = [f for f in fvgs
                  if not f.mitigated
                  and current_idx - f.start_idx < 50]

    # OB
    obs = detect_order_blocks(open_4h, high_4h, low_4h, close_4h, sh_4, sl_4)

    # 流动性
    sweeps = detect_liquidity_sweeps(high_4h, low_4h, open_4h, close_4h, sh_4, sl_4)

    # 信号
    signals = gen_sig(df_4h, fresh_fvgs, obs, sweeps, trend_4, current_idx)

    # 震荡度
    consol = detect_consolidation(high_4h, low_4h)

    return {
        'trend_1d': trend_d,
        'trend_4h': trend_4,
        'fvgs': fresh_fvgs,
        'obs': obs,
        'sweeps': sweeps,
        'signals': signals,
        'consolidation': float(consol[-1]),
        'current_price': float(close_4h[-1]),
        'current_time': str(df_4h.index[-1])
    }


# ============================================================================
# 主入口
# ============================================================================

def check_okx_connection() -> bool:
    """测试 OKX 连接"""
    ticker = fetch_okx_ticker('BTC-USDT-SWAP')
    if ticker:
        px = ticker['last']
        print(f'[OK] OKX 连接正常  BTC = ${px:,.0f}')
        return True
    print('[ERR] OKX 连接失败')
    return False


# ============================================================================
# 主入口
# ============================================================================

if __name__ == '__main__':
    proxy_info = PROXY if PROXY else 'direct'
    print(f'''
    ╔══════════════════════════════════════╗
    ║  加密货币量化交易系统                     ║
    ║  信号: FVG+猎杀+OB+突破+斜形+颈线         ║
    ║  数据: OKX REST  |  代理: {proxy_info:<18s} ║
    ╚══════════════════════════════════════╝
    ''')

    if not check_okx_connection():
        print('[WARN] OKX 连接失败\n')

    config = StrategyConfig(swing_window=3, risk_percent=2.0, min_rr_ratio=0.8)

    stats = run_backtest(
        symbol='BTC-USDT-SWAP',
        start='2022-01-01',
        end='2024-12-01',
        initial_balance=10000.0,
        config=config
    )

    # === 绘图 ===
    if stats.get('total_trades', 0) > 0:
        plot_results(stats)
    else:
        print("\n[WARN] 无交易记录 (可能是模拟数据导致信号不足)")

    # === 输出交易明细 ===
    if stats.get('trades'):
        print(f"\n{'='*60}")
        print(f"  交易明细 (最近20笔)")
        print(f"{'='*60}")
        trades_df = pd.DataFrame([{
            '进场时间': t.entry_time,
            '出场时间': t.exit_time,
            '方向': '多' if t.direction == 1 else '空',
            '进场价': f'{t.entry_price:.2f}',
            '出场价': f'{t.exit_price:.2f}',
            '盈亏': f'{t.pnl:+.2f}',
            '原因': t.exit_reason
        } for t in stats['trades'][-20:]])
        print(trades_df.to_string(index=False))
