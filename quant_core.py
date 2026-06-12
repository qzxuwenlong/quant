"""
量化交易核心模块 — 基于交易课程体系的指标计算

将老师的核心概念转化为可计算的量化指标：
  - 趋势定义 (高低点不断移动)
  - FVG/真空区 (Fair Value Gap = 价格快速通过的区域)
  - OB/密集区  (Order Block = 筹码交换密集区)
  - 流动性猎杀 (影线突破前高前低后反转)
  - 多空比情绪 (物极必反)
  - 盈亏比与仓位计算
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SwingPoint:
    """波段高低点"""
    index: int
    price: float
    is_high: bool  # True=高点, False=低点


@dataclass
class FVG:
    """真空区 / Fair Value Gap
    定义：三根K线中，第一根和第三根的K线实体之间没有重叠的区域
    对应老师说的"价格快速通过、几乎没有筹码交换的区域"
    """
    start_idx: int
    end_idx: int
    top: float       # 真空区上沿
    bottom: float    # 真空区下沿
    direction: int   # 1=上涨FVG(支撑), -1=下跌FVG(阻力)
    mitigated: bool = False  # 是否已被回踩测试过


@dataclass
class OrderBlock:
    """订单块 / 密集区
    定义：趋势反转前最后一根反向K线的实体区间
    对应老师说的"筹码交换密集区"、"主力吸筹/出货区"
    """
    start_idx: int
    end_idx: int
    top: float
    bottom: float
    direction: int   # 1=看涨OB(支撑), -1=看跌OB(阻力)
    tested: bool = False


@dataclass
class LiquiditySweep:
    """流动性猎杀
    定义：影线突破前高/前低，但实体收盘未突破
    对应老师说的"影线突破才算杀流动性，实体突破=真突破"
    """
    idx: int
    level: float      # 被猎杀的价位
    direction: int    # 1=向上猎杀（扫空头止损），-1=向下猎杀（扫多头止损）
    is_sweep: bool    # True=影线猎杀(假突破), False=实体突破(真突破)


# ============================================================================
# 1. 趋势定义 — "高点和低点不断移动"
# ============================================================================

def find_swing_points(high: np.ndarray, low: np.ndarray,
                      window: int = 5) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    找波段高低点
    老师原话："上升趋势 = 高点不断上移 + 低点不断上移"

    Args:
        high/low: K线高低价序列
        window: 左右各看多少根K线确认波段点
    Returns:
        (swing_highs, swing_lows)
    """
    highs, lows = [], []
    n = len(high)

    for i in range(window, n - window):
        # 波段高点：比左右各 window 根K线的高点都高
        if high[i] == max(high[i-window : i+window+1]):
            highs.append(SwingPoint(index=i, price=high[i], is_high=True))
        # 波段低点
        if low[i] == min(low[i-window : i+window+1]):
            lows.append(SwingPoint(index=i, price=low[i], is_high=False))

    return highs, lows


def detect_trend(swing_highs: List[SwingPoint],
                 swing_lows: List[SwingPoint],
                 lookback: int = 3) -> Dict:
    """
    判断当前趋势方向
    老师原话："低点不断抬高 = 多头趋势；高点不断降低 = 空头趋势"

    取最近 lookback 个波段点，判断趋势

    Returns:
        {
            'direction': 1(多头)/-1(空头)/0(震荡),
            'strength': 0.0~1.0 趋势强度,
            'last_high': 最后一个有效高点的价格,
            'last_low': 最后一个有效低点的价格
        }
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {'direction': 0, 'strength': 0, 'last_high': None, 'last_low': None}

    # 取最近的波段点
    recent_highs = sorted(swing_highs, key=lambda x: x.index)[-lookback:]
    recent_lows = sorted(swing_lows, key=lambda x: x.index)[-lookback:]

    high_prices = [h.price for h in recent_highs]
    low_prices = [l.price for l in recent_lows]

    # 判断趋势 — 放宽: lookback=2 就够了
    # 且只要2/3满足就算 (不需要ALL)
    min_ok = max(1, len(high_prices) - 1)  # 至少N-1个满足
    highs_rising = sum(1 for i in range(1, len(high_prices))
                       if high_prices[i] > high_prices[i-1]) >= min_ok
    lows_rising = sum(1 for i in range(1, len(low_prices))
                      if low_prices[i] > low_prices[i-1]) >= min_ok
    highs_falling = sum(1 for i in range(1, len(high_prices))
                        if high_prices[i] < high_prices[i-1]) >= min_ok
    lows_falling = sum(1 for i in range(1, len(low_prices))
                       if low_prices[i] < low_prices[i-1]) >= min_ok

    if highs_rising and lows_rising:
        direction = 1
        strength = min(1.0, sum(1 for i in range(1, len(high_prices))
                                if high_prices[i] > high_prices[i-1]) / (len(high_prices)-1))
    elif highs_falling and lows_falling:
        direction = -1
        strength = min(1.0, sum(1 for i in range(1, len(low_prices))
                                if low_prices[i] < low_prices[i-1]) / (len(low_prices)-1))
    else:
        direction = 0
        strength = 0.5

    return {
        'direction': direction,
        'strength': strength,
        'last_high': recent_highs[-1].price if recent_highs else None,
        'last_low': recent_lows[-1].price if recent_lows else None,
        'highs_rising': highs_rising,
        'lows_falling': lows_falling
    }


# ============================================================================
# 2. FVG 真空区检测 — "价格快速通过、没有筹码的区域"
# ============================================================================

def detect_fvg(open_: np.ndarray, high: np.ndarray,
               low: np.ndarray, close: np.ndarray,
               mode: str = 'body') -> List[FVG]:
    """
    检测 Fair Value Gap (真空区)

    mode='body':  用K线实体判断 (标准ICT, crypto更适用)
    mode='wick':  用影线极端值判断 (更严格, 信号更少)

    老师原话：
    "真空区就是价格一根线直接下来的地方，没有交易量集中的地方"

    FVG规则 (body mode): 三根K线为一组 K0-K1-K2
    上涨FVG: K0的收盘 < K2的开盘，之间没有重叠 → 价格跳空上涨
    下跌FVG: K0的收盘 > K2的开盘，之间没有重叠 → 价格跳空下跌
    真空区 = K0收盘价与K2开盘价之间的空隙
    """
    fvgs = []
    n = len(close)

    for i in range(1, n - 1):
        if mode == 'body':
            # === 上涨FVG: K[i-1]收盘 < K[i+1]开盘 (价格跳空上涨) ===
            if close[i-1] < open_[i+1]:
                fvg = FVG(
                    start_idx=i,
                    end_idx=i+1,
                    top=open_[i+1],     # 真空区上沿
                    bottom=close[i-1],  # 真空区下沿
                    direction=1
                )
                fvgs.append(fvg)

            # === 下跌FVG: K[i-1]收盘 > K[i+1]开盘 (价格跳空下跌) ===
            if close[i-1] > open_[i+1]:
                fvg = FVG(
                    start_idx=i,
                    end_idx=i+1,
                    top=close[i-1],     # 真空区上沿
                    bottom=open_[i+1],  # 真空区下沿
                    direction=-1
                )
                fvgs.append(fvg)

        else:  # mode == 'wick' (严格模式)
            if high[i-1] < low[i+1]:
                fvgs.append(FVG(
                    start_idx=i, end_idx=i+1,
                    top=low[i+1], bottom=high[i-1], direction=1
                ))
            if low[i-1] > high[i+1]:
                fvgs.append(FVG(
                    start_idx=i, end_idx=i+1,
                    top=low[i-1], bottom=high[i+1], direction=-1
                ))

    return fvgs


def check_fvg_mitigation(fvgs: List[FVG], high: np.ndarray,
                         low: np.ndarray, current_idx: int) -> List[FVG]:
    """
    检查哪些FVG已经被价格"测试过"（回踩确认）
    老师原话："这个位置已经被测试过了，再测试的话效果就弱了"
    一个位置测试次数越多，支撑/阻力越弱 ("一鼓作气再而衰三而竭")
    """
    for fvg in fvgs:
        if fvg.end_idx >= current_idx:
            continue
        # 检查从 FVG 结束到当前位置，价格是否进入了 FVG 区间
        for j in range(fvg.end_idx + 1, min(current_idx + 1, len(high))):
            if fvg.bottom <= high[j] and fvg.top >= low[j]:
                fvg.mitigated = True
                break
    return fvgs


def find_nearest_fvg(fvgs: List[FVG], current_price: float,
                     direction: int = 0) -> Optional[FVG]:
    """
    找到最近的未回踩的FVG
    direction: 1=找下方支撑FVG(做多), -1=找上方阻力FVG(做空), 0=都不找(返回None)
    """
    candidates = [f for f in fvgs if not f.mitigated]
    if direction == 0:
        return None  # 震荡不找FVG
    elif direction == 1:
        candidates = [f for f in candidates if f.direction == 1
                      and f.bottom < current_price]
        candidates.sort(key=lambda f: current_price - f.bottom)
    elif direction == -1:
        candidates = [f for f in candidates if f.direction == -1
                      and f.top > current_price]
        candidates.sort(key=lambda f: f.top - current_price)

    return candidates[0] if candidates else None


# ============================================================================
# 3. OB 订单块/密集区检测 — "筹码交换最密集的地方"
# ============================================================================

def detect_order_blocks(open_: np.ndarray, high: np.ndarray,
                        low: np.ndarray, close: np.ndarray,
                        swing_highs: List[SwingPoint],
                        swing_lows: List[SwingPoint]) -> List[OrderBlock]:
    """
    检测 Order Block (密集区/订单块)

    老师原话：
    "密集区的50%位置 = 主力的持仓均价"
    "密集区的峰值在震荡区间的50%处"

    OB定义：
    - 看涨OB: 下跌趋势中，最后一根阴线（或连续阴线）的实体 = 主力吸筹区
    - 看跌OB: 上涨趋势中，最后一根阳线（或连续阳线）的实体 = 主力出货区
    实际简化：在波段低点/高点附近找反转K线的实体区间
    """
    obs = []

    for sp in swing_lows:
        # 波段低点 = 潜在的看涨OB位置
        # 找低点之前的一根或连续阴线作为OB
        if sp.index < 2:
            continue
        # 找低点附近的反转K线
        ob_low = low[sp.index]
        ob_high = high[sp.index]
        # 往前扩展找吸筹区
        for j in range(sp.index - 1, max(0, sp.index - 5), -1):
            if close[j] < open_[j]:  # 阴线
                ob_low = min(ob_low, low[j])
                ob_high = max(ob_high, high[j])
            else:
                break

        if ob_high > ob_low:
            obs.append(OrderBlock(
                start_idx=sp.index - 1,
                end_idx=sp.index,
                top=ob_high,
                bottom=ob_low,
                direction=1
            ))

    for sp in swing_highs:
        # 波段高点 = 潜在的看跌OB位置
        if sp.index < 2:
            continue
        ob_low = low[sp.index]
        ob_high = high[sp.index]
        for j in range(sp.index - 1, max(0, sp.index - 5), -1):
            if close[j] > open_[j]:  # 阳线
                ob_low = min(ob_low, low[j])
                ob_high = max(ob_high, high[j])
            else:
                break

        if ob_high > ob_low:
            obs.append(OrderBlock(
                start_idx=sp.index - 1,
                end_idx=sp.index,
                top=ob_high,
                bottom=ob_low,
                direction=-1
            ))

    return obs


def get_ob_mid_price(ob: OrderBlock) -> float:
    """老师原话：密集区50%位置 = 主力持仓均价"""
    return (ob.top + ob.bottom) / 2


# ============================================================================
# 4. 流动性猎杀检测 — "影线突破才算杀流动性"
# ============================================================================

def detect_liquidity_sweeps(high: np.ndarray, low: np.ndarray,
                            open_: np.ndarray, close: np.ndarray,
                            swing_highs: List[SwingPoint],
                            swing_lows: List[SwingPoint],
                            lookback: int = 20) -> List[LiquiditySweep]:
    """
    检测流动性猎杀

    老师原话：
    "流动性就是你止损的位置，一般都在前高和前低"
    "影线突破才算杀流动性，实体突破不是"
    "它趁着流动性把空单的止损给打掉"

    检测逻辑：
    1. 价格影线突破前高/前低
    2. 收盘价回到突破位以内（确认是假突破/猎杀）
    """
    sweeps = []
    n = len(close)

    # 取最近的前高前低
    recent_high_levels = sorted(
        [(h.price, h.index) for h in swing_highs if h.index < n - 1],
        key=lambda x: x[0], reverse=True
    )[:lookback]
    recent_low_levels = sorted(
        [(l.price, l.index) for l in swing_lows if l.index < n - 1],
        key=lambda x: x[0]
    )[:lookback]

    for i in range(max(2, n - 30), n - 1):
        # 向上猎杀：影线突破前高但收盘回来
        for level, idx in recent_high_levels:
            if idx >= i:
                continue
            if high[i] > level and close[i] < level:
                sweeps.append(LiquiditySweep(
                    idx=i, level=level, direction=1, is_sweep=True
                ))
                break  # 每根K线只记一个

        # 向下猎杀：影线跌破前低但收盘回来
        for level, idx in recent_low_levels:
            if idx >= i:
                continue
            if low[i] < level and close[i] > level:
                sweeps.append(LiquiditySweep(
                    idx=i, level=level, direction=-1, is_sweep=True
                ))
                break

    return sweeps


# ============================================================================
# 5. 多空比情绪指标 (物极必反) — OKX API
# ============================================================================

# 缓存: 避免频繁调API
_SENTIMENT_CACHE = {}

def fetch_okx_sentiment(symbol: str = 'BTC', period: str = '1D',
                        proxies: dict = None) -> dict:
    """
    从 OKX 拉多空持仓人数比 (散户情绪)

    老师: "多空比极端高位 → 要杀多头; 极端低位 → 要杀空头"

    API: /api/v5/rubik/stat/contracts/long-short-account-ratio
    返回: { 'latest': 最新ratio, 'avg_7d': 7日均值,
            'avg_30d': 30日均值, 'extreme': 'overheat_long'/'overheat_short'/None }
    """
    import time, requests
    cache_key = f'{symbol}_{period}'
    now = time.time()
    if cache_key in _SENTIMENT_CACHE:
        cached_time, cached_data = _SENTIMENT_CACHE[cache_key]
        if now - cached_time < 3600:  # 1小时缓存
            return cached_data

    try:
        url = 'https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio'
        params = {'ccy': symbol.replace('/USDT',''), 'period': period, 'limit': '90'}
        r = requests.get(url, params=params, proxies=proxies, timeout=10)
        data = r.json()
        if data['code'] != '0' or not data['data']:
            return {'latest': 1.0, 'avg_7d': 1.0, 'avg_30d': 1.0, 'extreme': None}

        ratios = [float(item[1]) for item in data['data']]
        latest = ratios[-1]
        avg_7d = sum(ratios[-7:]) / min(7, len(ratios))
        avg_30d = sum(ratios[-30:]) / min(30, len(ratios))

        # 极端判断: >2.0 多头过热, <0.5 空头过热
        extreme = None
        if latest > 2.0:
            extreme = 'overheat_long'
        elif latest < 0.5:
            extreme = 'overheat_short'

        result = {'latest': latest, 'avg_7d': avg_7d,
                  'avg_30d': avg_30d, 'extreme': extreme}
        _SENTIMENT_CACHE[cache_key] = (now, result)
        return result
    except Exception:
        return {'latest': 1.0, 'avg_7d': 1.0, 'avg_30d': 1.0, 'extreme': None}


def get_sentiment_bonus(symbol: str = 'BTC', proxies: dict = None) -> float:
    """
    获取情绪修正值，用于调整信号置信度

    Returns: -0.15 (多头过热=做空加分), +0.15 (空头过热=做多加分), 0 (正常)
    """
    sent = fetch_okx_sentiment(symbol, proxies=proxies)
    extreme = sent.get('extreme')
    if extreme == 'overheat_long':
        return -0.15   # 多头过热 → 做空信号加权，做多信号降权
    elif extreme == 'overheat_short':
        return +0.15   # 空头过热 → 做多信号加权，做空信号降权
    return 0.0


# ============================================================================
# 6. 盈亏比与仓位计算 — "1:1减仓一半, 1:2再减一半"
# ============================================================================

def calculate_position_size(account_balance: float,
                            risk_percent: float,
                            entry_price: float,
                            stop_loss: float) -> float:
    """
    以损定量 — 老师原话："固定止损金额，反推仓位"

    Returns: 应开仓的数量(contracts)
    """
    risk_amount = account_balance * (risk_percent / 100)
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance == 0:
        return 0
    return risk_amount / stop_distance


def calculate_take_profit_levels(entry_price: float, stop_loss: float,
                                 direction: int = 1) -> Dict[str, float]:
    """
    计算分批止盈价位
    老师原话：
    "到达1:1减仓50%, 到达1:2再减仓50%(剩25%), 然后推保护"
    """
    risk = abs(entry_price - stop_loss)
    if direction == 1:  # 多头
        return {
            'tp1': entry_price + risk * 1.0,   # 1:1
            'tp2': entry_price + risk * 2.0,   # 1:2
            'tp3': entry_price + risk * 3.0,   # 1:3 (极限)
            'breakeven': entry_price            # 保本位
        }
    else:  # 空头
        return {
            'tp1': entry_price - risk * 1.0,
            'tp2': entry_price - risk * 2.0,
            'tp3': entry_price - risk * 3.0,
            'breakeven': entry_price
        }


def calculate_rr_ratio(entry_price: float, stop_loss: float,
                       take_profit: float, direction: int = 1) -> float:
    """计算盈亏比"""
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    return reward / risk if risk > 0 else 0


# ============================================================================
# 7. 跨周期趋势一致性 — "大周期定方向，小周期找入场"
def multi_timeframe_alignment(trend_4h: Dict, trend_1d: Dict) -> int:
    """
    多周期一致性检查
    老师原话："大周期定方向，小周期找入场"
    - 日线多头+4H多头 = 强烈做多
    - 日线空头+4H空头 = 强烈做空
    - 日线震荡+4H有方向 = 允许跟随4H (大周期没阻力)
    - 日线与4H反向 = 禁止 (逆大势)

    Returns: 1=做多, -1=做空, 0=观望
    """
    d4 = trend_4h['direction']
    d1 = trend_1d['direction']

    if d4 == d1:
        return d4                      # 完美共振
    elif d1 == 0 and d4 != 0:
        return d4                      # 日线无方向, 跟随4H
    else:
        return 0                       # 日线4H反向, 不做


# ============================================================================
# 8. 辅助函数
# ============================================================================

# ============================================================================
# 9. 斜形/衰竭检测 — "斜形在末尾=衰竭"
# ============================================================================

@dataclass
class Wedge:
    """斜形结构"""
    start_idx: int        # 起点
    end_idx: int          # 终点
    upper_slope: float    # 上轨斜率
    lower_slope: float    # 下轨斜率
    direction: int        # 1=上升斜形(看跌反转), -1=下降斜形(看涨反转)
    breakout_idx: int     # 突破K线位置 (-1 表示未突破)
    quality: float        # 0~1 形态质量


def detect_wedge(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 swing_highs: List[SwingPoint], swing_lows: List[SwingPoint],
                 current_idx: int, min_points: int = 5) -> Optional[Wedge]:
    """
    检测斜形/衰竭形态

    老师: "斜形在末尾=衰竭, 是反转信号"
    - 上升斜形: 高点和低点都抬升, 但上轨斜率 < 下轨斜率 (多头衰竭) → 看跌
    - 下降斜形: 高点和低点都降低, 但下轨收敛更快 (空头衰竭) → 看涨

    用最近 N 个波段点拟合两条趋势线, 判断是否收敛
    """
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return None

    # 取最近的 swing points (只能用到 current_idx 之前的)
    recent_h = sorted([h for h in swing_highs if h.index <= current_idx],
                      key=lambda x: x.index)[-4:]
    recent_l = sorted([l for l in swing_lows if l.index <= current_idx],
                      key=lambda x: x.index)[-4:]

    if len(recent_h) < 3 or len(recent_l) < 3:
        return None

    # 线性回归拟合上轨(用高点)和下轨(用低点)
    h_x = np.array([h.index for h in recent_h])
    h_y = np.array([h.price for h in recent_h])
    l_x = np.array([l.index for l in recent_l])
    l_y = np.array([l.price for l in recent_l])

    # 归一化 x 以便比较斜率
    h_x_norm = (h_x - h_x[0]) / max(1, h_x[-1] - h_x[0])
    l_x_norm = (l_x - l_x[0]) / max(1, l_x[-1] - l_x[0])

    upper_slope = np.polyfit(h_x_norm, h_y, 1)[0]
    lower_slope = np.polyfit(l_x_norm, l_y, 1)[0]

    # 判断斜形类型
    # 上升斜形: 两线都向上, 下轨斜率 > 上轨 → 多头衰竭
    #   (下轨上升更快 = 价格被压缩 = 即将选择方向)
    if upper_slope > 0 and lower_slope > upper_slope * 1.05:
        wedge = Wedge(start_idx=recent_h[0].index, end_idx=current_idx,
                      upper_slope=upper_slope, lower_slope=lower_slope,
                      direction=1, breakout_idx=-1, quality=0.5)

        # 检查是否已向下突破下轨
        last_low_y = np.polyval(np.polyfit(l_x_norm, l_y, 1),
                                (current_idx - l_x[0]) / max(1, l_x[-1] - l_x[0]))
        if close[current_idx] < last_low_y:
            wedge.breakout_idx = current_idx
            wedge.quality = 0.75
        return wedge

    # 下降斜形: 两线都向下, |上轨斜率| > |下轨斜率| → 空头衰竭
    #   (上轨下降更快 = 价格被压缩 = 即将反转)
    if upper_slope < 0 and lower_slope < 0 and abs(upper_slope) > abs(lower_slope) * 1.05:
        wedge = Wedge(start_idx=recent_h[0].index, end_idx=current_idx,
                      upper_slope=upper_slope, lower_slope=lower_slope,
                      direction=-1, breakout_idx=-1, quality=0.6)

        last_high_y = np.polyval(np.polyfit(h_x_norm, h_y, 1),
                                 (current_idx - h_x[0]) / max(1, h_x[-1] - h_x[0]))
        if close[current_idx] > last_high_y:
            wedge.breakout_idx = current_idx
            wedge.quality = 0.8
        return wedge

    return None


# ============================================================================
# 10. 颈线位检测 — "多空分水岭, 顶底转换位"
# ============================================================================

@dataclass
class Neckline:
    """颈线位"""
    price: float          # 颈线价位
    direction: int        # 1=阻力转支撑(做多), -1=支撑转阻力(做空)
    touches: int          # 被测试次数
    formed_at: int        # 形成时的K线索引
    broken_at: int        # 被突破时的K线索引 (-1=未破)
    retested: bool        # 是否已被回踩确认


def detect_necklines(swing_highs: List[SwingPoint],
                     swing_lows: List[SwingPoint],
                     high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     current_idx: int, tolerance: float = 0.02) -> List[Neckline]:
    """
    检测颈线位 — 只找最好的2个(支撑/阻力各一), 避免O(n²)全量扫描

    老师: "颈线位=多空分水岭"
    """
    necklines = []
    n = len(close)

    # 辅助: 从波段点列表找最佳价格聚类
    def _find_best_cluster(points, max_to_check=30):
        """找最大聚类, 返回 (avg_price, touch_count, formed_at) 或 None"""
        recent = sorted([p for p in points if p.index <= current_idx],
                        key=lambda x: x.index)[-max_to_check:]
        best = None
        for i in range(len(recent) - 2):
            cluster = [recent[i]]
            base = recent[i].price
            for j in range(i + 1, len(recent)):
                if abs(recent[j].price - base) / base < tolerance:
                    cluster.append(recent[j])
            if len(cluster) >= 3:
                avg = np.mean([c.price for c in cluster])
                if best is None or len(cluster) > best[1]:
                    best = (avg, len(cluster), cluster[-1].index)
        return best

    # 阻力颈线 (高点聚类)
    best_h = _find_best_cluster(swing_highs)
    if best_h and best_h[1] >= 4:
        avg_price, touches, formed_at = best_h
        broken = -1
        for k in range(formed_at + 1, n):
            if close[k] > avg_price:
                broken = k
                break
        retested = False
        if broken > 0:
            for k in range(broken + 1, n):
                if abs(low[k] - avg_price) / avg_price < tolerance:
                    retested = True
                    break
        necklines.append(Neckline(
            price=avg_price, direction=1, touches=touches,
            formed_at=formed_at, broken_at=broken, retested=retested
        ))

    # 支撑颈线 (低点聚类)
    best_l = _find_best_cluster(swing_lows)
    if best_l and best_l[1] >= 4:
        avg_price, touches, formed_at = best_l
        broken = -1
        for k in range(formed_at + 1, n):
            if close[k] < avg_price:
                broken = k
                break
        retested = False
        if broken > 0:
            for k in range(broken + 1, n):
                if abs(high[k] - avg_price) / avg_price < tolerance:
                    retested = True
                    break
        necklines.append(Neckline(
            price=avg_price, direction=-1, touches=touches,
            formed_at=formed_at, broken_at=broken, retested=retested
        ))

    return necklines


# ============================================================================
# 11. 波浪结构检测
# ============================================================================

@dataclass
class WaveStructure:
    """波浪结构"""
    start_idx: int          # 起点
    end_idx: int            # 终点
    direction: int          # 1=上升5浪, -1=下降5浪
    waves: list             # [(idx, price), ...] 5个浪的转折点
    complete: bool          # 5浪是否完成
    fib_ratios: dict        # 斐波那契比率
    confidence: float       # 0~1


def detect_elliott_wave(swing_highs: List[SwingPoint],
                        swing_lows: List[SwingPoint],
                        high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        current_idx: int) -> Optional[WaveStructure]:
    """
    检测完成的5浪推动结构

    规则 (来自老师):
    - 浪2 回调浪1的 50-78.6% (不破起点)
    - 浪3 不能是最短的 (通常最长)
    - 浪4 回调浪3的 23.6-50% (不与浪1重叠)
    - 浪5 通常 = 浪1 的长度 (或浪1的61.8%)

    只找"已确认完成"的结构 — 不做预测
    """
    # 取最近的 swing points
    recent_h = sorted([h for h in swing_highs if h.index <= current_idx],
                      key=lambda x: x.index)
    recent_l = sorted([l for l in swing_lows if l.index <= current_idx],
                      key=lambda x: x.index)

    def _find_impulse(swings, direction):
        """在波段点序列中找5浪推动"""
        if len(swings) < 5:
            return None

        # 取最后 N 个同向波段点
        points = swings[-min(len(swings), 12):]
        if len(points) < 5:
            return None

        # 尝试前5个、后5个、中间5个等不同的组合
        candidates = []
        for offset in range(min(3, len(points) - 4)):
            p5 = points[offset:offset+5]
            w1 = abs(p5[1].price - p5[0].price)
            w2 = abs(p5[2].price - p5[1].price)
            w3 = abs(p5[3].price - p5[2].price)
            w4 = abs(p5[4].price - p5[3].price)
            w5 = abs(p5[4].price - p5[3].price)  # placeholder, 浪5是最后一个到当前价

            if w1 <= 0 or w3 <= 0:
                continue

            # 规则1: 浪2回调50-90%浪1
            retrace_2 = w2 / w1 if w1 > 0 else 999
            if retrace_2 < 0.3 or retrace_2 > 0.95:
                continue

            # 规则2: 浪3不是最短的
            if w3 < w1 or w3 < w5 * 0.8:
                continue

            # 规则3: 浪4回调浪3的23.6-50%
            retrace_4 = w4 / w3 if w3 > 0 else 999
            if retrace_4 < 0.15 or retrace_4 > 0.6:
                continue

            # 如果目前只有4浪, 检查浪4是否与浪1重叠
            if direction == 1:
                if p5[3].price <= p5[0].price * 1.02:  # 重叠了
                    continue
            else:
                if p5[3].price >= p5[0].price * 0.98:  # 重叠了
                    continue

            # 通过基本规则 → 可能是有效5浪
            # 计算质量分数
            quality = 0.5
            if 0.5 <= retrace_2 <= 0.786:
                quality += 0.1
            if 0.236 <= retrace_4 <= 0.5:
                quality += 0.1
            if 0.8 <= w5 / max(w1, 0.001) <= 1.2:
                quality += 0.1  # 浪5 ≈ 浪1
            if w3 >= w1 * 1.2:
                quality += 0.1  # 浪3延长

            candidates.append({
                'points': p5,
                'quality': min(1.0, quality),
                'ratios': {'w2/w1': retrace_2, 'w3/w1': w3/w1, 'w4/w3': retrace_4}
            })

        if not candidates:
            return None
        best = max(candidates, key=lambda x: x['quality'])
        if best['quality'] < 0.5:
            return None
        return best

    # 尝试上升5浪 (牛市推动)
    up = _find_impulse(recent_l, 1)  # 低点是上升浪的转折点
    if up:
        pts = up['points']
        return WaveStructure(
            start_idx=pts[0].index, end_idx=pts[-1].index,
            direction=1,
            waves=[(p.index, p.price) for p in pts],
            complete=True,
            fib_ratios=up['ratios'],
            confidence=up['quality']
        )

    # 尝试下降5浪 (熊市推动)
    down = _find_impulse(recent_h, -1)
    if down:
        pts = down['points']
        return WaveStructure(
            start_idx=pts[0].index, end_idx=pts[-1].index,
            direction=-1,
            waves=[(p.index, p.price) for p in pts],
            complete=True,
            fib_ratios=down['ratios'],
            confidence=down['quality']
        )

    return None


def detect_abc_correction(swing_highs: List[SwingPoint],
                          swing_lows: List[SwingPoint],
                          current_idx: int) -> Optional[dict]:
    """
    检测ABC三波回调 (5浪推动后的反向调整)

    老师: "5浪走完→ABC调整→等C浪结束做反弹/反转"
    简化: 在5浪结束后找3波反向结构
    """
    recent_h = sorted([h for h in swing_highs if h.index <= current_idx],
                      key=lambda x: x.index)
    recent_l = sorted([l for l in swing_lows if l.index <= current_idx],
                      key=lambda x: x.index)

    def _find_abc(swings, direction):
        if len(swings) < 3:
            return None
        pts = swings[-min(len(swings), 8):]
        if len(pts) < 3:
            return None

        # 简单的3波识别: A-B-C
        for offset in range(min(3, len(pts) - 2)):
            a, b, c = pts[offset], pts[offset+1], pts[offset+2]
            len_a = abs(b.price - a.price)
            len_c = abs(c.price - b.price)

            if len_a <= 0:
                continue

            # C通常 >= A (或 = 1.618 * A)
            ratio = len_c / len_a
            if 0.8 <= ratio <= 2.0:
                return {
                    'points': [a, b, c],
                    'ratio_c_a': ratio,
                    'complete': c.index <= current_idx - 2  # C已经走完
                }
        return None

    # 上升5浪后的ABC下跌
    abc = _find_abc(recent_h if recent_h else recent_l, -1)
    if abc and abc['complete']:
        return abc

    # 下降5浪后的ABC反弹
    abc2 = _find_abc(recent_l if recent_l else recent_h, 1)
    if abc2 and abc2['complete']:
        return abc2

    return None


# ============================================================================
# 12. 辅助函数
# ============================================================================

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  period: int = 14) -> np.ndarray:
    """ATR — 用于动态止损"""
    atr = np.zeros(len(close))
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(abs(high[1:] - close[:-1]),
                               abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, tr[0])

    for i in range(period, len(close)):
        atr[i] = np.mean(tr[i-period+1:i+1])

    # 填充前面
    atr[:period] = atr[period]
    return atr


def detect_consolidation(high: np.ndarray, low: np.ndarray,
                         window: int = 20) -> np.ndarray:
    """
    检测震荡/横盘区间
    老师原话："横盘越久，破位的概率越大"

    Returns: consolidation_score (0~1, 越大越震荡)
    """
    n = len(high)
    score = np.zeros(n)
    for i in range(window, n):
        h_range = max(high[i-window:i+1]) - min(low[i-window:i+1])
        avg_price = np.mean(close[i-window:i+1])
        if avg_price > 0:
            score[i] = 1 - min(1.0, h_range / (avg_price * 0.05))
    return score


# ============================================================================
# 9. 综合信号生成
# ============================================================================

@dataclass
class TradingSignal:
    """交易信号"""
    timestamp: pd.Timestamp
    direction: int        # 1=做多, -1=做空
    entry_price: float
    stop_loss: float
    take_profit_1: float  # 1:1 减仓位
    take_profit_2: float  # 1:2 再减仓位
    rr_ratio: float
    signal_type: str      # 'fvg_reclaim', 'ob_reversal', 'liquidity_sweep', 'breakout'
    confidence: float     # 0~1
    reason: str           # 信号触发原因


def generate_signals(df: pd.DataFrame,
                     fvgs: List[FVG],
                     obs: List[OrderBlock],
                     sweeps: List[LiquiditySweep],
                     swing_highs: List[SwingPoint],
                     swing_lows: List[SwingPoint],
                     trend_info: Dict,
                     current_idx: int) -> List[TradingSignal]:
    """
    综合所有指标生成交易信号:
      信号1: FVG回踩
      信号2: 流动性猎杀反转
      信号3: OB(密集区)回踩    [NEW]
      信号4: 突破+回踩入场     [NEW]
    """
    signals = []
    current_price = float(df['close'].iloc[current_idx])
    current_time = df.index[current_idx]
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr = calculate_atr(high_arr, low_arr, close_arr)
    current_atr = atr[current_idx]
    trend_dir = trend_info.get('direction', 0)

    if trend_dir == 0:
        return signals

    # === 力度过滤 — 老师："这根K线力度不够，不急着进" ===
    # K线力度 = 实体/ATR + 成交量/均量
    vol_arr = df['volume'].values if 'volume' in df.columns else np.ones(len(close_arr))
    avg_vol = np.mean(vol_arr[max(0, current_idx-20):current_idx+1])
    current_vol = vol_arr[current_idx]
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

    body = abs(close_arr[current_idx] - df['open'].values[current_idx])
    body_ratio = body / current_atr if current_atr > 0 else 0

    # 强信号: 实体>0.5xATR 且 量>1.2x均量 → 信度+0.1
    # 弱信号: 实体<0.3xATR 或 量<0.8x均量 → 信度-0.15
    is_strong = body_ratio >= 0.5 and vol_ratio >= 1.2
    is_weak = body_ratio < 0.3 or vol_ratio < 0.8
    strength_bonus = 0.10 if is_strong else (-0.15 if is_weak else 0)

    # === 信号1: FVG回踩 ===
    # 找最近的顺势FVG作为支撑/阻力参考
    # 用FVG边界做止损，盈亏比自然过滤远距离FVG
    nearest_fvg = find_nearest_fvg(fvgs, current_price, direction=trend_dir)
    if nearest_fvg:
        if nearest_fvg.direction == 1 and trend_dir == 1:
            # 做多: FVG支撑在下方，止损=FVG底部-ATR缓冲
            entry = current_price
            sl = nearest_fvg.bottom - current_atr * 1.0
            tp_levels = calculate_take_profit_levels(entry, sl, 1)
            rr = calculate_rr_ratio(entry, sl, tp_levels['tp1'], 1)

            if rr >= 0.8:
                signals.append(TradingSignal(
                    timestamp=current_time, direction=1,
                    entry_price=entry, stop_loss=sl,
                    take_profit_1=tp_levels['tp1'],
                    take_profit_2=tp_levels['tp2'],
                    rr_ratio=rr, signal_type='fvg_reclaim',
                    confidence=0.55,
                    reason=f'FVG多 {nearest_fvg.bottom:.0f}-{nearest_fvg.top:.0f}'
                ))

        elif nearest_fvg.direction == -1 and trend_dir == -1:
            # 做空: FVG阻力在上方，止损=FVG顶部+ATR缓冲
            entry = current_price
            sl = nearest_fvg.top + current_atr * 1.0
            tp_levels = calculate_take_profit_levels(entry, sl, -1)
            rr = calculate_rr_ratio(entry, sl, tp_levels['tp1'], -1)

            if rr >= 0.8:
                signals.append(TradingSignal(
                    timestamp=current_time, direction=-1,
                    entry_price=entry, stop_loss=sl,
                    take_profit_1=tp_levels['tp1'],
                    take_profit_2=tp_levels['tp2'],
                    rr_ratio=rr, signal_type='fvg_reclaim',
                    confidence=0.55,
                    reason=f'FVG空 {nearest_fvg.bottom:.0f}-{nearest_fvg.top:.0f}'
                ))

    # === 信号2: 流动性猎杀后反转 ===
    # 猎杀必须在最近5根K线内发生
    for sweep in sweeps:
        bars_ago = current_idx - sweep.idx
        if bars_ago < 0 or bars_ago > 5:
            continue

        # 向上猎杀后价格回落 — 做空信号
        # 条件: 1) 影线扫了前高  2) 当前价格回落到被扫价位以下
        #        3) 趋势为空头 (顺势) 或 趋势=0 (允许反转猜顶)
        if sweep.direction == 1 and close_arr[current_idx] < sweep.level:
            if trend_dir in (-1, 0):  # 空头趋势或震荡都允许(猎杀本身是强反转信号)
                entry = current_price
                sl = sweep.level + current_atr * 1.5  # 止损 = 前高之上
                tp_levels = calculate_take_profit_levels(entry, sl, -1)
                rr = calculate_rr_ratio(entry, sl, tp_levels['tp1'], -1)
                if rr >= 1.0:
                    signals.append(TradingSignal(
                        timestamp=current_time, direction=-1,
                        entry_price=entry, stop_loss=sl,
                        take_profit_1=tp_levels['tp1'],
                        take_profit_2=tp_levels['tp2'],
                        rr_ratio=rr, signal_type='liquidity_sweep',
                        confidence=0.7,
                        reason=f'猎杀前高{sweep.level:.2f}后回落'
                    ))

        # 向下猎杀后价格回升 — 做多信号
        if sweep.direction == -1 and close_arr[current_idx] > sweep.level:
            if trend_dir in (1, 0):
                entry = current_price
                sl = sweep.level - current_atr * 1.5
                tp_levels = calculate_take_profit_levels(entry, sl, 1)
                rr = calculate_rr_ratio(entry, sl, tp_levels['tp1'], 1)
                if rr >= 1.0:
                    signals.append(TradingSignal(
                        timestamp=current_time, direction=1,
                        entry_price=entry, stop_loss=sl,
                        take_profit_1=tp_levels['tp1'],
                        take_profit_2=tp_levels['tp2'],
                        rr_ratio=rr, signal_type='liquidity_sweep',
                        confidence=0.7,
                        reason=f'猎杀前低{sweep.level:.2f}后回升'
                    ))

    # === 信号3: OB(密集区)回踩 ===
    # 老师: "密集区50%位置=主力均价, 价格回到密集区就是机会"
    for ob in obs:
        if ob.direction != trend_dir:
            continue
        mid = (ob.top + ob.bottom) / 2
        in_zone = (ob.bottom - current_atr * 0.5 <= current_price <=
                   ob.top + current_atr * 0.5)
        if current_idx - ob.end_idx > 30:
            continue
        if not in_zone:
            continue

        if ob.direction == 1:
            entry = current_price
            sl = ob.bottom - current_atr * 0.5
            tp = calculate_take_profit_levels(entry, sl, 1)
            rr = calculate_rr_ratio(entry, sl, tp['tp1'], 1)
            if rr >= 1.0:  # OB需要更高的RR门槛
                signals.append(TradingSignal(
                    timestamp=current_time, direction=1,
                    entry_price=entry, stop_loss=sl,
                    take_profit_1=tp['tp1'], take_profit_2=tp['tp2'],
                    rr_ratio=rr, signal_type='ob_reversal',
                    confidence=0.6,
                    reason=f'OB多 {ob.bottom:.0f}-{ob.top:.0f}'
                ))
        else:
            entry = current_price
            sl = ob.top + current_atr * 0.5
            tp = calculate_take_profit_levels(entry, sl, -1)
            rr = calculate_rr_ratio(entry, sl, tp['tp1'], -1)
            if rr >= 1.0:
                signals.append(TradingSignal(
                    timestamp=current_time, direction=-1,
                    entry_price=entry, stop_loss=sl,
                    take_profit_1=tp['tp1'], take_profit_2=tp['tp2'],
                    rr_ratio=rr, signal_type='ob_reversal',
                    confidence=0.6,
                    reason=f'OB空 {ob.bottom:.0f}-{ob.top:.0f}'
                ))

    # === 信号4: 突破+回踩 ("最稳的做多方式") ===
    recent_highs = sorted([h for h in swing_highs if h.index < current_idx],
                          key=lambda x: x.index)
    recent_lows  = sorted([l for l in swing_lows if l.index < current_idx],
                          key=lambda x: x.index)

    if trend_dir == 1 and len(recent_highs) >= 2 and len(recent_lows) >= 1:
        prev_high = recent_highs[-2]
        last_low = recent_lows[-1]
        breakout = any(close_arr[j] > prev_high.price
                       for j in range(prev_high.index + 1, current_idx))
        if breakout:
            dist = abs(current_price - prev_high.price)
            if dist <= current_atr * 2.0:
                entry = current_price
                sl = last_low.price - current_atr * 0.5
                if abs(entry - sl) <= current_atr * 5:
                    tp = calculate_take_profit_levels(entry, sl, 1)
                    rr = calculate_rr_ratio(entry, sl, tp['tp1'], 1)
                    if rr >= 1.0:
                        signals.append(TradingSignal(
                            timestamp=current_time, direction=1,
                            entry_price=entry, stop_loss=sl,
                            take_profit_1=tp['tp1'], take_profit_2=tp['tp2'],
                            rr_ratio=rr, signal_type='breakout',
                            confidence=0.65,
                            reason=f'突破回踩多 前高{prev_high.price:.0f}'
                        ))

    if trend_dir == -1 and len(recent_lows) >= 2 and len(recent_highs) >= 1:
        prev_low = recent_lows[-2]
        last_high = recent_highs[-1]
        breakdown = any(close_arr[j] < prev_low.price
                        for j in range(prev_low.index + 1, current_idx))
        if breakdown:
            dist = abs(current_price - prev_low.price)
            if dist <= current_atr * 2.0:
                entry = current_price
                sl = last_high.price + current_atr * 0.5
                if abs(entry - sl) <= current_atr * 5:
                    tp = calculate_take_profit_levels(entry, sl, -1)
                    rr = calculate_rr_ratio(entry, sl, tp['tp1'], -1)
                    if rr >= 1.0:
                        signals.append(TradingSignal(
                            timestamp=current_time, direction=-1,
                            entry_price=entry, stop_loss=sl,
                            take_profit_1=tp['tp1'], take_profit_2=tp['tp2'],
                            rr_ratio=rr, signal_type='breakout',
                            confidence=0.65,
                            reason=f'跌破回抽空 前低{prev_low.price:.0f}'
                        ))

    # === 信号5: 斜形突破 ===
    wedge = detect_wedge(high_arr, low_arr, close_arr, swing_highs, swing_lows, current_idx)
    # 斜形信号条件严格: 质量>0.7 + 突破确认 + 高RR
    if wedge and wedge.breakout_idx > 0 and wedge.quality >= 0.7:
        bars_since = current_idx - wedge.breakout_idx
        if 1 <= bars_since <= 5:  # 突破后1-5根K内有效
            if wedge.direction == 1 and trend_dir <= 0:
                entry = current_price
                recent_h = sorted([h for h in swing_highs if h.index <= current_idx],
                                  key=lambda x: x.index)[-1]
                sl = recent_h.price + current_atr * 0.5
                tp = calculate_take_profit_levels(entry, sl, -1)
                rr = calculate_rr_ratio(entry, sl, tp['tp1'], -1)
                if rr >= 1.3:
                    signals.append(TradingSignal(
                        timestamp=current_time, direction=-1,
                        entry_price=entry, stop_loss=sl,
                        take_profit_1=tp['tp1'], take_profit_2=tp['tp2'],
                        rr_ratio=rr, signal_type='wedge',
                        confidence=0.6,
                        reason=f'上升斜形衰竭 下破'
                    ))
            elif wedge.direction == -1 and trend_dir >= 0:
                entry = current_price
                recent_l = sorted([l for l in swing_lows if l.index <= current_idx],
                                  key=lambda x: x.index)[-1]
                sl = recent_l.price - current_atr * 0.5
                tp = calculate_take_profit_levels(entry, sl, 1)
                rr = calculate_rr_ratio(entry, sl, tp['tp1'], 1)
                if rr >= 1.3:
                    signals.append(TradingSignal(
                        timestamp=current_time, direction=1,
                        entry_price=entry, stop_loss=sl,
                        take_profit_1=tp['tp1'], take_profit_2=tp['tp2'],
                        rr_ratio=rr, signal_type='wedge',
                        confidence=0.6,
                        reason=f'下降斜形衰竭 上破'
                    ))

    # === K线力度 + 成交量修正 ===
    for s in signals:
        s.confidence = min(1.0, max(0.05, s.confidence + strength_bonus))

    # === 信号6: 颈线位回踩 ===
    necklines = detect_necklines(swing_highs, swing_lows, high_arr, low_arr,
                                 close_arr, current_idx)
    for nl in necklines:
        # 只取已突破且已回踩的颈线 (最稳)
        if nl.broken_at < 0 or not nl.retested:
            continue
        # 必须在最近10根K内回踩
        if current_idx - nl.broken_at > 20:
            continue
        # 价格必须在颈线附近
        if abs(current_price - nl.price) / nl.price > 0.02:
            continue

        if nl.direction == 1 and trend_dir >= 0:
            # 阻力转支撑 → 做多
            entry = current_price
            sl = nl.price * 0.98  # 颈线下方2%
            tp = calculate_take_profit_levels(entry, sl, 1)
            rr = calculate_rr_ratio(entry, sl, tp['tp1'], 1)
            if rr >= 1.3:  # 颈线信号要求高RR
                signals.append(TradingSignal(
                    timestamp=current_time, direction=1,
                    entry_price=entry, stop_loss=sl,
                    take_profit_1=tp['tp1'], take_profit_2=tp['tp2'],
                    rr_ratio=rr, signal_type='neckline',
                    confidence=0.7,
                    reason=f'颈线支撑回踩 {nl.price:.0f} ({nl.touches}次测试)'
                ))

        if nl.direction == -1 and trend_dir <= 0:
            # 支撑转阻力 → 做空
            entry = current_price
            sl = nl.price * 1.02  # 颈线上方2%
            tp = calculate_take_profit_levels(entry, sl, -1)
            rr = calculate_rr_ratio(entry, sl, tp['tp1'], -1)
            if rr >= 1.3:
                signals.append(TradingSignal(
                    timestamp=current_time, direction=-1,
                    entry_price=entry, stop_loss=sl,
                    take_profit_1=tp['tp1'], take_profit_2=tp['tp2'],
                    rr_ratio=rr, signal_type='neckline',
                    confidence=0.7,
                    reason=f'颈线阻力回抽 {nl.price:.0f} ({nl.touches}次测试)'
                ))

    # === 信号7: 波浪结构 (5浪完成+ABC结束) ===
    wave = detect_elliott_wave(swing_highs, swing_lows, high_arr, low_arr, close_arr, current_idx)
    if wave and wave.complete and wave.confidence >= 0.6:
        abc = detect_abc_correction(swing_highs, swing_lows, current_idx)
        if abc and abc['complete']:
            if wave.direction == 1 and trend_dir >= 0:
                entry = current_price
                sl = abc['points'][-1].price - current_atr * 1.0
                tp = calculate_take_profit_levels(entry, sl, 1)
                rr = calculate_rr_ratio(entry, sl, tp['tp1'], 1)
                if rr >= 1.2:
                    signals.append(TradingSignal(
                        timestamp=current_time, direction=1,
                        entry_price=entry, stop_loss=sl,
                        take_profit_1=tp['tp1'], take_profit_2=tp['tp2'],
                        rr_ratio=rr, signal_type='elliott_wave',
                        confidence=wave.confidence,
                        reason='5浪+ABC完成 w3/w1=' + str(round(wave.fib_ratios.get('w3/w1',0),1))
                    ))
            elif wave.direction == -1 and trend_dir <= 0:
                entry = current_price
                sl = abc['points'][-1].price + current_atr * 1.0
                tp = calculate_take_profit_levels(entry, sl, -1)
                rr = calculate_rr_ratio(entry, sl, tp['tp1'], -1)
                if rr >= 1.2:
                    signals.append(TradingSignal(
                        timestamp=current_time, direction=-1,
                        entry_price=entry, stop_loss=sl,
                        take_profit_1=tp['tp1'], take_profit_2=tp['tp2'],
                        rr_ratio=rr, signal_type='elliott_wave',
                        confidence=wave.confidence,
                        reason='5浪+ABC完成 w3/w1=' + str(round(wave.fib_ratios.get('w3/w1',0),1))
                    ))

    return signals
