"""
量化交易策略 — 基于老师交易体系的完整策略

策略流程（对应老师体系）:
  1. 多周期趋势判断 → 定方向 (只做顺势)
  2. FVG/OB/流动性猎杀 → 找进场
  3. 盈亏比 >= 1.5 → 过滤
  4. 1:1减仓50%, 1:2再减50% → 持仓管理
  5. 止损跟随(破结构位才移止损) → 趋势跟踪
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from quant_core import (
    SwingPoint, FVG, OrderBlock, LiquiditySweep, TradingSignal,
    find_swing_points, detect_trend, detect_fvg, check_fvg_mitigation,
    detect_order_blocks, detect_liquidity_sweeps, find_nearest_fvg,
    calculate_take_profit_levels, calculate_rr_ratio, calculate_atr,
    multi_timeframe_alignment, generate_signals, get_sentiment_bonus
)


# ============================================================================
# 策略配置
# ============================================================================

@dataclass
class StrategyConfig:
    """策略参数"""
    # 多周期
    primary_tf: str = '4h'       # 主交易周期
    higher_tf: str = '1d'        # 大周期（定方向）

    # 波段点检测
    swing_window: int = 3        # 左右各3根K (4H=24h) 确认波段点

    # 风险
    risk_percent: float = 2.0    # 每笔最大亏损 2%
    min_rr_ratio: float = 0.8    # 最低盈亏比 (FVG止损有优势)

    # FVG
    fvg_max_age: int = 50        # FVG最多保留多少根K

    # 震荡过滤
    consolidation_threshold: float = 0.7  # >0.7 认为震荡，不开单

    # 老师说的：震荡越久越危险，不参与
    max_consolidation_bars: int = 30


# ============================================================================
# 仓位状态
# ============================================================================

@dataclass
class Position:
    """持仓状态"""
    direction: int         # 1=多, -1=空
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float
    take_profit_1: float   # 1:1
    take_profit_2: float   # 1:2
    initial_size: float    # 初始仓位
    remaining_size: float  # 剩余仓位
    tp1_hit: bool = False  # 1:1是否到达
    tp2_hit: bool = False  # 1:2是否到达
    signal_type: str = ''


@dataclass
class Trade:
    """已完成交易记录"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str


# ============================================================================
# 主策略类
# ============================================================================

class TrendFVGStrategy:
    """
    趋势+FVG策略

    对应老师体系：
    - 顺势交易 (大周期定方向)
    - FVG/真空区回踩入场
    - 流动性猎杀辅助确认
    - 1:1/1:2分批止盈
    """

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self._signal_queue: deque = deque(maxlen=100)

    def update(self, df_4h: pd.DataFrame, df_1d: pd.DataFrame,
               balance: float) -> Dict:
        """
        每个K线周期调用一次

        Args:
            df_4h: 4H数据
            df_1d: 日线数据
            balance: 当前账户余额

        Returns:
            交易决策字典
        """
        # 1. 处理已有持仓
        if self.position is not None:
            result = self._manage_position(df_4h, balance)
            # 减仓或继续持仓 → 不产生新信号(同一根K不重复开仓)
            if result.get('action') in ('reduce', 'hold'):
                return result
            # 止损/全平 → 仓位已清，继续往下检查新信号
            # (position 已被 _close_position 置为 None)

        # 2. 大周期趋势判断
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        sh_1d, sl_1d = find_swing_points(high_1d, low_1d, self.config.swing_window)
        trend_1d = detect_trend(sh_1d, sl_1d)

        # 3. 主周期分析
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        open_4h = df_4h['open'].values
        close_4h = df_4h['close'].values

        sh_4h, sl_4h = find_swing_points(high_4h, low_4h, self.config.swing_window)
        trend_4h = detect_trend(sh_4h, sl_4h)

        # 4. 多周期一致性 — 老师："日线定方向，周线看大局"
        alignment = multi_timeframe_alignment(trend_4h, trend_1d)
        trend_bonus = 0.0

        # 日线共振
        if alignment != 0:
            if trend_4h['direction'] == alignment:
                trend_bonus = 0.10
        elif trend_4h['direction'] != 0 and trend_1d['direction'] != 0:
            trend_bonus = -0.20  # 日线反向, 降信度

        # 周线趋势 (从日线降采样)
        if len(df_1d) >= 50:
            df_1w = df_1d.resample('1W').agg({'high': 'max', 'low': 'min',
                                               'open': 'first', 'close': 'last'}).dropna()
            if len(df_1w) >= 10:
                sh_w, sl_w = find_swing_points(df_1w['high'].values, df_1w['low'].values,
                                               window=self.config.swing_window)
                trend_w = detect_trend(sh_w, sl_w)
                if trend_w['direction'] == trend_4h['direction'] != 0:
                    trend_bonus += 0.05  # 周线同向, 再加一点

        # 5. 检测 FVG / OB / 流动性
        fvgs = detect_fvg(open_4h, high_4h, low_4h, close_4h)
        current_idx = len(close_4h) - 1
        fvgs = check_fvg_mitigation(fvgs, high_4h, low_4h, current_idx)
        # 清理过期FVG
        fvgs = [f for f in fvgs
                if current_idx - f.start_idx < self.config.fvg_max_age]

        obs = detect_order_blocks(open_4h, high_4h, low_4h, close_4h, sh_4h, sl_4h)
        sweeps = detect_liquidity_sweeps(high_4h, low_4h, open_4h, close_4h,
                                         sh_4h, sl_4h)

        # 6. 震荡过滤 — 老师："震荡行情不做"
        consol = self._check_consolidation(high_4h, low_4h, close_4h)
        if consol > self.config.consolidation_threshold:
            return {'action': 'hold', 'reason': f'震荡行情(score={consol:.2f})，等待突破'}

        # 7. 生成信号
        signals = generate_signals(df_4h, fvgs, obs, sweeps,
                                   sh_4h, sl_4h, trend_4h, current_idx)
        # 应用多周期共振加分 + 情绪修正
        sentiment_bonus = 0.0
        try:
            sentiment_bonus = get_sentiment_bonus('BTC')
        except Exception:
            pass

        for s in signals:
            # 多头: 空头过热+分   空头: 多头过热+分
            if s.direction == 1:
                final_bonus = trend_bonus + sentiment_bonus
            else:
                final_bonus = trend_bonus - sentiment_bonus
            s.confidence = min(1.0, max(0.1, s.confidence + final_bonus))

        # 8. 信号过滤
        best_signal = self._filter_best_signal(signals)

        if best_signal and best_signal.confidence >= 0.5:
            # 检查盈亏比
            if best_signal.rr_ratio < self.config.min_rr_ratio:
                return {'action': 'hold',
                        'reason': f'盈亏比{best_signal.rr_ratio:.1f}<{self.config.min_rr_ratio}'}

            # 开仓
            size = self._calc_size(balance, best_signal)
            if size <= 0:
                return {'action': 'hold', 'reason': '仓位太小，跳过'}

            self.position = Position(
                direction=best_signal.direction,
                entry_price=best_signal.entry_price,
                entry_time=best_signal.timestamp,
                stop_loss=best_signal.stop_loss,
                take_profit_1=best_signal.take_profit_1,
                take_profit_2=best_signal.take_profit_2,
                initial_size=size,
                remaining_size=size,
                signal_type=best_signal.signal_type,
            )

            return {
                'action': 'enter',
                'direction': 'long' if best_signal.direction == 1 else 'short',
                'entry_price': best_signal.entry_price,
                'stop_loss': best_signal.stop_loss,
                'tp1': best_signal.take_profit_1,
                'tp2': best_signal.take_profit_2,
                'size': size,
                'rr': best_signal.rr_ratio,
                'confidence': best_signal.confidence,
                'reason': best_signal.reason
            }

        return {'action': 'hold', 'reason': '无符合条件的信号'}

    def _manage_position(self, df: pd.DataFrame, balance: float) -> Dict:
        """
        持仓管理 — 老师的规则:
        1. 到1:1 → 减仓50%
        2. 到1:2 → 再减仓50%(剩25%)
        3. 剩下的推保本止损
        4. 破了结构位(前低/前高) → 全平
        """
        pos = self.position
        current_price = float(df['close'].iloc[-1])
        current_idx = len(df) - 1
        high = df['high'].values
        low = df['low'].values

        # === 止损检查 ===
        if pos.direction == 1:
            if low[-1] <= pos.stop_loss:
                return self._close_position(current_price, df.index[-1],
                                            '止损', balance)
        else:
            if high[-1] >= pos.stop_loss:
                return self._close_position(current_price, df.index[-1],
                                            '止损', balance)

        # === 1:1 减仓 ===
        if not pos.tp1_hit:
            if pos.direction == 1 and high[-1] >= pos.take_profit_1:
                # 减仓50%
                exit_size = pos.remaining_size * 0.5
                pnl = exit_size * (pos.take_profit_1 - pos.entry_price)
                pos.remaining_size -= exit_size
                pos.tp1_hit = True
                # 推保本止损
                pos.stop_loss = pos.entry_price

                self.trades.append(Trade(
                    entry_time=pos.entry_time,
                    exit_time=df.index[-1],
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=pos.take_profit_1,
                    size=exit_size,
                    pnl=pnl,
                    pnl_pct=pnl/balance,
                    exit_reason='TP1(1:1)减仓50%'
                ))
                return {'action': 'reduce', 'reason': 'TP1到达，减仓50%，止损推保本',
                        'pnl': pnl}

            elif pos.direction == -1 and low[-1] <= pos.take_profit_1:
                exit_size = pos.remaining_size * 0.5
                pnl = exit_size * (pos.entry_price - pos.take_profit_1)
                pos.remaining_size -= exit_size
                pos.tp1_hit = True
                pos.stop_loss = pos.entry_price

                self.trades.append(Trade(
                    entry_time=pos.entry_time,
                    exit_time=df.index[-1],
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=pos.take_profit_1,
                    size=exit_size,
                    pnl=pnl,
                    pnl_pct=pnl/balance,
                    exit_reason='TP1(1:1)减仓50%'
                ))
                return {'action': 'reduce', 'reason': 'TP1到达，减仓50%',
                        'pnl': pnl}

        # === 1:2 减仓 ===
        if pos.tp1_hit and not pos.tp2_hit:
            if pos.direction == 1 and high[-1] >= pos.take_profit_2:
                exit_size = pos.remaining_size * 0.5  # 剩余的一半
                pnl = exit_size * (pos.take_profit_2 - pos.entry_price)
                pos.remaining_size -= exit_size
                pos.tp2_hit = True

                self.trades.append(Trade(
                    entry_time=pos.entry_time,
                    exit_time=df.index[-1],
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=pos.take_profit_2,
                    size=exit_size,
                    pnl=pnl,
                    pnl_pct=pnl/balance,
                    exit_reason='TP2(1:2)减仓50%'
                ))
                return {'action': 'reduce', 'reason': 'TP2到达，再减仓50%',
                        'pnl': pnl}

            elif pos.direction == -1 and low[-1] <= pos.take_profit_2:
                exit_size = pos.remaining_size * 0.5
                pnl = exit_size * (pos.entry_price - pos.take_profit_2)
                pos.remaining_size -= exit_size
                pos.tp2_hit = True

                self.trades.append(Trade(
                    entry_time=pos.entry_time,
                    exit_time=df.index[-1],
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=pos.take_profit_2,
                    size=exit_size,
                    pnl=pnl,
                    pnl_pct=pnl/balance,
                    exit_reason='TP2(1:2)减仓50%'
                ))
                return {'action': 'reduce', 'reason': 'TP2到达，再减仓50%',
                        'pnl': pnl}

        # === 跟随止损 — 老师：等新结构点出现再移止损 ===
        # 老师："TP1后推保本，行情走出新低/新高后止损跟到结构位"
        # 实现：TP1到达后 → 止损推保本
        #       TP2到达后 → 找最近的波段低点(多)/高点(空) 做止损
        if pos.tp1_hit:
            sh, sl = find_swing_points(high, low, window=self.config.swing_window)
            atr_val = calculate_atr(high, low, df['close'].values)[-1]

            if pos.direction == 1:
                # 多头：找当前价下方最近的波段低点 → 止损移到低点下方
                recent_lows = sorted([l for l in sl if l.index >= current_idx - 30],
                                     key=lambda x: x.price, reverse=True)
                if recent_lows:
                    struct_sl = recent_lows[0].price - atr_val * 0.5
                    if struct_sl > pos.stop_loss:
                        pos.stop_loss = struct_sl
                elif pos.stop_loss < pos.entry_price:
                    pass  # 已经保本了, 维持
            else:
                # 空头：找当前价上方最近的波段高点 → 止损移到高点上方
                recent_highs = sorted([h for h in sh if h.index >= current_idx - 30],
                                      key=lambda x: x.price)
                if recent_highs:
                    struct_sl = recent_highs[0].price + atr_val * 0.5
                    if struct_sl < pos.stop_loss:
                        pos.stop_loss = struct_sl
                elif pos.stop_loss > pos.entry_price:
                    pass  # 已经保本了, 维持

        return {'action': 'hold', 'reason': '持仓中'}

    def _close_position(self, exit_price: float, exit_time,
                        reason: str, balance: float) -> Dict:
        """平掉剩余仓位"""
        pos = self.position
        if pos.direction == 1:
            pnl = pos.remaining_size * (exit_price - pos.entry_price)
        else:
            pnl = pos.remaining_size * (pos.entry_price - exit_price)

        self.trades.append(Trade(
            entry_time=pos.entry_time,
            exit_time=exit_time,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.remaining_size,
            pnl=pnl,
            pnl_pct=pnl/balance if balance > 0 else 0,
            exit_reason=reason
        ))

        self.position = None
        return {'action': 'close', 'reason': reason, 'pnl': pnl}

    def _filter_best_signal(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """选择置信度最高且符合条件的信号"""
        if not signals:
            return None
        valid = [s for s in signals if s.rr_ratio >= self.config.min_rr_ratio]
        if not valid:
            return None
        return max(valid, key=lambda s: s.confidence)

    def _calc_size(self, balance: float, signal: TradingSignal) -> float:
        """计算仓位大小，带仓位上限保护"""
        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return 0
        risk_amount = balance * (self.config.risk_percent / 100)
        size = risk_amount / risk
        # 仓位上限: 不超过账户余额 (1x杠杆), 防止极端滑点爆仓
        max_size = balance / signal.entry_price
        return min(size, max_size)

    def _check_consolidation(self, high, low, close) -> float:
        """检查是否在震荡区间"""
        window = 20
        n = len(close)
        if n < window:
            return 0
        recent_high = max(high[n-window:n])
        recent_low = min(low[n-window:n])
        avg = np.mean(close[n-window:n])
        if avg == 0:
            return 0
        range_pct = (recent_high - recent_low) / avg
        # 小于5%的波动 = 窄幅震荡
        if range_pct < 0.05:
            return 0.9
        elif range_pct < 0.10:
            return 0.5
        return 0.0

    def get_stats(self) -> Dict:
        """策略统计"""
        if not self.trades:
            return {'total_trades': 0}

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            'total_trades': len(self.trades),
            'win_rate': len(wins) / len(pnls) * 100 if pnls else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_pnl': sum(pnls),
            'profit_factor': abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            'max_consecutive_losses': self._max_consecutive(pnls, 'loss'),
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
        }

    @staticmethod
    def _max_consecutive(pnls: List[float], mode: str) -> int:
        max_c, cur = 0, 0
        for p in pnls:
            if mode == 'loss' and p <= 0:
                cur += 1
                max_c = max(max_c, cur)
            elif mode == 'win' and p > 0:
                cur += 1
                max_c = max(max_c, cur)
            else:
                cur = 0
        return max_c
