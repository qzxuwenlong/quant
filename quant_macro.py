"""
宏观数据模块 — DXY/SPX/VIX/GOLD
  DXY (美元指数): 涨→风险资产承压, 跌→BTC利好
  SPX (标普500):  涨→风险偏好, 跌→避险
  GLD (黄金):     涨→避险情绪

用法:
  from quant_macro import get_macro_bias
  bias = get_macro_bias()  # 返回 -1(偏空) / 0(中性) / 1(偏多)
"""
import os, yfinance as yf
import numpy as np

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 缓存
_CACHE = {}
_CACHE_TIME = 0


def _trend(prices, short=5, long=20):
    """判断短期趋势: 短均>长均=上升"""
    if len(prices) < long:
        return 0
    s = np.mean(prices[-short:])
    l = np.mean(prices[-long:])
    return 1 if s > l else (-1 if s < l else 0)


def _fetch(ticker, period='1mo'):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        if len(df) < 5:
            return None
        return df
    except:
        return None


def get_macro_bias():
    """
    返回宏观偏向:
      1 = 利好BTC (DXY跌+SPX涨)
      0 = 中性
     -1 = 利空BTC (DXY涨+SPX跌)

    老师: "美元一跌几乎所有资产都涨, 美股涨币也涨"
    """
    import time
    global _CACHE, _CACHE_TIME
    if time.time() - _CACHE_TIME < 3600 and _CACHE:
        return _CACHE

    score = 0
    details = {}

    # DXY: 涨→BTC空, 跌→BTC多
    dxy = _fetch('DX-Y.NYB', '1mo')
    if dxy is not None:
        t = _trend(dxy['Close'].values)
        if t == 1:
            score -= 1
        elif t == -1:
            score += 1
        details['dxy'] = {'trend': t, 'price': float(dxy['Close'].iloc[-1])}

    # SPX: 涨→risk-on→BTC多, 跌→risk-off→BTC空
    spx = _fetch('^GSPC', '1mo')
    if spx is not None:
        t = _trend(spx['Close'].values)
        if t == 1:
            score += 1
        elif t == -1:
            score -= 1
        details['spx'] = {'trend': t, 'price': float(spx['Close'].iloc[-1])}

    # Gold: 涨→避险→对BTC复杂 (但通常短期同向)
    gold = _fetch('GC=F', '1mo')
    if gold is not None:
        t = _trend(gold['Close'].values)
        details['gold'] = {'trend': t, 'price': float(gold['Close'].iloc[-1])}

    bias = 1 if score > 0 else (-1 if score < 0 else 0)
    details['score'] = score
    details['bias'] = bias

    _CACHE = {'bias': bias, 'score': score, 'details': details}
    _CACHE_TIME = time.time()
    return _CACHE


def get_macro_bonus():
    """
    宏观信度修正 (弱影响):
      仅当 DXY+SPX 同时指向同一方向时才有效
      单边信号不加不减 (避免随机砍信号)
    """
    d = get_macro_bias()
    score = d['score']  # -2 ~ +2
    details = d['details']

    dxy_t = details.get('dxy', {}).get('trend', 0)
    spx_t = details.get('spx', {}).get('trend', 0)
    dxy_p = details.get('dxy', {}).get('price', 0)
    spx_p = details.get('spx', {}).get('price', 0)

    # 只有 DXY 和 SPX 同向时才有意义 (score=±2)
    if abs(score) >= 2:
        bonus = (1 if score > 0 else -1) * 0.05  # 弱修正
        tag = ['▼▼', '──', '▲▲'][(score//2)+1]
    else:
        bonus = 0.0  # 不一致时不加不减
        tag = ['▼─', '──', '─▲'][score+1]

    print(f'[MACRO] {tag} DXY={dxy_p:.0f}({dxy_t:+d}) SPX={spx_p:.0f}({spx_t:+d}) bonus={bonus:+.2f}')
    return bonus, score
