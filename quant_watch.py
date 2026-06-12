"""
Quick market watch — one-shot status check
  python quant_watch.py
"""
import sys, os, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_scanner import scan_all, PROXY, get_ranked_symbols
from quant_core import fetch_okx_sentiment

_proxies = {'http': PROXY, 'https': PROXY} if PROXY else None


def check(coins):
    results = scan_all(top_n=coins)
    proxy = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}

    # Sentiment
    btc = fetch_okx_sentiment('BTC', proxies=proxy)
    tag = '!!' if btc['extreme'] else '  '
    print(f'{tag} BTC sentiment: {btc["latest"]:.2f} (7d={btc["avg_7d"]:.2f}) {btc["extreme"] or ""}')

    # Signals
    if not results:
        print('Signals: 0 (market quiet)')
    else:
        print(f'Signals: {len(results)}')
        for r in results:
            sym = r['symbol'].replace('/USDT:USDT', '')
            d = r['direction']; st = r['signal_type']
            e = r['entry']; s = r['stop']; t1 = r['tp1']; t2 = r['tp2']
            d4 = 'v' if r['trend_4h'] == -1 else ('^' if r['trend_4h'] == 1 else '-')
            d1 = 'v' if r['trend_1d'] == -1 else ('^' if r['trend_1d'] == 1 else '-')
            print(f'  {sym:<10s} {d:5s} {st:12s} E={e:.4f} SL={s:.4f} TP1={t1:.4f} TP2={t2:.4f} RR={r["rr"]:.1f} C={r["confidence"]:.2f} {d4}{d1}')

    # Check tracked positions
    pos_file = os.path.join(os.path.dirname(__file__), 'positions.json')
    if os.path.exists(pos_file):
        import json
        with open(pos_file) as f:
            pos = json.load(f)
        active = [p for p in pos if p.get('status') == 'open']
        if active:
            print(f'\nOpen positions: {len(active)}')
            for p in active:
                inst = p['symbol'].replace('/USDT:USDT', '').replace('/USDT', '')
                inst_id = f'{inst}-USDT-SWAP'
                try:
                    r = requests.get('https://www.okx.com/api/v5/market/ticker',
                                    params={'instId': inst_id}, proxies=_proxies, timeout=8)
                    d = r.json()
                    if d['code'] == '0':
                        last = float(d['data'][0]['last'])
                        if p['direction'] == 'SHORT':
                            pnl = (p['entry'] - last) / p['entry'] * 100
                        else:
                            pnl = (last - p['entry']) / p['entry'] * 100
                        print(f'  {inst:<10s} {p["direction"]:5s} entry={p["entry"]:.4f} now={last:.4f} PnL={pnl:+.2f}%')
                except:
                    print(f'  {inst:<10s} price fetch failed')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--coins', type=int, default=15)
    a = ap.parse_args()
    check(a.coins)
