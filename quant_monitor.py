"""
Position tracker + signal monitor
  python quant_monitor.py                  # check positions
  python quant_monitor.py --rescan         # interactive scan & open
  python quant_monitor.py --auto           # auto-execute all signals
  python quant_monitor.py --close SYM      # close position
"""
import sys, os, time, json, argparse, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_scanner import _to_okx_id, scan_all, PROXY
from quant_executor import OKXExecutor

_proxies = {'http': PROXY, 'https': PROXY} if PROXY else None
POS = os.path.join(os.path.dirname(__file__), 'positions.json')


def load():
    if os.path.exists(POS):
        with open(POS) as f:
            return json.load(f)
    return []


def save(data):
    with open(POS, 'w') as f:
        json.dump(data, f, indent=2)


def price(inst_id):
    try:
        r = requests.get('https://www.okx.com/api/v5/market/ticker',
                         params={'instId': inst_id}, proxies=_proxies, timeout=10)
        d = r.json()
        return float(d['data'][0]['last']) if d['code'] == '0' else None
    except:
        return None


def enter(sym, dr, entry, stop, tp1, tp2, sig, conf):
    data = load()
    for p in data:
        if p['symbol'] == sym and p['status'] == 'open':
            print(f'{sym} already open')
            return
    data.append(dict(symbol=sym, direction=dr, entry=entry,
                     stop=stop, tp1=tp1, tp2=tp2,
                     signal=sig, confidence=conf,
                     at=time.strftime('%Y-%m-%d %H:%M'), status='open'))
    save(data)
    print(f'[OPEN] {sym} {dr} @ {entry}')


def check():
    data = load()
    active = [p for p in data if p.get('status') == 'open']
    if not active:
        print('No open positions')
        return

    print(f'\n--- Positions ({len(active)} open) ---')
    print(f'{"Sym":8s} {"Dir":5s} {"Entry":>10s} {"Now":>10s} {"PnL%":>8s} {"Stop":>10s} {"TP1":>10s} {"Conf":>5s} Status')
    print('-' * 90)

    for p in data:
        if p['status'] != 'open':
            continue
        inst = _to_okx_id(p['symbol'])
        now = price(inst)
        if now is None:
            print(f'  {p["symbol"]:8s} data error')
            continue

        d = p['direction']
        e = p['entry']
        s = p['stop']
        t1 = p['tp1']
        t2 = p['tp2']
        cf = p['confidence']

        if d == 'SHORT':
            pnl = (e - now) / e * 100
            stopped = now >= s
            tp2_hit = now <= t2
            tp1_hit = now <= t1
        else:
            pnl = (now - e) / e * 100
            stopped = now <= s
            tp2_hit = now >= t2
            tp1_hit = now >= t1

        if stopped:
            st = 'STOPPED'
            p['status'] = 'closed'
        elif tp2_hit:
            st = 'TP2'
            p['status'] = 'closed'
        elif tp1_hit:
            st = 'TP1 (scale 50%%, SL->BE)'
            p['stop'] = e
        else:
            st = 'HOLD'

        print(f'{p["symbol"]:8s} {d:5s} {e:>10.4f} {now:>10.4f} {pnl:>+7.2f}%% {s:>10.4f} {t1:>10.4f} {cf:>4.1f}  {st}')

    save(data)


def scan_and_open(top_n=20):
    print('Scanning...')
    results = scan_all(top_n=top_n)
    if not results:
        print('No signals')
        return

    print(f'\n{len(results)} signals found:')
    for r in results:
        sym = r['symbol'].replace('/USDT:USDT', '')
        d = r['direction']
        st = r['signal_type']
        e = r['entry']; s = r['stop']; t = r['tp1']; c = r['confidence']
        print(f'  {sym:<8s} {d:5s} {st:12s} entry={e:.4f} stop={s:.4f} tp1={t:.4f} conf={c:.2f}')

    for r in results:
        sym = r['symbol']
        dr = r['direction']
        prompt = '\nOpen ' + sym + ' ' + dr + '? (y/n): '
        ans = input(prompt).strip().lower()
        if ans == 'y':
            enter(sym, d, r['entry'], r['stop'], r['tp1'], r['tp2'],
                  r['signal_type'], r['confidence'])


def auto_execute(top_n=20, risk_pct=2.0):
    """Auto scan + place real orders (no confirmation)"""
    results = scan_all(top_n=top_n)
    if not results:
        print('No signals')
        return

    ex = OKXExecutor()
    bal = ex.get_balance()
    print(f'Balance: ${bal:,.0f} | Signals: {len(results)} | Risk: {risk_pct}%/trade')

    for r in results:
        sym = r['symbol']
        inst = _to_okx_id(sym)
        dr = r['direction']
        entry = r['entry']
        stop = r['stop']

        risk_amt = bal * (risk_pct / 100)
        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            print(f'  {sym}: stop=0 skip')
            continue

        ct_val = ex.get_contract_size(inst)
        sz = max(1, int(risk_amt / stop_dist / ct_val))

        tp1 = r['tp1']; tp2 = r['tp2']
        print(f'  {sym} {dr} entry~{entry:.4f} stop={stop:.4f} tp1={tp1:.4f} tp2={tp2:.4f} risk=${risk_amt:.0f} sz={sz}')
        ex.market_order(inst, dr.lower(), sz, tp_price=tp2, sl_price=stop)
        enter(sym.replace('/USDT:USDT', ''), dr, entry, stop, tp1, tp2,
              r['signal_type'], r['confidence'])

    print('\nPositions:')
    check()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rescan', action='store_true')
    ap.add_argument('--auto', action='store_true', help='Auto-execute all signals')
    ap.add_argument('--open', type=str)
    ap.add_argument('--close', type=str)
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--risk', type=float, default=2.0, help='Risk % per trade')
    a = ap.parse_args()

    if a.auto:
        auto_execute(a.top, a.risk)
    elif a.rescan:
        scan_and_open(a.top)
    elif a.open:
        parts = a.open.split(',')
        if len(parts) >= 6:
            enter(parts[0].strip(), parts[1].strip().upper(),
                  float(parts[2]), float(parts[3]),
                  float(parts[4]), float(parts[5]),
                  'manual', float(parts[6]) if len(parts) > 6 else 0.7)
    elif a.close:
        data = load()
        for p in data:
            if p['symbol'].upper() == a.close.upper() and p['status'] == 'open':
                inst = _to_okx_id(p['symbol'])
                now = price(inst)
                if now:
                    if p['direction'] == 'SHORT':
                        pnl = (p['entry'] - now) / p['entry'] * 100
                    else:
                        pnl = (now - p['entry']) / p['entry'] * 100
                    p['status'] = 'closed'
                    print(f'[CLOSE] {p["symbol"]} PnL: {pnl:+.2f}%')
                else:
                    p['status'] = 'closed'
                save(data)
                break
    else:
        check()
        print('\n--rescan | --open SYM,DIR,ENTRY,STOP,TP1,TP2,CONF | --close SYM')
