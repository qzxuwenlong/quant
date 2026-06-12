"""
OKX 自动交易执行器

环境变量:
  OKX_API_KEY    — API Key
  OKX_SECRET     — Secret
  OKX_PASSWORD   — Passphrase

用法:
  from quant_executor import OKXExecutor
  ex = OKXExecutor()
  ex.market_order('OP-USDT-SWAP', 'short', size=10)   # 市价开空
  ex.close_position('OP-USDT-SWAP')                     # 平仓
"""
import os, time, base64, hmac, hashlib, json, requests

# 从 .env 加载 (优先级: 环境变量 > .env 文件)
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()

PROXY_HOST = os.environ.get('OKX_PROXY', 'http://127.0.0.1:7890')
_proxies = {'http': PROXY_HOST, 'https': PROXY_HOST} if PROXY_HOST else None
OKX_API = 'https://www.okx.com'


class OKXExecutor:
    def __init__(self, api_key=None, secret=None, password=None, simulated=None):
        self.api_key = api_key or os.environ.get('OKX_API_KEY', '')
        self.secret = secret or os.environ.get('OKX_API_SECRET', '')
        self.password = password or os.environ.get('OKX_PASSPHRASE', '')
        if simulated is None:
            simulated = os.environ.get('OKX_SIMULATED', '1') == '1'
        self.sim = simulated or not (self.api_key and self.secret and self.password)
        if self.sim:
            print('[OKX] SIMULATION mode')
        else:
            print('[OKX] LIVE mode')

    def _request(self, method, path, body=None, params=None):
        if self.sim:
            return {'code': '0', 'msg': 'SIMULATION', 'data': [{'ordId': 'sim_000'}]}

        import urllib.parse
        from datetime import datetime, timezone

        query = ('?' + urllib.parse.urlencode(params)) if params else ''
        full_path = path + query
        body_str = json.dumps(body, separators=(',', ':')) if body else ''

        # OKX 支持 ISO 8601 格式 (兼容性最好)
        timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

        sign_text = timestamp + method + full_path + body_str
        mac = hmac.new(self.secret.encode('utf-8'), sign_text.encode('utf-8'), hashlib.sha256)
        signature = base64.b64encode(mac.digest()).decode('ascii')

        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.password,
            'Content-Type': 'application/json',
        }
        url = OKX_API + full_path
        # 重试3次 (签名偶发失效通常是时间戳偏差)
        for attempt in range(3):
            if attempt > 0:
                # 重新生成时间戳和签名
                timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
                sign_text = timestamp + method + full_path + body_str
                mac = hmac.new(self.secret.encode('utf-8'), sign_text.encode('utf-8'), hashlib.sha256)
                headers['OK-ACCESS-SIGN'] = base64.b64encode(mac.digest()).decode('ascii')
                headers['OK-ACCESS-TIMESTAMP'] = timestamp
                time.sleep(0.3)

            try:
                if method == 'GET':
                    r = requests.get(url, headers=headers, proxies=_proxies, timeout=10)
                else:
                    r = requests.post(url, headers=headers, data=body_str, proxies=_proxies, timeout=10)
                result = r.json()
                if result.get('code') not in ('50113', '50112'):
                    return result
            except Exception:
                if attempt == 2:
                    raise
        return {'code': '-1', 'msg': 'Max retries exceeded'}

    def market_order(self, inst_id, direction, size, tp_price=None, sl_price=None):
        """
        市价单 + 自动挂止盈止损
        OKX attachAlgoOrds: [{'tpTriggerPx':..., 'tpOrdPx':'-1', 'slTriggerPx':..., 'slOrdPx':'-1'}]
        -1 = 市价成交
        """
        side = 'buy' if direction == 'long' else 'sell'

        body = {
            'instId': inst_id,
            'tdMode': 'cross',
            'side': side,
            'ordType': 'market',
            'sz': str(size),
        }

        if tp_price or sl_price:
            algo = {}
            if tp_price:
                algo['tpTriggerPx'] = str(tp_price)
                algo['tpOrdPx'] = '-1'
            if sl_price:
                algo['slTriggerPx'] = str(sl_price)
                algo['slOrdPx'] = '-1'
            body['attachAlgoOrds'] = [algo]

        result = self._request('POST', '/api/v5/trade/order', body)
        if result['code'] == '0':
            ord_id = result['data'][0]['ordId']
            tp_s = f' TP@{tp_price}' if tp_price else ''
            sl_s = f' SL@{sl_price}' if sl_price else ''
            print(f'[ORDER] {direction.upper()} {inst_id} sz={size}{tp_s}{sl_s} id={ord_id}')
            return ord_id
        else:
            print(f'[ERR] Order failed: {result}')
            return None

    def close_position(self, inst_id, pos_size=None, pos_side=None):
        """市价全平 (sim模式需手动传pos_size)"""
        if self.sim:
            if pos_size and pos_side:
                close_side = 'sell' if pos_side == 'long' else 'buy'
                body = {'instId': inst_id, 'tdMode': 'cross', 'side': close_side,
                        'ordType': 'market', 'sz': str(int(pos_size))}
                r = self._request('POST', '/api/v5/trade/order', body)
                print(f'[CLOSE] {inst_id} sz={pos_size} (sim)')
                return True
            print('[SIM] No position data in simulation')
            return None

        pos = self._request('GET', f'/api/v5/account/positions?instId={inst_id}')
        if pos['code'] != '0' or not pos['data']:
            print(f'[WARN] No position for {inst_id}')
            return None

        pos_data = pos['data'][0]
        pos_qty = abs(float(pos_data.get('pos', 0) or 0))
        if pos_qty == 0:
            return None

        pos_side = 'long' if float(pos_data['pos']) > 0 else 'short'
        close_side = 'sell' if pos_side == 'long' else 'buy'

        body = {
            'instId': inst_id,
            'tdMode': 'cross',
            'side': close_side,
            'ordType': 'market',
            'sz': str(int(pos_qty))
        }
        result = self._request('POST', '/api/v5/trade/order', body)
        if result['code'] == '0':
            print(f'[CLOSE] {inst_id} sz={pos_qty}')
            return True
        print(f'[ERR] Close failed: {result}')
        return False

    def get_balance(self):
        """查 USDT 余额"""
        r = self._request('GET', '/api/v5/account/balance')
        if r['code'] == '0':
            for item in r['data'][0].get('details', []):
                if item['ccy'] == 'USDT':
                    return float(item['availBal'])
        return 0

    def get_contract_size(self, inst_id):
        """获取合约面值"""
        r = self._request('GET', f'/api/v5/public/instruments?instType=SWAP&instId={inst_id}')
        if r['code'] == '0' and r['data']:
            return float(r['data'][0]['ctVal'])
        return 0.01  # default
