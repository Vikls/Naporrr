# trading/bybit_api_manager.py
import asyncio
import time
import hmac
import hashlib
import json
import aiohttp
from typing import Dict, Optional, Any, Tuple, List, Union
from config.settings import settings
from utils.logger import logger

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

class BybitAPIManager:
    """Bybit Unified Trading HTTP wrapper with aiohttp and IMPROVED MONITORING"""

    def __init__(self):
        self.tcfg = settings.trading
        self.api_cfg = settings.api
        self.secrets = settings.secrets
        mode = self.tcfg.mode.lower()
        self.demo = (mode == "demo")

        # Choose keys
        if self.demo and self.secrets.demo_bybit_api_key and self.secrets.demo_bybit_api_secret:
            api_key = self.secrets.demo_bybit_api_key
            api_secret = self.secrets.demo_bybit_api_secret
        elif not self.demo and self.secrets.live_bybit_api_key and self.secrets.live_bybit_api_secret:
            api_key = self.secrets.live_bybit_api_key
            api_secret = self.secrets.live_bybit_api_secret
        else:
            api_key = self.secrets.bybit_api_key
            api_secret = self.secrets.bybit_api_secret

        if not (api_key and api_secret):
            logger.error("[API] ❌ Missing API keys!")
            raise Exception("BYBIT API keys are required")

        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api-demo.bybit.com" if self.demo else "https://api.bybit.com"
        self.retry_attempts = self.api_cfg.retry_attempts
        self.retry_delay = self.api_cfg.retry_delay

        # Caches
        self._inst_cache: Dict[str, Dict[str, Any]] = {}
        self._inst_expire: Dict[str, float] = {}
        self._ticker_cache: Dict[str, Dict[str, Any]] = {}
        
        # ПОКРАЩЕННЯ: Кеш для моніторингу позицій
        self._position_cache: Dict[str, Dict] = {}
        self._position_cache_ttl = 1.0  # 1 секунда TTL для позицій

        # aiohttp session
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(f"[API] ✅ Init - mode={mode.upper()} base={self.base_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("[API] Session closed")

    def _generate_signature(self, timestamp: str, recv_window: str, params_str: str) -> str:
        param_str = f"{timestamp}{self.api_key}{recv_window}{params_str}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _build_query_string_for_signature(self, params: Union[Dict[str, Any], List[Tuple[str, Any]]]) -> str:
        if not params:
            return ""
        if isinstance(params, list):
            parts = [f"{k}={v}" for (k, v) in params]
        else:
            parts = [f"{k}={params[k]}" for k in params]
        return "&".join(parts)

    def _build_json_string(self, params: Dict[str, Any]) -> str:
        if not params:
            return ""
        return json.dumps(params, separators=(',', ':'), sort_keys=True, ensure_ascii=True)

    def _build_headers(self, timestamp: str, signature: str, recv_window: str = "5000") -> Dict[str, str]:
        return {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-SIGN-TYPE': '2',
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': recv_window,
            'Content-Type': 'application/json'
        }

    def ws_auth_args(self) -> list:
        ts_ms = str(int(time.time() * 1000))
        pre_sign = ts_ms + self.api_key
        sign = hmac.new(
            self.api_secret.encode("utf-8"),
            pre_sign.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return [self.api_key, ts_ms, sign]

    async def _backoff(self, attempt: int, base: float):
        delay = min(10.0, base * (2 ** attempt))
        await asyncio.sleep(delay)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Union[Dict[str, Any], List[Tuple[str, Any]]]] = None,
        signed: bool = True
    ) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        last_exc: Optional[Exception] = None
        recv_window = "5000"

        for attempt in range(self.retry_attempts):
            try:
                timestamp = str(int(time.time() * 1000))

                if signed:
                    if method.upper() == "GET":
                        params_str = self._build_query_string_for_signature(params) if params else ""
                    else:
                        params_str = self._build_json_string(params) if isinstance(params, dict) and params else ""
                    signature = self._generate_signature(timestamp, recv_window, params_str)
                    headers = self._build_headers(timestamp, signature, recv_window)
                else:
                    headers = {'Content-Type': 'application/json'}

                if method.upper() == "GET":
                    async with session.get(url, params=params, headers=headers) as resp:
                        result = await resp.json(content_type=None)
                elif method.upper() == "POST":
                    if params:
                        json_data = self._build_json_string(params) if isinstance(params, dict) else "{}"
                        async with session.post(url, data=json_data, headers=headers) as resp:
                            result = await resp.json(content_type=None)
                    else:
                        async with session.post(url, headers=headers) as resp:
                            result = await resp.json(content_type=None)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if isinstance(result, dict):
                    ret_code = result.get('retCode', -1)

                    if ret_code == 0:
                        return result

                    if ret_code == 10002:
                        logger.warning(f"[API] Rate limit attempt {attempt+1}")
                        await self._backoff(attempt, self.retry_delay)
                        last_exc = Exception(result.get('retMsg', 'Rate limit'))
                        continue

                    if ret_code == 10003 or 'sign' in str(result.get('retMsg', '')).lower():
                        logger.error(f"[API] Sig error {endpoint}: {result.get('retMsg')}")
                        return result

                    if ret_code == 10001 and 'order not modified' in str(result.get('retMsg', '')).lower():
                        logger.info(f"[API] No-op: {result.get('retMsg')} (code: 10001)")
                        return result

                    logger.error(f"[API] Error: {result.get('retMsg')} (code: {ret_code})")
                    return result

                return result

            except asyncio.TimeoutError:
                last_exc = Exception("Timeout")
                logger.warning(f"[API] Timeout attempt {attempt+1}")
                await self._backoff(attempt, self.retry_delay)
            except Exception as e:
                last_exc = e
                logger.warning(f"[API] Error attempt {attempt+1}: {e}")
                await self._backoff(attempt, self.retry_delay)

        if last_exc:
            logger.error(f"[API] All retries failed: {last_exc}")
            raise last_exc
        return None

    async def check_time_sync(self) -> bool:
        try:
            result = await self._make_request("GET", "/v5/market/time", signed=False)
            if not result or result.get('retCode') != 0:
                return False

            server_time = int(result['result']['timeSecond'])
            local_time = int(time.time())
            diff = abs(server_time - local_time)

            if diff > settings.api.validate_time_diff_sec:
                logger.error(f"[API] Time diff: {diff}s")
                return False

            logger.info(f"[API] ✅ Time sync OK ({diff}s)")
            return True
        except Exception as e:
            logger.error(f"[API] Time sync error: {e}")
            return False

    async def get_instrument_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        ttl = settings.api.instrument_cache_ttl
        exp = self._inst_expire.get(symbol, 0)

        if symbol in self._inst_cache and exp > now:
            return self._inst_cache[symbol]

        try:
            result = await self._make_request(
                "GET",
                "/v5/market/instruments-info",
                params={"category": "linear", "symbol": symbol},
                signed=False
            )
            if result and result.get('retCode') == 0:
                lst = result.get('result', {}).get('list', [])
                if lst:
                    info = lst[0]
                    self._inst_cache[symbol] = info
                    self._inst_expire[symbol] = now + ttl
                    return info
            return None
        except Exception as e:
            logger.error(f"[API] get_instrument_info {symbol}: {e}")
            return None

    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        now = time.time()
        ttl = settings.api.ticker_cache_ttl
        entry = self._ticker_cache.get(symbol)

        if entry and entry['expires'] > now:
            return entry['price']

        try:
            result = await self._make_request(
                "GET",
                "/v5/market/tickers",
                params={"category": "linear", "symbol": symbol},
                signed=False
            )

            if result and result.get('retCode') == 0:
                lst = result.get('result', {}).get('list', [])
                if lst:
                    price = _safe_float(lst[0].get('lastPrice'))
                    self._ticker_cache[symbol] = {'price': price, 'expires': now + ttl}
                    return price
            return None
        except Exception as e:
            logger.error(f"[API] get_ticker_price {symbol}: {e}")
            return None

    async def get_wallet_balance(self) -> Optional[float]:
        try:
            logger.info("[API] 🔄 Getting balance...")
            params: List[Tuple[str, Any]] = [("accountType", "UNIFIED")]
            result = await self._make_request(
                "GET",
                "/v5/account/wallet-balance",
                params=params
            )
            if result and result.get("retCode") == 0:
                list_acc = result.get("result", {}).get("list", [])
                for acc in list_acc:
                    if acc.get('accountType') == 'UNIFIED':
                        total_margin_balance = _safe_float(acc.get('totalMarginBalance'))
                        total_equity = _safe_float(acc.get('totalEquity'))
                        logger.info(f"💰 Margin: {total_margin_balance:.2f} | Equity: {total_equity:.2f}")
                        if total_margin_balance > 0:
                            return total_margin_balance
                        elif total_equity > 0:
                            return total_equity
                return None
            return None
        except Exception as e:
            logger.error(f"[API] get_wallet_balance: {e}")
            return None

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: str,
        order_type: str,
        price: Optional[str] = None,
        position_idx: int = 0,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        try:
            params: Dict[str, Any] = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": qty,
                "positionIdx": position_idx
            }

            if order_type == "Limit":
                if not price:
                    return {"retCode": -1, "retMsg": "Price required"}
                params["price"] = price
                params["timeInForce"] = time_in_force

            logger.info(f"[API] Order: {symbol} {side} {qty} {order_type}" + (f" @ {price}" if price else ""))
            result = await self._make_request("POST", "/v5/order/create", params=params)
            return result or {"retCode": -1, "retMsg": "No response"}
        except Exception as e:
            logger.error(f"[API] place_order {symbol}: {e}")
            return {"retCode": -1, "retMsg": str(e)}

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        try:
            result = await self._make_request(
                "POST",
                "/v5/order/cancel",
                params={"category": "linear", "symbol": symbol, "orderId": order_id}
            )
            return result or {"retCode": -1, "retMsg": "No response"}
        except Exception as e:
            logger.error(f"[API] cancel_order {symbol}: {e}")
            return {"retCode": -1, "retMsg": str(e)}

    async def amend_price(self, symbol: str, order_id: str, new_price: str) -> Dict[str, Any]:
        try:
            result = await self._make_request(
                "POST",
                "/v5/order/amend",
                params={"category": "linear", "symbol": symbol, "orderId": order_id, "price": new_price}
            )
            return result or {"retCode": -1, "retMsg": "No response"}
        except Exception as e:
            logger.error(f"[API] amend_price {symbol}: {e}")
            return {"retCode": -1, "retMsg": str(e)}

    async def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        try:
            rt_params: List[Tuple[str, Any]] = [
                ("category", "linear"),
                ("orderId", order_id),
                ("symbol", symbol),
            ]
            result = await self._make_request(
                "GET",
                "/v5/order/realtime",
                params=rt_params
            )

            if result and result.get('retCode') == 0:
                lst = result.get('result', {}).get('list', [])
                if lst:
                    o = lst[0]
                    return {
                        "status": o.get("orderStatus"),
                        "cumExecQty": _safe_float(o.get("cumExecQty")),
                        "avgPrice": _safe_float(o.get("avgPrice")),
                        "leavesQty": _safe_float(o.get("leavesQty")),
                        "raw": o
                    }

            hist_params: List[Tuple[str, Any]] = [
                ("category", "linear"),
                ("orderId", order_id),
                ("symbol", symbol),
            ]
            result = await self._make_request(
                "GET",
                "/v5/order/history",
                params=hist_params
            )
            if result and result.get('retCode') == 0:
                lst = result.get('result', {}).get('list', [])
                if lst:
                    o = lst[0]
                    return {
                        "status": o.get("orderStatus"),
                        "cumExecQty": _safe_float(o.get("cumExecQty")),
                        "avgPrice": _safe_float(o.get("avgPrice")),
                        "leavesQty": _safe_float(o.get("leavesQty")),
                        "raw": o
                    }
            return None
        except Exception as e:
            logger.error(f"[API] get_order_status {symbol}: {e}")
            return None

    async def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        position_idx: int = 0,
        tpsl_mode: str = "Full"
    ) -> Dict[str, Any]:
        try:
            params: Dict[str, Any] = {
                "category": "linear",
                "symbol": symbol,
                "positionIdx": position_idx,
                "tpslMode": tpsl_mode
            }
            if take_profit:
                params["takeProfit"] = take_profit
            if stop_loss:
                params["stopLoss"] = stop_loss

            logger.info(f"[API] SL/TP: {symbol} SL={stop_loss} TP={take_profit}")
            result = await self._make_request("POST", "/v5/position/trading-stop", params=params)
            return result or {"retCode": -1, "retMsg": "No response"}
        except Exception as e:
            logger.error(f"[API] set_trading_stop {symbol}: {e}")
            return {"retCode": -1, "retMsg": str(e)}

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        try:
            get_params: List[Tuple[str, Any]] = [
                ("category", "linear"),
                ("symbol", symbol)
            ]
            positions = await self._make_request(
                "GET",
                "/v5/position/list",
                params=get_params
            )

            if positions and positions.get('retCode') == 0:
                pos_list = positions.get('result', {}).get('list', [])
                if pos_list:
                    current_leverage = float(pos_list[0].get('leverage', '0'))
                    if current_leverage == leverage:
                        return {"retCode": 0, "retMsg": "Already set"}

            result = await self._make_request(
                "POST",
                "/v5/position/set-leverage",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "buyLeverage": str(leverage),
                    "sellLeverage": str(leverage)
                }
            )
            return result or {"retCode": -1, "retMsg": "No response"}
        except Exception as e:
            logger.error(f"[API] set_leverage {symbol}: {e}")
            return {"retCode": -1, "retMsg": str(e)}

    async def get_positions(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            if symbol:
                params: List[Tuple[str, Any]] = [("category", "linear"), ("symbol", symbol)]
            else:
                params = [("category", "linear"), ("settleCoin", "USDT")]

            result = await self._make_request("GET", "/v5/position/list", params=params)
            return result
        except Exception as e:
            logger.error(f"[API] get_positions: {e}")
            return None

    async def get_closed_pnl(self, symbol: str, limit: int = 10, start_time: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """ПОКРАЩЕНИЙ: Отримання закритого PnL з правильними параметрами"""
        try:
            params: List[Tuple[str, Any]] = [
                ("category", "linear"),
                ("limit", limit),
                ("symbol", symbol),
            ]
            
            # КРИТИЧНО: Використовуємо startTime у мілісекундах
            if start_time is not None:
                params.append(("startTime", start_time))
                
            result = await self._make_request(
                "GET",
                "/v5/position/closed-pnl",
                params=params
            )
            return result
        except Exception as e:
            logger.error(f"[API] get_closed_pnl {symbol}: {e}")
            return None

    async def get_recent_executions(
        self,
        symbol: str,
        limit: int = 50,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            params: List[Tuple[str, Any]] = [("category", "linear"), ("limit", limit), ("symbol", symbol)]
            if start_time is not None:
                params.append(("startTime", start_time))
            if end_time is not None:
                params.append(("endTime", end_time))

            result = await self._make_request("GET", "/v5/execution/list", params=params)
            return result
        except Exception as e:
            logger.error(f"[API] get_recent_executions {symbol}: {e}")
            return None

    async def get_order_history(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100,
        order_filter: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            params: List[Tuple[str, Any]] = [("category", "linear")]
            if order_id:
                params.append(("orderId", order_id))
            if limit:
                params.append(("limit", limit))
            params.append(("symbol", symbol))
            if start_time is not None:
                params.append(("startTime", start_time))
            if end_time is not None:
                params.append(("endTime", end_time))
            if order_filter:
                params.append(("orderFilter", order_filter))

            result = await self._make_request("GET", "/v5/order/history", params=params)
            return result
        except Exception as e:
            logger.error(f"[API] get_order_history {symbol}: {e}")
            return None

    async def get_tpsl_orders(self, symbol: str) -> Dict[str, Optional[str]]:
        try:
            rt_params: List[Tuple[str, Any]] = [
                ("category", "linear"),
                ("symbol", symbol),
                ("orderFilter", "tpslOrder")
            ]
            rt = await self._make_request(
                "GET",
                "/v5/order/realtime",
                params=rt_params
            )

            tp_id: Optional[str] = None
            sl_id: Optional[str] = None

            def extract_ids(items: List[Dict[str, Any]]):
                nonlocal tp_id, sl_id
                for o in items:
                    t = (o.get("stopOrderType") or o.get("tpslType") or "").lower()
                    oid = o.get("orderId")
                    status = (o.get("orderStatus") or "").lower()
                    if not oid:
                        continue
                    if t in ("take_profit", "takeprofit", "tp") and tp_id is None and status in ("untriggered", "new", "created"):
                        tp_id = oid
                    if t in ("stop_loss", "stoploss", "sl", "stop") and sl_id is None and status in ("untriggered", "new", "created"):
                        sl_id = oid

            if rt and rt.get("retCode") == 0:
                items = rt.get("result", {}).get("list", []) or []
                extract_ids(items)

            if tp_id is None or sl_id is None:
                hist = await self.get_order_history(symbol, limit=200, order_filter="tpslOrder")
                if hist and hist.get("retCode") == 0:
                    items = hist.get("result", {}).get("list", []) or []
                    items_sorted = sorted(items, key=lambda x: int(x.get("updatedTime", x.get("createdTime", "0"))), reverse=True)
                    extract_ids(items_sorted)

            return {"tp": tp_id, "sl": sl_id}
        except Exception as e:
            logger.error(f"[API] get_tpsl_orders {symbol}: {e}")
            return {"tp": None, "sl": None}

    async def get_active_tpsl_orders(self, symbol: str) -> list:
        """Отримання АКТИВНИХ TP/SL ордерів в реальному часі"""
        try:
            result = await self._make_request(
                "GET", 
                "/v5/order/realtime",
                params=[
                    ("category", "linear"),
                    ("symbol", symbol),
                    ("orderFilter", "tpslOrder")
                ]
            )
            if result and result.get('retCode') == 0:
                return result.get('result', {}).get('list', [])
            return []
        except Exception as e:
            logger.error(f"[API] get_active_tpsl_orders error: {e}")
            return []

    async def get_positions_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """ПОКРАЩЕНИЙ: Отримання інформації про позицію для моніторингу з КЕШУВАННЯМ"""
        try:
            current_time = time.time()
            cache_key = f"position_summary_{symbol}"
            
            # Перевіряємо кеш
            if hasattr(self, '_position_cache'):
                cached_data = self._position_cache.get(cache_key)
                if cached_data and current_time - cached_data['timestamp'] < self._position_cache_ttl:
                    return cached_data['data']
            
            # Робимо запит до API
            result = await self.get_positions(symbol)
            if result and result.get('retCode') == 0:
                positions = result.get('result', {}).get('list', [])
                if positions:
                    pos_data = positions[0]
                    summary = {
                        "side": pos_data.get('side', 'FLAT'),
                        "size": _safe_float(pos_data.get('size'), 0.0),
                        "entry_price": _safe_float(pos_data.get('avgPrice'), 0.0),
                        "liq_price": _safe_float(pos_data.get('liqPrice'), 0.0),
                        "leverage": _safe_float(pos_data.get('leverage'), 1.0),
                        "unrealised_pnl": _safe_float(pos_data.get('unrealisedPnl'), 0.0),
                        "realised_pnl": _safe_float(pos_data.get('realisedPnl'), 0.0)
                    }
                else:
                    summary = {"side": "FLAT", "size": 0.0}
                
                # Кешуємо результат
                if not hasattr(self, '_position_cache'):
                    self._position_cache = {}
                self._position_cache[cache_key] = {
                    'data': summary,
                    'timestamp': current_time
                }
                
                return summary
            return None
        except Exception as e:
            logger.error(f"[API] get_positions_summary {symbol}: {e}")
            return None

    @staticmethod
    def quantize(value: float, step: float, mode: str = "floor") -> float:
        if step <= 0:
            return value
        import math
        k = value / step
        k = math.ceil(k) if mode == "ceil" else math.floor(k)
        return round(k * step, 15)

    def normalize_qty_price(
        self,
        symbol: str,
        info: Dict[str, Any],
        qty: float,
        price: float
    ) -> Tuple[float, float, Dict[str, Any]]:
        lot_filter = info.get("lotSizeFilter", {})
        price_filter = info.get("priceFilter", {})

        min_qty = _safe_float(lot_filter.get("minOrderQty"), 0.0)
        qty_step = _safe_float(lot_filter.get("qtyStep"), 0.001)
        tick_size = _safe_float(price_filter.get("tickSize"), 0.01)

        min_cost = _safe_float(lot_filter.get("minOrderCost"), 0.0)
        if min_cost <= 0:
            min_cost = _safe_float(lot_filter.get("minOrderAmt"), 0.0)

        q = self.quantize(qty, qty_step, "floor")
        if q < min_qty:
            q = min_qty

        p = self.quantize(price, tick_size, "floor")
        min_price = _safe_float(price_filter.get("minPrice"), 0.0)
        if min_price > 0 and p < min_price:
            p = min_price

        notional = q * p
        if min_cost > 0 and notional < min_cost:
            target_qty = min_cost / max(p, 1e-9)
            q2 = self.quantize(target_qty, qty_step, "ceil")
            if q2 < min_qty:
                q2 = min_qty
            q = q2
            notional = q * p

        meta = {
            "min_qty": min_qty,
            "qty_step": qty_step,
            "tick_size": tick_size,
            "min_cost": min_cost,
            "min_price": min_price,
            "final_notional": notional,
        }
        return q, p, meta