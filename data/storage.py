# data/storage.py
import time
import aiohttp
import asyncio
import uuid
import json
from utils.logger import logger
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional, Callable, Awaitable
from config.settings import settings


@dataclass
class TradeEntry:
    ts: float
    price: float
    size: float
    side: str
    is_aggressive: bool
    symbol: str = ""


@dataclass
class OrderBookLevel:
    price: float
    size: float
    ts: float


@dataclass
class OrderBookSnapshot:
    ts: float
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    best_bid: float
    best_ask: float


@dataclass
class SuspiciousOrder:
    price: float
    size: float
    side: str
    placed_ts: float
    removed_ts: Optional[float] = None
    lifetime_ms: Optional[float] = None


@dataclass
class Position:
    symbol: str
    side: str
    qty: float
    entry_price: float
    position_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    current_price: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    leverage: int = 1
    liq_price: float = 0.0
    timestamp: float = field(default_factory=time.time)
    status: str = "OPEN"
    last_update: float = field(default_factory=time.time)
    closed_timestamp: float = 0.0
    exit_price: float = 0.0
    close_reason: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0
    exchange_order_id: str = ""
    tp_order_id: str = ""
    sl_order_id: str = ""
    _position_updated: bool = False
    _closure_logged: bool = False
    _pnl_synced: bool = False
    _exchange_close_reason: str = ""
    _open_logged: bool = False
    meta_open: str = ""
    _closed_without_pnl_since: float = 0.0
    pnl_confirmed: bool = False
    avg_exit_price: float = 0.0
    realized_pnl: float = 0.0
    _external_close_processed: bool = False
    _last_monitor_check: float = field(default_factory=time.time)
    max_lifetime_sec: float = 0.0


class DataStorage:
    """Оновлене сховище з spread tracking для O'Hara Method 7"""

    async def init_orderbook_rest(self, symbol: str):
        """REST ініціалізація orderbook"""
        url_bases = [
            settings.system.rest_market_base.rstrip("/"),
            "https://api-demo.bybit.com",
            "https://api.bybit.com",
        ]
        seen = set()
        bases = []
        for b in url_bases:
            if b not in seen:
                bases.append(b)
                seen.add(b)

        for base in bases:
            url = f"{base}/v5/market/orderbook"
            params = {"category": "linear", "symbol": symbol, "limit": 50}
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, params=params) as response:
                        if response.status != 200:
                            continue
                        raw = await response.text()
                        try:
                            data = json.loads(raw)
                        except Exception:
                            continue

                        if data.get('retCode') == 0:
                            bids = data['result']['b']
                            asks = data['result']['a']
                            self.update_order_book(symbol, bids, asks)
                            logger.info(f"[REST_OB] Initialized {symbol}")
                            return
            except Exception as e:
                logger.warning(f"[REST_OB] Failed {symbol} via {base}: {e}")

        logger.warning(f"[REST_OB] Failed to init {symbol}")

    def __init__(
        self,
        retention_seconds: int,
        large_order_side_percent: float,
        spoof_lifetime_ms: int,
        large_order_min_abs: float,
        max_depth: int,
    ):
        self.retention_seconds = retention_seconds
        self.large_order_side_percent = large_order_side_percent
        self.spoof_lifetime_ms = spoof_lifetime_ms
        self.large_order_min_abs = large_order_min_abs
        self.max_depth = max_depth

        self._trades: Dict[str, Deque[TradeEntry]] = {}
        self._order_books: Dict[str, OrderBookSnapshot] = {}
        self._suspicious_orders: Dict[str, List[SuspiciousOrder]] = {}
        self._active_large_levels: Dict[str, Dict[str, Dict[float, Dict[str, float]]]] = {}
        self._book_maps: Dict[str, Dict[str, Dict[float, float]]] = {}

        self.positions: Dict[str, Position] = {}
        self._position_callbacks: List[Callable[[Position], Awaitable[None]]] = []
        self._closed_positions_history: Dict[str, Position] = {}
        self._recent_closed_pnl: Dict[str, List[Dict]] = {}

        self._sync_in_progress = False
        self._last_sync_attempt = 0
        self._sync_timeout = 15
        self._last_monitor_update: Dict[str, float] = {}
        
        # 🆕 O'HARA METHOD 7: Spread tracking
        self._current_spreads: Dict[str, float] = {}  # {symbol: spread_bps}

    def init_symbol(self, symbol: str):
        if symbol not in self._trades:
            self._trades[symbol] = deque()
        if symbol not in self._suspicious_orders:
            self._suspicious_orders[symbol] = []
        if symbol not in self._active_large_levels:
            self._active_large_levels[symbol] = {"bid": {}, "ask": {}}
        if symbol not in self._book_maps:
            self._book_maps[symbol] = {"bids": {}, "asks": {}}

    def add_position_callback(self, callback: Callable[[Position], Awaitable[None]]):
        self._position_callbacks.append(callback)

    async def _trigger_position_callbacks(self, position: Position):
        """Виклик callbacks"""
        try:
            if position._position_updated:
                for callback in self._position_callbacks:
                    try:
                        await callback(position)
                    except Exception as e:
                        logger.error(f"❌ [CALLBACK_ERROR] {position.symbol}: {e}")
                position._position_updated = False
        except Exception as e:
            logger.error(f"❌ [CALLBACK_TRIGGER_ERROR] {e}")

    async def update_position_from_exchange(self, position_data: Dict):
        """Спрощене оновлення позиції"""
        symbol = position_data['symbol']
        current_time = time.time()

        if symbol in self._last_monitor_update:
            time_since_last = current_time - self._last_monitor_update[symbol]
            if time_since_last < 1.0:
                return
        
        self._last_monitor_update[symbol] = current_time

        def safe_float(value, default=0.0):
            try:
                return float(value) if value not in [None, ''] else default
            except (ValueError, TypeError):
                return default

        size = safe_float(position_data.get('size'), 0)
        mark_price = safe_float(position_data.get('markPrice'), 0)
        unrealised_pnl = safe_float(position_data.get('unrealisedPnl'), 0)
        realised_pnl = safe_float(position_data.get('realisedPnl'), 0)
        avg_price = safe_float(position_data.get('avgPrice'), 0)

        exchange_side = position_data.get('side', '').upper()
        if exchange_side == 'BUY':
            new_side = 'LONG'
        elif exchange_side == 'SELL':
            new_side = 'SHORT'
        else:
            new_side = 'LONG' if size > 0 else 'SHORT' if size < 0 else 'UNKNOWN'

        is_now_open = abs(size) > 0.001
        new_status = "OPEN" if is_now_open else "CLOSED"

        if symbol in self.positions:
            position = self.positions[symbol]
            old_status = position.status

            if new_status == "CLOSED" and position._external_close_processed:
                return

            if new_side != 'UNKNOWN' and position.side != new_side and abs(size) > 0.001:
                logger.warning(f"🔄 [SIDE_CHANGE] {symbol}: {position.side} -> {new_side}")
                position.side = new_side

            position.qty = abs(size)
            position.entry_price = avg_price if avg_price > 0 else position.entry_price
            position.current_price = mark_price
            position.unrealised_pnl = unrealised_pnl
            position.realised_pnl = realised_pnl
            position._pnl_synced = True
            position.status = new_status
            position.last_update = current_time
            position._last_monitor_check = current_time
            position._position_updated = True

            if new_status == "CLOSED" and old_status == "OPEN":
                position._external_close_processed = True
                position.closed_timestamp = current_time
                position.exit_price = mark_price if mark_price > 0 else position.exit_price
                
                if not position.close_reason:
                    position.close_reason = "PENDING"
                
                self._closed_positions_history[symbol] = position
                logger.info(f"🔒 [EXCHANGE_CLOSE] {symbol}: Status changed to CLOSED")

        else:
            if new_side == 'UNKNOWN':
                return

            position = Position(
                symbol=symbol,
                side=new_side,
                qty=abs(size),
                entry_price=avg_price if avg_price > 0 else mark_price,
                current_price=mark_price,
                unrealised_pnl=unrealised_pnl,
                realised_pnl=realised_pnl,
                leverage=int(position_data.get('leverage', 1)),
                liq_price=safe_float(position_data.get('liqPrice'), 0),
                status=new_status,
                _pnl_synced=True,
                _position_updated=True,
                _last_monitor_check=current_time
            )
            
            if new_status == "CLOSED":
                position._external_close_processed = True
                position.close_reason = "PENDING"
                
            self.positions[symbol] = position
            logger.info(f"✅ [EXCHANGE_CREATE] {symbol}: {new_side} {abs(size)}")

        await self._trigger_position_callbacks(position)

    async def force_sync_positions(self, api_client):
        """ШВИДКА синхронізація"""
        if self._sync_in_progress:
            return False
            
        self._sync_in_progress = True
        self._last_sync_attempt = time.time()
        
        try:
            async with asyncio.timeout(15):
                return await self._fast_sync_positions(api_client)
        except asyncio.TimeoutError:
            logger.error("❌ [SYNC] Timeout")
            return False
        except Exception as e:
            logger.error(f"❌ [SYNC] Error: {e}")
            return False
        finally:
            self._sync_in_progress = False

    async def _fast_sync_positions(self, api_client):
        """Швидка синхронізація"""
        try:
            response = await api_client.get_positions()
            if not response or response.get('retCode') != 0:
                return False

            exchange_positions = response.get('result', {}).get('list', [])
            
            for pos_data in exchange_positions:
                symbol = pos_data.get('symbol')
                if symbol in settings.pairs.trade_pairs:
                    await self.update_position_from_exchange(pos_data)
            
            logger.debug(f"✅ [FAST_SYNC] {len(exchange_positions)} positions")
            return True
            
        except Exception as e:
            logger.error(f"❌ [FAST_SYNC] Error: {e}")
            return False

    def apply_execution_event(self, exec_data: Dict):
        """Спрощена обробка execution"""
        try:
            symbol = exec_data.get("symbol")
            if not symbol:
                return
                
            pos = self.positions.get(symbol)
            if not pos:
                return

            order_id = exec_data.get("orderId") or exec_data.get("orderID")
            
            if order_id:
                exec_type = str(exec_data.get('execType', '')).upper()
                if any(tp in exec_type for tp in ['TAKE_PROFIT', 'TP']):
                    pos.tp_order_id = order_id
                    logger.debug(f"[EXEC] {symbol}: TP order ID saved: {order_id}")
                elif any(sl in exec_type for sl in ['STOP_LOSS', 'STOP', 'SL']):
                    pos.sl_order_id = order_id
                    logger.debug(f"[EXEC] {symbol}: SL order ID saved: {order_id}")

            closed_pnl = exec_data.get("closedPnl")
            if closed_pnl is not None:
                try:
                    pos.realised_pnl = float(closed_pnl)
                    pos._pnl_synced = True
                except (ValueError, TypeError):
                    pass

            pos._position_updated = True
            asyncio.create_task(self._trigger_position_callbacks(pos))

        except Exception as e:
            logger.error(f"❌ [EXEC_EVENT] Error: {e}")

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        return self.positions.copy()

    def get_open_positions(self) -> Dict[str, Position]:
        return {sym: pos for sym, pos in self.positions.items() if pos.status == "OPEN"}

    def get_closed_positions(self) -> Dict[str, Position]:
        return {sym: pos for sym, pos in self.positions.items() if pos.status == "CLOSED"}

    def get_closed_positions_history(self) -> Dict[str, Position]:
        return self._closed_positions_history.copy()

    def add_trade(self, symbol: str, price: float, size: float, side: str, is_aggressive: bool):
        self.init_symbol(symbol)
        now = time.time()
        dq = self._trades[symbol]
        dq.append(TradeEntry(ts=now, price=price, size=size, side=side, is_aggressive=is_aggressive, symbol=symbol))
        cutoff = now - self.retention_seconds
        while dq and dq[0].ts < cutoff:
            dq.popleft()

    def update_order_book(self, symbol: str, bids: List[List[str]], asks: List[List[str]]):
        self.init_symbol(symbol)
        now = time.time()
        bid_map, ask_map = {}, {}
        bid_total = 0.0
        ask_total = 0.0
        for p, s in bids:
            pf = float(p)
            sf = float(s)
            bid_map[pf] = sf
            bid_total += sf
        for p, s in asks:
            pf = float(p)
            sf = float(s)
            ask_map[pf] = sf
            ask_total += sf
        self._book_maps[symbol]["bids"] = bid_map
        self._book_maps[symbol]["asks"] = ask_map
        self._rebuild_snapshot_and_detect(symbol, bid_total, ask_total, now)

    def apply_order_book_delta(self, symbol: str, bids_delta: List[List[str]], asks_delta: List[List[str]]):
        self.init_symbol(symbol)
        now = time.time()
        bid_map = self._book_maps[symbol]["bids"]
        ask_map = self._book_maps[symbol]["asks"]
        for p, s in bids_delta:
            pf = float(p)
            sf = float(s)
            if sf == 0 or abs(sf) < 1e-20:
                bid_map.pop(pf, None)
            else:
                bid_map[pf] = sf
        for p, s in asks_delta:
            pf = float(p)
            sf = float(s)
            if sf == 0 or abs(sf) < 1e-20:
                ask_map.pop(pf, None)
            else:
                ask_map[pf] = sf
        bid_total = sum(bid_map.values())
        ask_total = sum(ask_map.values())
        self._rebuild_snapshot_and_detect(symbol, bid_total, ask_total, now)

    def _rebuild_snapshot_and_detect(self, symbol: str, bid_total: float, ask_total: float, now: float):
        if not isinstance(symbol, str) or not symbol:
            return
        if not isinstance(bid_total, (int, float)) or not isinstance(ask_total, (int, float)):
            return
        if bid_total < 0 or ask_total < 0:
            return

        bid_map = self._book_maps.get(symbol, {}).get("bids", {})
        ask_map = self._book_maps.get(symbol, {}).get("asks", {})

        if not bid_map or not ask_map:
            return

        try:
            sorted_bids = sorted(bid_map.items(), key=lambda x: x[0], reverse=True)[:self.max_depth]
            sorted_asks = sorted(ask_map.items(), key=lambda x: x[0])[:self.max_depth]
        except Exception:
            return

        if not sorted_bids or not sorted_asks:
            return

        bids_levels = []
        asks_levels = []
        for p, s in sorted_bids:
            if p > 0 and s > 0:
                bids_levels.append(OrderBookLevel(price=p, size=s, ts=now))
        for p, s in sorted_asks:
            if p > 0 and s > 0:
                asks_levels.append(OrderBookLevel(price=p, size=s, ts=now))

        if not bids_levels or not asks_levels:
            return

        best_bid = bids_levels[0].price if bids_levels else 0.0
        best_ask = asks_levels[0].price if asks_levels else 0.0

        if best_bid <= 0 or best_ask <= 0:
            return
        if best_bid >= best_ask and best_ask > 0:
            return

        self._order_books[symbol] = OrderBookSnapshot(
            ts=now,
            bids=bids_levels,
            asks=asks_levels,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        
        # 🆕 O'HARA METHOD 7: Track spread
        if best_bid > 0 and best_ask > best_bid:
            spread_bps = (best_ask - best_bid) / best_bid * 10000
            self._current_spreads[symbol] = spread_bps

        large_bid_threshold = max(self.large_order_side_percent * bid_total, self.large_order_min_abs)
        large_ask_threshold = max(self.large_order_side_percent * ask_total, self.large_order_min_abs)

        active = self._active_large_levels[symbol]
        current_bids_map = {lvl.price: lvl.size for lvl in bids_levels}
        current_asks_map = {lvl.price: lvl.size for lvl in asks_levels}

        to_remove_bid = []
        for price, meta in active["bid"].items():
            if price not in current_bids_map or current_bids_map[price] < 1e-12:
                lifetime_ms = (now - meta["first_seen_ts"]) * 1000.0
                if lifetime_ms < settings.imbalance.spoof_lifetime_ms:
                    self._suspicious_orders[symbol].append(
                        SuspiciousOrder(
                            price=price,
                            size=meta["size"],
                            side="bid",
                            placed_ts=meta["first_seen_ts"],
                            removed_ts=now,
                            lifetime_ms=lifetime_ms,
                        )
                    )
                to_remove_bid.append(price)
        for p in to_remove_bid:
            active["bid"].pop(p, None)

        to_remove_ask = []
        for price, meta in active["ask"].items():
            if price not in current_asks_map or current_asks_map[price] < 1e-12:
                lifetime_ms = (now - meta["first_seen_ts"]) * 1000.0
                if lifetime_ms < settings.imbalance.spoof_lifetime_ms:
                    self._suspicious_orders[symbol].append(
                        SuspiciousOrder(
                            price=price,
                            size=meta["size"],
                            side="ask",
                            placed_ts=meta["first_seen_ts"],
                            removed_ts=now,
                            lifetime_ms=lifetime_ms,
                        )
                    )
                to_remove_ask.append(price)
        for p in to_remove_ask:
            active["ask"].pop(p, None)

        for lvl in bids_levels:
            if lvl.size >= large_bid_threshold and lvl.price not in active["bid"]:
                active["bid"][lvl.price] = {"size": lvl.size, "first_seen_ts": now}
        for lvl in asks_levels:
            if lvl.size >= large_ask_threshold and lvl.price not in active["ask"]:
                active["ask"][lvl.price] = {"size": lvl.size, "first_seen_ts": now}

    def get_trades(self, symbol: str):
        return list(self._trades.get(symbol, []))

    def get_order_book(self, symbol: str):
        return self._order_books.get(symbol)
    
    def get_current_spread_bps(self, symbol: str) -> Optional[float]:
        """🆕 O'HARA METHOD 7: Get current spread in basis points"""
        return self._current_spreads.get(symbol)

    def get_suspicious_orders(self, symbol: str, last_seconds: int = 60):
        now = time.time()
        return [
            o
            for o in self._suspicious_orders.get(symbol, [])
            if (o.removed_ts or now) >= now - last_seconds
        ]