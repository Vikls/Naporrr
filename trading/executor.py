# trading/executor.py
import asyncio
import time
import uuid
import csv
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from config.settings import settings
from utils.logger import logger
from trading.bybit_api_manager import BybitAPIManager
from trading.risk_manager import RiskManager
from data.storage import Position, DataStorage

@dataclass
class ActiveOrder:
    symbol: str
    side: str
    qty: float
    price: float
    position_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    order_id: Optional[str] = None
    filled_qty: float = 0.0
    avg_price: float = 0.0
    created_ts: float = field(default_factory=time.time)
    last_update_ts: float = field(default_factory=time.time)
    reprice_attempts: int = 0
    state: str = "WORKING"
    is_reversed: bool = False
    double_size: bool = False
    signal_info: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0
    adaptive_lifetime_sec: float = 0.0

class TokenBucket:
    """Rate limiting через token bucket"""
    def __init__(self, capacity: int, refill_per_sec: float):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1):
        async with self._lock:
            await self._refill()
            while self.tokens < tokens:
                need = tokens - self.tokens
                wait = need / self.refill_per_sec if self.refill_per_sec > 0 else 0.1
                await asyncio.sleep(max(0.01, wait))
                await self._refill()
            self.tokens -= tokens

    async def _refill(self):
        now = time.time()
        delta = now - self.last_refill
        if delta <= 0:
            return
        add = delta * self.refill_per_sec
        self.tokens = min(self.capacity, self.tokens + add)
        self.last_refill = now

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _safe_cast_float(value) -> float:
    try:
        return float(value) if value not in [None, ''] else 0.0
    except (ValueError, TypeError):
        return 0.0

class CloseReasonDetector:
    """ЦЕНТРАЛІЗОВАНИЙ детектор причин закриття позицій"""
    
    def __init__(self, settings_risk):
        self.settings = settings_risk
        self.logger = logger
    
    def determine_close_reason(self, position: Position, exit_price: float, 
                              current_time: float, exchange_data: Optional[Dict] = None) -> str:
        """Головний метод визначення причини закриття"""
        symbol = position.symbol
        lifetime_sec = current_time - position.timestamp
        
        self.logger.debug(f"[CLOSE_REASON] {symbol}: lifetime={lifetime_sec:.0f}s, "
                         f"exit={exit_price:.6f}, entry={position.entry_price:.6f}")
        
        # 🔴 ПРІОРИТЕТ 1: TIME_EXIT (перевіряємо ПЕРШИМ)
        max_lifetime = self._get_max_lifetime(position)
        time_exit_threshold = max_lifetime * 0.95  # 95% від максимального часу
        
        if lifetime_sec >= time_exit_threshold:
            self.logger.info(f"⏰ [CLOSE_REASON] {symbol}: TIME_EXIT "
                           f"({lifetime_sec:.0f}s >= {time_exit_threshold:.0f}s)")
            return "TIME_EXIT"
        
        # 🟢 ПРІОРИТЕТ 2: Перевірка вже встановленої причини від orchestrator
        if position.close_reason in ["REVERSE", "opp_signal", "EMERGENCY_CLOSE"]:
            self.logger.info(f"✅ [CLOSE_REASON] {symbol}: Using orchestrator reason: {position.close_reason}")
            return position.close_reason
        
        # 🟡 ПРІОРИТЕТ 3: TP/SL по orderId (якщо є біржеві дані)
        if exchange_data:
            reason_from_exchange = self._check_exchange_order_ids(position, exchange_data)
            if reason_from_exchange:
                self.logger.info(f"🎯 [CLOSE_REASON] {symbol}: From exchange data: {reason_from_exchange}")
                return reason_from_exchange
        
        # 🟡 ПРІОРИТЕТ 4: TP/SL по ціні з adaptive tolerance
        if exit_price > 0 and position.entry_price > 0:
            reason_by_price = self._check_price_based_reason(position, exit_price)
            if reason_by_price:
                self.logger.info(f"📊 [CLOSE_REASON] {symbol}: By price: {reason_by_price}")
                return reason_by_price
        
        # 🔵 ПРІОРИТЕТ 5: Fallback
        self.logger.info(f"🔄 [CLOSE_REASON] {symbol}: Using fallback: EXCHANGE_CLOSE")
        return "EXCHANGE_CLOSE"
    
    def _get_max_lifetime(self, position: Position) -> float:
        """Отримати максимальний lifetime для позиції (може бути адаптивним)"""
        if hasattr(position, 'max_lifetime_sec') and position.max_lifetime_sec > 0:
            return position.max_lifetime_sec
        return self.settings.max_position_lifetime_sec
    
    def _check_exchange_order_ids(self, position: Position, exchange_data: Dict) -> Optional[str]:
        """Перевірка по orderId з біржі"""
        order_id = exchange_data.get("orderId") or exchange_data.get("orderID")
        
        if not order_id:
            return None
        
        # Порівнюємо з збереженими TP/SL orderIds
        if order_id == position.tp_order_id:
            return "TP_HIT"
        elif order_id == position.sl_order_id:
            return "SL_HIT"
        
        # Перевірка exec_type
        exec_type = str(exchange_data.get('execType', '')).upper()
        if any(tp in exec_type for tp in ['TAKE_PROFIT', 'TP']):
            return "TP_HIT"
        elif any(sl in exec_type for sl in ['STOP_LOSS', 'STOP', 'SL']):
            return "SL_HIT"
        
        return None
    
    def _check_price_based_reason(self, position: Position, exit_price: float) -> Optional[str]:
        """Перевірка по ціні з ADAPTIVE tolerance"""
        if exit_price <= 0 or position.entry_price <= 0:
            return None
        
        tolerance = self._get_adaptive_tolerance(position.symbol, position.entry_price)
        
        tp_diff = abs(exit_price - position.take_profit)
        sl_diff = abs(exit_price - position.stop_loss)
        
        # Збільшений tolerance для SL (більше slippage)
        sl_tolerance = tolerance * 2.0
        
        self.logger.debug(f"[PRICE_CHECK] {position.symbol}: TP_diff={tp_diff:.6f}, "
                         f"SL_diff={sl_diff:.6f}, tolerance={tolerance:.6f}")
        
        if position.side == "LONG":
            # TP для LONG - ціна вище entry
            if tp_diff <= tolerance and exit_price >= position.entry_price:
                return "TP_HIT"
            # SL для LONG - ціна нижче entry
            elif sl_diff <= sl_tolerance and exit_price <= position.entry_price:
                return "SL_HIT"
            # Жорсткі межі
            elif exit_price >= position.take_profit:
                return "TP_HIT"
            elif exit_price <= position.stop_loss:
                return "SL_HIT"
        
        else:  # SHORT
            # TP для SHORT - ціна нижче entry
            if tp_diff <= tolerance and exit_price <= position.entry_price:
                return "TP_HIT"
            # SL для SHORT - ціна вище entry
            elif sl_diff <= sl_tolerance and exit_price >= position.entry_price:
                return "SL_HIT"
            # Жорсткі межі
            elif exit_price <= position.take_profit:
                return "TP_HIT"
            elif exit_price >= position.stop_loss:
                return "SL_HIT"
        
        return None
    
    def _get_adaptive_tolerance(self, symbol: str, entry_price: float) -> float:
        """Адаптивний tolerance на основі символу та ціни"""
        # Базовий tolerance 0.1%
        base_tolerance = entry_price * 0.001
        
        # Специфічні налаштування для символів
        if symbol in ['BNBUSDT', 'SOLUSDT', 'AAVEUSDT', 'ETHUSDT']:
            # Великі монети - менший tolerance
            min_tolerance = max(entry_price * 0.0005, 0.1)
            max_tolerance = entry_price * 0.002
        elif symbol in ['ADAUSDT', 'DOGEUSDT', 'TRXUSDT', 'HFTUSDT']:
            # Малі монети - більший tolerance
            min_tolerance = max(entry_price * 0.001, 0.0001)
            max_tolerance = entry_price * 0.003
        else:
            # Інші
            min_tolerance = entry_price * 0.0007
            max_tolerance = entry_price * 0.0025
        
        tolerance = max(min(base_tolerance, max_tolerance), min_tolerance)
        
        return tolerance

class TradeExecutor:
    """🆕 ВИПРАВЛЕНИЙ Trade Executor з правильним розрахунком розміру позиції"""
    
    def __init__(self, storage: DataStorage, api: BybitAPIManager):
        self.storage = storage
        self.tcfg = settings.trading
        self.exec_cfg = settings.execution
        self.api = api
        self.risk = RiskManager(api_manager=api)
        
        # Централізований детектор причин
        self.close_reason_detector = CloseReasonDetector(settings.risk)

        self.active_orders: Dict[str, ActiveOrder] = {}
        self.positions: Dict[str, Position] = {}
        self._watch_tasks: Dict[str, asyncio.Task] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Rate limiting
        self.limit_sec = TokenBucket(self.tcfg.max_orders_per_second, self.tcfg.max_orders_per_second)
        self.limit_min = TokenBucket(self.tcfg.max_orders_per_minute, self.tcfg.max_orders_per_minute / 60.0)
        
        self._lock = asyncio.Lock()
        self._start_balance = self.tcfg.start_balance_usdt
        
        # Захист від дублювання
        self._blocked_symbols: Dict[str, str] = {}
        self._running = False
        self._processed_closures: Dict[str, float] = {}
        self._reverse_close_tracking: Dict[str, float] = {}
        self._pnl_attempts_cache: Dict[str, int] = {}

    def _init_trade_log(self):
        """Ініціалізація CSV файлу"""
        try:
            if not settings.logging.trades_log.exists():
                settings.logging.trades_log.parent.mkdir(parents=True, exist_ok=True)
                with open(settings.logging.trades_log, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "event", "symbol", "side", 
                        "qty", "price", "sl", "tp", 
                        "reason", "meta"
                    ])
                logger.info("✅ [TRADE_LOG] Initialized CSV file")
        except Exception as e:
            logger.error(f"❌ [TRADE_LOG] Failed to initialize: {e}")

    async def start(self):
        """Запуск executor"""
        logger.info("🚀 [EXECUTOR] Starting...")
        
        self._init_trade_log()
        
        if not await self.api.check_time_sync():
            logger.warning("[EXECUTOR] ⚠️ Time sync failed")
        
        # Встановлюємо leverage
        for sym in settings.pairs.trade_pairs:
            try:
                await self.api.set_leverage(sym, settings.trading.leverage)
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.warning(f"[EXECUTOR] Leverage set fail {sym}: {e}")
        
        # Отримуємо баланс
        bal = await self.api.get_wallet_balance()
        if bal is not None:
            self._start_balance = bal
            logger.info(f"[EXECUTOR] 💰 Balance: {bal:.2f} USDT")
        
        self._running = True
        
        # Паралельний або послідовний моніторинг
        if self.tcfg.enable_parallel_monitoring:
            self._monitor_task = asyncio.create_task(self._parallel_monitor_loop())
            logger.info("✅ [EXECUTOR] Started with PARALLEL monitoring")
        else:
            self._monitor_task = asyncio.create_task(self._sequential_monitor_loop())
            logger.info("✅ [EXECUTOR] Started with sequential monitoring")

    async def stop(self):
        """Зупинка executor"""
        logger.info("🛑 [EXECUTOR] Stopping...")
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        for t in list(self._watch_tasks.values()):
            t.cancel()
        await asyncio.gather(*self._watch_tasks.values(), return_exceptions=True)
        self._watch_tasks.clear()
        
        logger.info("✅ [EXECUTOR] Stopped")

    # ==================== MONITORING ====================
    
    async def _parallel_monitor_loop(self):
        """ПАРАЛЕЛЬНИЙ моніторинг позицій (швидший)"""
        interval = self.tcfg.monitor_positions_interval_sec
        batch_size = self.tcfg.monitoring_batch_size
        
        logger.info(f"🔍 [PARALLEL_MONITOR] interval={interval}s, batch_size={batch_size}")
        
        while self._running:
            await asyncio.sleep(interval)
            
            try:
                # Отримуємо відкриті позиції
                symbols = [sym for sym, pos in self.positions.items() if pos.status == "OPEN"]
                
                if not symbols:
                    continue
                
                # Розділяємо на батчі
                batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
                
                # Обробляємо батчі паралельно
                for batch in batches:
                    tasks = [self._check_position_status(sym) for sym in batch]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"❌ [PARALLEL_MONITOR] Error: {e}")
    
    async def _sequential_monitor_loop(self):
        """Послідовний моніторинг (старий варіант)"""
        interval = self.tcfg.monitor_positions_interval_sec
        logger.info(f"🔍 [SEQUENTIAL_MONITOR] interval={interval}s")
        
        while self._running:
            await asyncio.sleep(interval)
            
            try:
                symbols = [sym for sym, pos in self.positions.items() if pos.status == "OPEN"]
                for sym in symbols:
                    await self._check_position_status(sym)
            except Exception as e:
                logger.error(f"❌ [SEQUENTIAL_MONITOR] Error: {e}")

    async def _check_position_status(self, symbol: str):
        """ОНОВЛЕНА перевірка статусу позиції"""
        if symbol in self._blocked_symbols:
            return
        
        pos = self.positions.get(symbol)
        if not pos or pos.status != "OPEN":
            return
        
        current_time = time.time()
        
        # АДАПТИВНИЙ LIFETIME з vol_data
        if hasattr(pos, 'max_lifetime_sec') and pos.max_lifetime_sec > 0:
            max_life = pos.max_lifetime_sec
        else:
            # Розрахунок адаптивного lifetime якщо не збережено
            max_life = settings.risk.max_position_lifetime_sec
        
        # Пріоритетна перевірка TIME_EXIT
        if current_time - pos.timestamp > max_life:
            lifetime_min = (current_time - pos.timestamp) / 60.0
            logger.info(f"[MONITOR] ⏰ Closing {symbol} {pos.side} due to TIME_EXIT "
                       f"({lifetime_min:.1f}min > {max_life/60:.1f}min)")
            await self.close_position(symbol, reason="TIME_EXIT")
            return

        # Перевірка на біржі
        try:
            summary = await self.api.get_positions_summary(symbol)
            if not summary:
                return
            
            exchange_side = summary.get("side", "FLAT")
            exchange_size = summary.get("size", 0.0)
            
            if exchange_side == "FLAT" or exchange_size == 0.0:
                closure_key = f"{symbol}_{exchange_side}_{exchange_size}"
                if closure_key in self._processed_closures:
                    if time.time() - self._processed_closures[closure_key] < 60:
                        return
                
                logger.info(f"[MONITOR] 🔍 Position closed externally: {symbol}")
                self._processed_closures[closure_key] = time.time()
                await self._finalize_close_external(symbol)
        
        except Exception as e:
            logger.error(f"❌ [MONITOR] Error checking {symbol}: {e}")

    async def _finalize_close_external(self, symbol: str):
        """Обробка зовнішнього закриття позиції"""
        pos = self.positions.get(symbol)
        if not pos or pos.status != "OPEN":
            return
        
        logger.info(f"🔒 [EXTERNAL_CLOSE] Processing {symbol}")
        
        pos.status = "CLOSED"
        pos.closed_timestamp = time.time()
        self._blocked_symbols[symbol] = "awaiting_pnl"
        
        # Очікуємо PnL
        await self._await_and_record_pnl(symbol, pos)

    async def close_position(self, symbol: str, reason: str):
        """ОНОВЛЕНЕ закриття позиції з централізованою логікою"""
        pos = self.positions.get(symbol)
        if not pos:
            logger.warning(f"[CLOSE] Position not found: {symbol}")
            return
        if pos.status != "OPEN":
            logger.warning(f"[CLOSE] Position not OPEN: {symbol} (status={pos.status})")
            return
        
        # Зберігаємо причину ДО закриття (для пріоритизації)
        if reason in ["REVERSE", "TIME_EXIT", "opp_signal"]:
            pos.close_reason = reason
        else:
            pos.close_reason = "PENDING"
        
        logger.info(f"[CLOSE] 🎯 Closing {symbol} {pos.side} {pos.qty} (reason={reason})")
        
        close_side = "Buy" if pos.side == "SHORT" else "Sell"
        qty_str = self._fmt_qty(pos.qty)
        
        try:
            res = await self.api.place_order(
                symbol=symbol,
                side=close_side,
                qty=qty_str,
                order_type="Market",
                position_idx=0
            )
            
            if res.get("retCode") == 0:
                order_id = res.get("result", {}).get("orderId", "")
                if order_id:
                    pos.exchange_order_id = order_id
                    logger.info(f"✅ [CLOSE_ORDER] {symbol}: {order_id}")
                
                pos.status = "CLOSED"
                pos.closed_timestamp = time.time()
                
                self._blocked_symbols[symbol] = "awaiting_pnl"
                asyncio.create_task(self._await_and_record_pnl(symbol, pos))
                
                logger.info(f"⏳ [CLOSE_INITIATED] {symbol}: waiting for PnL...")
            else:
                logger.error(f"❌ [CLOSE] Failed {symbol}: {res.get('retMsg')}")
        
        except Exception as e:
            logger.error(f"❌ [CLOSE] Exception {symbol}: {e}")

    async def _await_and_record_pnl(self, symbol: str, pos: Position) -> None:
        """ОНОВЛЕНА логіка отримання PnL з централізованим визначенням причини"""
        max_attempts = 8
        pnl_received = False
        avg_exit_price = 0.0
        pnl_val = 0.0
        
        logger.info(f"💰 [PNL_AWAIT] Starting for {symbol}")
        
        for attempt in range(max_attempts):
            try:
                start_time_ms = int((pos.timestamp - 120) * 1000)
                
                pnl_result = await self.api.get_closed_pnl(
                    symbol=symbol,
                    limit=10,
                    start_time=start_time_ms
                )
                
                if pnl_result and pnl_result.get('retCode') == 0:
                    pnl_list = pnl_result.get('result', {}).get('list', [])
                    if pnl_list:
                        sorted_pnl = sorted(
                            pnl_list,
                            key=lambda x: int(x.get("updatedTime", "0")),
                            reverse=True
                        )
                        latest_pnl = sorted_pnl[0]
                        
                        pnl_val = _safe_cast_float(latest_pnl.get("closedPnl", 0))
                        avg_exit_price = _safe_cast_float(latest_pnl.get("avgExitPrice", 0))
                        
                        if avg_exit_price > 0:
                            pos.realised_pnl = pnl_val
                            pos.avg_exit_price = avg_exit_price
                            pos.pnl_confirmed = True
                            
                            # ЦЕНТРАЛІЗОВАНЕ визначення причини
                            current_time = time.time()
                            final_reason = self.close_reason_detector.determine_close_reason(
                                position=pos,
                                exit_price=avg_exit_price,
                                current_time=current_time,
                                exchange_data=latest_pnl
                            )
                            
                            pos.close_reason = final_reason
                            
                            # Для CSV конвертуємо opp_signal -> STRATEGY_SIGNAL
                            csv_reason = "STRATEGY_SIGNAL" if final_reason == "opp_signal" else final_reason
                            
                            lifetime_min = (current_time - pos.timestamp) / 60.0
                            meta = f"pnl={pnl_val:.6f}; avgExit={avg_exit_price:.6f}; lifetime={lifetime_min:.1f}min"
                            
                            self._log_trade(
                                "CLOSE", symbol, pos.side, pos.qty, avg_exit_price,
                                pos.stop_loss, pos.take_profit, csv_reason, meta
                            )
                            
                            # Додаємо в історію для статистики
                            self.risk.add_to_history(
                                symbol=symbol,
                                side=pos.side,
                                pnl=pnl_val,
                                close_reason=final_reason,
                                lifetime_sec=current_time - pos.timestamp
                            )
                            
                            logger.info(f"💰 [PNL_CONFIRMED] {symbol}: PnL={pnl_val:.4f}, "
                                      f"Exit={avg_exit_price:.6f}, Reason={csv_reason}")
                            
                            pnl_received = True
                            break
                        else:
                            logger.warning(f"⚠️ [PNL] Zero exit price for {symbol}, attempt {attempt + 1}")
                    else:
                        logger.warning(f"⚠️ [PNL] Empty list for {symbol}, attempt {attempt + 1}")
                else:
                    error_msg = pnl_result.get('retMsg', 'Unknown') if pnl_result else 'No response'
                    logger.warning(f"⚠️ [PNL] API error for {symbol}: {error_msg}")
                
                await asyncio.sleep(2.0)
            
            except Exception as e:
                logger.error(f"❌ [PNL] Error for {symbol} (attempt {attempt + 1}): {e}")
                await asyncio.sleep(2.0)
        
        # FALLBACK якщо не отримали PnL
        if not pnl_received:
            try:
                ticker_price = await self.api.get_ticker_price(symbol)
                if ticker_price and ticker_price > 0:
                    avg_exit_price = ticker_price
                    
                    if pos.side == "LONG":
                        pnl_val = (avg_exit_price - pos.entry_price) * pos.qty
                    else:
                        pnl_val = (pos.entry_price - avg_exit_price) * pos.qty
                else:
                    avg_exit_price = pos.entry_price
                    pnl_val = 0.0
                
                # ЦЕНТРАЛІЗОВАНЕ визначення причини
                current_time = time.time()
                final_reason = self.close_reason_detector.determine_close_reason(
                    position=pos,
                    exit_price=avg_exit_price,
                    current_time=current_time
                )
                
                pos.close_reason = final_reason
                csv_reason = "STRATEGY_SIGNAL" if final_reason == "opp_signal" else final_reason
                
                lifetime_min = (current_time - pos.timestamp) / 60.0
                meta = f"pnl={pnl_val:.6f}; avgExit={avg_exit_price:.6f}; lifetime={lifetime_min:.1f}min (FALLBACK)"
                
                self._log_trade(
                    "CLOSE", symbol, pos.side, pos.qty, avg_exit_price,
                    pos.stop_loss, pos.take_profit, csv_reason, meta
                )
                
                # Додаємо в історію
                self.risk.add_to_history(
                    symbol=symbol,
                    side=pos.side,
                    pnl=pnl_val,
                    close_reason=final_reason,
                    lifetime_sec=current_time - pos.timestamp
                )
                
                logger.warning(f"⚠️ [PNL_FALLBACK] {symbol}: {csv_reason}")
            
            except Exception as e:
                logger.error(f"❌ [PNL_FALLBACK] Error for {symbol}: {e}")
        
        # Cleanup
        self._pnl_attempts_cache.pop(symbol, None)
        self._blocked_symbols.pop(symbol, None)
        logger.info(f"✅ [PNL_COMPLETE] {symbol}")

    def _log_trade(self, event: str, symbol: str, side: str, qty: float, price: float,
                   sl: float, tp: float, reason: str, meta: str):
        """ОНОВЛЕНЕ логування з захистом від дублікатів"""
        try:
            current_time = time.time()
            
            # Захист від дублікатів для CLOSE
            if event == "CLOSE":
                close_key = f"{symbol}_{event}_{side}_{qty:.6f}_{price:.6f}_{reason}"
                
                if close_key in self._processed_closures:
                    last_log_time = self._processed_closures[close_key]
                    if current_time - last_log_time < 10:
                        logger.debug(f"🛡️ [DUPLICATE_LOG] Skipping: {symbol}")
                        return
                
                self._processed_closures[close_key] = current_time
                
                # Очищення старих записів
                expired_keys = [k for k, v in self._processed_closures.items() 
                              if current_time - v > 60]
                for k in expired_keys:
                    self._processed_closures.pop(k, None)
            
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
            
            # Запис у CSV
            with open(settings.logging.trades_log, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp_str, event, symbol, side,
                    f"{qty:.6f}", f"{price:.6f}",
                    f"{sl:.6f}", f"{tp:.6f}",
                    reason, meta
                ])
            
            if event == "CLOSE":
                logger.info(f"📝 [TRADE_LOG] {event}: {symbol} {side} {qty:.6f} @ {price:.6f} | {reason}")
            else:
                logger.debug(f"📝 [TRADE_LOG] {event}: {symbol} {side} {qty:.6f} @ {price:.6f}")
        
        except Exception as e:
            logger.error(f"❌ [TRADE_LOG] Error: {e}", exc_info=True)

    # ==================== OPEN POSITION ====================
    
    def can_open(self, symbol: str) -> bool:
        """Перевірка можливості відкриття"""
        if symbol in self._blocked_symbols:
            return False
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.status == "OPEN":
                return False
        if symbol in self.active_orders:
            return False
        return True

    async def open_position_limit(self, symbol: str, direction: str, ref_price: float,
                                  best_bid: Optional[float] = None, best_ask: Optional[float] = None,
                                  is_reversed: bool = False, double_size: bool = False,
                                  signal_info: str = "", volatility_data: Dict[str, Any] = None):
        """
        🆕 ВИПРАВЛЕНЕ відкриття позиції з правильним розрахунком розміру
        """
        logger.info(f"[OPEN] 🔓 Attempting {symbol} {direction} at {ref_price:.6f}")
        
        # Перевірка можливості відкриття
        if not await self._can_open_position(symbol, is_reversed):
            return
        
        try:
            # Отримуємо баланс
            balance = await self.api.get_wallet_balance()
            if balance is None or balance <= 0:
                logger.error(f"❌ [OPEN] Cannot get balance for {symbol}")
                return
            
            logger.info(f"[OPEN] 💰 Balance for {symbol}: ${balance:.2f}")

            # 🆕 ВИПРАВЛЕННЯ: Використовуємо новий метод з правильним розрахунком
            qty = await self.risk.calc_base_qty(symbol, ref_price, balance, self.api)
            if qty <= 0:
                logger.error(f"❌ [OPEN] Invalid calculated qty for {symbol}: {qty}")
                return
            
            if double_size:
                original_qty = qty
                qty *= 2
                logger.info(f"[OPEN] {symbol}: Double size {original_qty:.6f} -> {qty:.6f}")
            
            # Отримуємо інструмент для нормалізації
            inst = await self.api.get_instrument_info(symbol)
            if not inst:
                logger.error(f"[OPEN] ❌ No instrument info for {symbol}")
                return
            
            # Нормалізація ціни
            _, price, meta = self.api.normalize_qty_price(symbol, inst, qty, ref_price)
            
            # Додаткова перевірка мінімального номіналу
            min_notional = float(inst.get('lotSizeFilter', {}).get('minOrderAmt', 0))
            notional = qty * price
            if min_notional > 0 and notional < min_notional:
                logger.warning(f"[OPEN] {symbol}: Notional ${notional:.2f} < min ${min_notional:.2f}")
                required_qty = min_notional / price
                qty, price, meta = self.api.normalize_qty_price(symbol, inst, required_qty, price)
                logger.info(f"[OPEN] {symbol}: Adjusted to min notional: {qty:.6f}")
            
            # Покращення ціни для лімітних ордерів
            if best_bid and best_ask:
                improve = self.exec_cfg.passive_improve_bps / 10000.0
                if direction.upper() == "BUY":
                    price_ref = best_bid * (1 - improve)
                else:
                    price_ref = best_ask * (1 + improve)
                _, price_adj, _ = self.api.normalize_qty_price(symbol, inst, qty, price_ref)
                if price_adj != price:
                    logger.info(f"[OPEN] {symbol}: Price improved {price:.6f} -> {price_adj:.6f}")
                    price = price_adj
            
            side_pos = "LONG" if direction.upper() == "BUY" else "SHORT"
            
            # Розрахунок SL/TP
            sl, tp = self.risk.calc_sl_tp(
                side=side_pos,
                entry_price=price,
                volatility_data=volatility_data,
                symbol=symbol
            )
            
            # Адаптивний lifetime
            current_volatility = volatility_data.get('recent_volatility', 0.1) if volatility_data else 0.1
            adaptive_lifetime_sec = self.risk.get_adaptive_lifetime_seconds(symbol, current_volatility)
            
            logger.info(f"[OPEN] 📊 {symbol}: Qty={qty:.6f}, Price=${price:.6f}, SL=${sl:.6f}, TP=${tp:.6f}, Lifetime={adaptive_lifetime_sec/60:.1f}min")
        
        except Exception as e:
            logger.error(f"[OPEN] ❌ Error calculating params for {symbol}: {e}", exc_info=True)
            return
        
        # Створення та відправка ордера
        success = await self._create_and_submit_order(
            symbol, direction, qty, price, is_reversed, 
            double_size, signal_info, sl, tp, adaptive_lifetime_sec
        )
        
        if success:
            logger.info(f"[OPEN] ✅ Successfully opened {symbol} {direction} {qty:.6f} @ ${price:.6f}")
        else:
            logger.error(f"[OPEN] ❌ Failed to open {symbol}")

    async def _can_open_position(self, symbol: str, is_reversed: bool) -> bool:
        """Перевірка можливості відкриття позиції"""
        if not self.can_open(symbol) and not is_reversed:
            logger.info(f"[OPEN] ⏹️ Skip {symbol} - blocked")
            return False
        
        async with self._lock:
            current_pos = self.storage.get_position(symbol)
            if current_pos and current_pos.status == "OPEN" and not is_reversed:
                logger.info(f"[OPEN] ⏹️ Skip {symbol} - already open")
                return False
            
            open_positions_count = len([p for p in self.positions.values() 
                                      if getattr(p, 'status', 'OPEN') == "OPEN"])
            if open_positions_count >= settings.risk.max_open_positions and not is_reversed:
                logger.info(f"[OPEN] ⏹️ Max positions reached ({open_positions_count})")
                return False
        
        return True

    async def _create_and_submit_order(self, symbol: str, direction: str, qty: float, price: float,
                                      is_reversed: bool, double_size: bool, signal_info: str,
                                      sl: float, tp: float, adaptive_lifetime_sec: float) -> bool:
        """Створення та відправка ордера"""
        async with self._lock:
            if symbol in self.active_orders:
                logger.info(f"[OPEN] ⏹️ Skip {symbol} - active order exists")
                return False
            
            order = ActiveOrder(
                symbol=symbol,
                side=direction,
                qty=qty,
                price=price,
                is_reversed=is_reversed,
                double_size=double_size,
                signal_info=signal_info,
                stop_loss=sl,
                take_profit=tp,
                adaptive_lifetime_sec=adaptive_lifetime_sec
            )
            self.active_orders[symbol] = order
        
        # Rate limiting
        await self.limit_sec.acquire()
        await self.limit_min.acquire()
        
        logger.info(f"[OPEN] 🎯 Placing order: {symbol} {direction} {qty:.6f} @ ${price:.6f}")
        
        # Відправка ордера
        res = await self.api.place_order(
            symbol=symbol,
            side=self._api_side(direction),
            qty=self._fmt_qty(qty),
            order_type="Limit",
            price=f"{price:.6f}",  # Використовуємо більше знаків після коми
            position_idx=0,
        )
        
        if res.get("retCode") != 0:
            logger.error(f"[OPEN] ❌ Failed {symbol}: {res.get('retMsg')}")
            async with self._lock:
                self.active_orders.pop(symbol, None)
            return False
        
        order_id = res.get("result", {}).get("orderId")
        if not order_id:
            logger.error(f"[OPEN] ❌ Missing orderId for {symbol}")
            async with self._lock:
                self.active_orders.pop(symbol, None)
            return False
        
        # Оновлення ордера
        async with self._lock:
            if symbol in self.active_orders:
                self.active_orders[symbol].order_id = order_id
        
        # Запуск моніторингу ордера
        t = asyncio.create_task(self._watch_order(symbol))
        self._watch_tasks[symbol] = t
        
        return True

    # ==================== ORDER WATCHING ====================
    
    async def _watch_order(self, symbol: str):
        """Відстеження ордера"""
        cfg = self.exec_cfg
        start = time.time()
        last_reprice = start

        while True:
            await asyncio.sleep(cfg.poll_interval_sec)
            
            async with self._lock:
                order = self.active_orders.get(symbol)
            if not order:
                return

            status = await self.api.get_order_status(symbol, order.order_id)
            if not status:
                elapsed = time.time() - start
                if elapsed > cfg.max_wait_sec:
                    await self._finalize_order_cancel(symbol, "status_timeout")
                    return
                continue

            st = status["status"]
            cum = status["cumExecQty"]
            avgp = status["avgPrice"]

            async with self._lock:
                if symbol in self.active_orders:
                    o = self.active_orders[symbol]
                    o.filled_qty = cum
                    o.avg_price = avgp
                    if st == "PartiallyFilled":
                        o.state = "PARTIAL"
                    elif st == "Filled":
                        o.state = "FILLED"
                    elif st in ("Cancelled", "Rejected"):
                        o.state = st.upper()

            target = order.qty

            if cfg.require_full_fill and st == "Filled" and cum > 0:
                await self._create_or_update_position_from_fill(symbol, cum, avgp or order.price, order)
                await self._apply_sl_tp(symbol)
                async with self._lock:
                    self.active_orders.pop(symbol, None)
                return

            if (not cfg.require_full_fill) and target > 0 and cum / target >= cfg.min_partial_pct:
                await self._create_or_update_position_from_fill(symbol, cum, avgp or order.price, order)
                await self._apply_sl_tp(symbol)
                async with self._lock:
                    self.active_orders.pop(symbol, None)
                return

            if st in ("Cancelled", "Rejected"):
                async with self._lock:
                    self.active_orders.pop(symbol, None)
                return

            now = time.time()
            if now - last_reprice >= cfg.reprice_every_sec:
                last_reprice = now
                order.reprice_attempts += 1
                
                current_status = await self.api.get_order_status(symbol, order.order_id)
                if not current_status:
                    continue
                    
                current_order_status = current_status.get("status")
                if current_order_status not in ["New", "PartiallyFilled"]:
                    continue
                
                if order.reprice_attempts > settings.trading.max_reprice_attempts:
                    elapsed = now - start
                    if elapsed >= cfg.fallback_after_sec:
                        await self._fallback_to_market(symbol, order, cum)
                        return
                    else:
                        continue
                
                step = cfg.reprice_step_bps / 10000.0
                if order.side.upper() == "BUY":
                    new_price = order.price * (1 + step)
                else:
                    new_price = order.price * (1 - step)
                
                inst = await self.api.get_instrument_info(symbol)
                if inst:
                    _, new_price_q, _ = self.api.normalize_qty_price(symbol, inst, order.qty, new_price)
                    new_price = new_price_q
                
                verify_status = await self.api.get_order_status(symbol, order.order_id)
                if verify_status and verify_status["status"] in ["New", "PartiallyFilled"]:
                    amend = await self.api.amend_price(symbol, order.order_id, f"{new_price:.6f}")
                    
                    if amend.get("retCode") == 0:
                        async with self._lock:
                            if symbol in self.active_orders:
                                self.active_orders[symbol].price = new_price
                    elif amend.get("retCode") == 110001:
                        final_status = await self.api.get_order_status(symbol, order.order_id)
                        if final_status:
                            final_st = final_status.get("status")
                            final_cum = final_status.get("cumExecQty", 0)
                            
                            if final_st == "Filled" and final_cum > 0:
                                await self._create_or_update_position_from_fill(
                                    symbol, final_cum, 
                                    final_status.get("avgPrice") or order.price, 
                                    order
                                )
                                await self._apply_sl_tp(symbol)
                                async with self._lock:
                                    self.active_orders.pop(symbol, None)
                                return

            elapsed = now - start
            if elapsed >= cfg.fallback_after_sec:
                await self._fallback_to_market(symbol, order, cum)
                return

    async def _fallback_to_market(self, symbol: str, order: ActiveOrder, cum: float):
        """Fallback до маркет ордера"""
        mode = self.exec_cfg.fallback_mode.lower()
        if mode == "none":
            await self._finalize_order_cancel(symbol, "fallback_none")
            return
        if mode == "cancel":
            await self._finalize_order_cancel(symbol, "fallback_cancel")
            return
        if mode == "market":
            if self.exec_cfg.cancel_before_fallback and order.order_id:
                await self._finalize_order_cancel(symbol, "fallback_to_market_cancel")

            remaining = max(order.qty - cum, 0.0)
            qty_market = remaining if remaining > 0 else order.qty

            res = await self.api.place_order(
                symbol=symbol,
                side=self._api_side(order.side),
                qty=self._fmt_qty(qty_market),
                order_type="Market"
            )
            if res.get("retCode") == 0:
                await asyncio.sleep(0.4)
                mid_status = await self.api.get_order_status(symbol, res.get("result", {}).get("orderId", ""))
                if mid_status and mid_status.get("cumExecQty", 0) > 0:
                    await self._create_or_update_position_from_fill(
                        symbol,
                        mid_status["cumExecQty"],
                        mid_status["avgPrice"] or order.price,
                        order
                    )
                    await self._apply_sl_tp(symbol)

            async with self._lock:
                self.active_orders.pop(symbol, None)

    async def _finalize_order_cancel(self, symbol: str, reason: str):
        """Скасування ордера"""
        async with self._lock:
            order = self.active_orders.get(symbol)
        if order and order.order_id:
            await self.api.cancel_order(symbol, order.order_id)
        async with self._lock:
            self.active_orders.pop(symbol, None)
        logger.info(f"[EXECUTOR] ❌ Order cancelled {symbol} ({reason})")

    async def _create_or_update_position_from_fill(self, symbol: str, filled_qty: float, 
                                                   avg_price: float, order: ActiveOrder):
        """ОНОВЛЕНЕ створення позиції з адаптивним lifetime"""
        side_pos = "LONG" if order.side.upper() == "BUY" else "SHORT"
        sl = order.stop_loss
        tp = order.take_profit
        
        existing = self.positions.get(symbol)
        if existing and existing.status == "OPEN":
            existing.side = side_pos
            existing.qty = filled_qty
            existing.entry_price = avg_price
            existing.stop_loss = sl
            existing.take_profit = tp
            if order.signal_info:
                existing.meta_open = order.signal_info
            
            # Зберігаємо адаптивний lifetime
            if hasattr(order, 'adaptive_lifetime_sec'):
                existing.max_lifetime_sec = order.adaptive_lifetime_sec
            
            existing._position_updated = True
            logger.info(f"✅ [POSITION_UPDATED] {symbol} ID:{existing.position_id}")
        else:
            position = Position(
                symbol=symbol,
                side=side_pos,
                qty=filled_qty,
                entry_price=avg_price,
                stop_loss=sl,
                take_profit=tp,
                position_id=order.position_id,
                meta_open=order.signal_info
            )
            
            # Зберігаємо адаптивний lifetime
            if hasattr(order, 'adaptive_lifetime_sec'):
                position.max_lifetime_sec = order.adaptive_lifetime_sec
            
            self.positions[symbol] = position
            logger.info(f"✅ [POSITION_CREATED] {symbol} ID:{position.position_id}")
            position._position_updated = True
        
        self._log_trade("OPEN", symbol, side_pos, filled_qty, avg_price, 
                       sl, tp, "limit_filled", f"entryId={order.order_id or 'NA'}; signal={order.signal_info}")

    async def _apply_sl_tp(self, symbol: str, entry_price: float = None,
                          sl: float = None, tp: float = None, side: str = None):
        """Застосування SL/TP"""
        pos = self.positions.get(symbol)
        if not pos:
            return
        
        entry_price = entry_price or pos.entry_price
        sl = sl or pos.stop_loss
        tp = tp or pos.take_profit
        side = side or pos.side
        
        if not entry_price or not sl or not tp or not side:
            return

        decimals = self._get_price_decimal_places(symbol)
        sl_str = f"{sl:.{decimals}f}"
        tp_str = f"{tp:.{decimals}f}"
        
        try:
            r = await self.api.set_trading_stop(symbol, take_profit=tp_str, stop_loss=sl_str, position_idx=0)
            
            if r.get("retCode") == 0:
                logger.info(f"[EXECUTOR] ✅ SL/TP set {symbol} TP={tp_str} SL={sl_str}")
                pos.take_profit = tp
                pos.stop_loss = sl
                pos._position_updated = True
            elif r.get("retCode") == 34040:
                logger.debug(f"[EXECUTOR] SL/TP already set for {symbol}")
            else:
                logger.warning(f"[EXECUTOR] ⚠️ SL/TP fail {symbol}")
        except Exception as e:
            logger.error(f"[EXECUTOR] ❌ SL/TP exception for {symbol}: {e}")

    def _get_price_precision(self, symbol: str) -> float:
        """Точність ціни"""
        precision_map = {
            'BTCUSDT': 0.1, 'ETHUSDT': 0.01, 'BNBUSDT': 0.01,
            'SOLUSDT': 0.001, 'ADAUSDT': 0.0001, 'DOGEUSDT': 0.0001,
            'AVAXUSDT': 0.001, 'AAVEUSDT': 0.01, 'TRXUSDT': 0.00001
        }
        return precision_map.get(symbol, 0.0001)

    def _get_price_decimal_places(self, symbol: str) -> int:
        """Кількість десяткових знаків"""
        precision = self._get_price_precision(symbol)
        if precision >= 1: return 0
        if precision >= 0.1: return 1
        if precision >= 0.01: return 2
        if precision >= 0.001: return 3
        if precision >= 0.0001: return 4
        return 5

    def _api_side(self, side_str: str) -> str:
        """Конвертація сторони"""
        return "Buy" if side_str.upper() == "BUY" else "Sell"

    def _fmt_qty(self, qty: float) -> str:
        """Форматування кількості"""
        return f"{qty:.10f}".rstrip("0").rstrip(".")

    def get_stats(self) -> Dict[str, Any]:
        """РОЗШИРЕНА статистика з історією"""
        open_positions = {sym: pos for sym, pos in self.positions.items() if pos.status == "OPEN"}
        closed_positions = {sym: pos for sym, pos in self.positions.items() if pos.status == "CLOSED"}
        realized_pnl = sum(getattr(pos, 'realised_pnl', 0) for pos in self.positions.values())
        unrealized_pnl = sum(getattr(pos, 'unrealised_pnl', 0) for pos in open_positions.values())
        
        # Статистика з risk manager
        risk_stats = self.risk.get_statistics()
        
        return {
            "open_positions_count": len(open_positions),
            "closed_positions_count": len(closed_positions),
            "total_positions": len(self.positions),
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl,
            "risk_management": risk_stats,
            "positions_details": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "qty": pos.qty,
                    "entry_price": pos.entry_price,
                    "current_price": getattr(pos, 'current_price', 0),
                    "unrealised_pnl": getattr(pos, 'unrealised_pnl', 0),
                    "realised_pnl": getattr(pos, 'realised_pnl', 0),
                    "status": pos.status,
                    "close_reason": pos.close_reason if pos.status == "CLOSED" else "",
                    "max_lifetime_min": getattr(pos, 'max_lifetime_sec', 1800) / 60.0
                }
                for pos in self.positions.values()
            ]
        }