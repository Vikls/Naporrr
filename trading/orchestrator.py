# trading/orchestrator.py
import asyncio
import time
import json
from typing import Optional, Dict
from utils.logger import logger
from config.settings import settings
from data.storage import DataStorage, Position
from analysis.imbalance import ImbalanceAnalyzer
from analysis.volume import VolumeAnalyzer
from analysis.signals import SignalGenerator
from trading.executor import TradeExecutor

class TradingOrchestrator:
    """Оновлений Orchestrator з O'Hara методами"""
    
    def __init__(self, storage: DataStorage, imbalance_analyzer: ImbalanceAnalyzer,
                 volume_analyzer: VolumeAnalyzer, signal_generator: SignalGenerator, executor: TradeExecutor):
        self.storage = storage
        self.imb = imbalance_analyzer
        self.vol = volume_analyzer
        self.sig_gen = signal_generator
        self.executor = executor
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_open_ts: Dict[str, float] = {}
        self._last_close_ts: Dict[str, float] = {}
        self._last_signal: Dict[str, Dict] = {}
        self._last_trade_time: Dict[str, float] = {}
        self._reverse_pending: Dict[str, bool] = {}
        
        self._position_status_cache: Dict[str, Dict] = {}
        self._cache_ttl = 3.0

    async def _fast_check_exchange_position_status(self, symbol: str) -> bool:
        """ШВИДКА перевірка статусу"""
        current_time = time.time()
        
        if symbol in self._position_status_cache:
            cached = self._position_status_cache[symbol]
            if current_time - cached['timestamp'] < self._cache_ttl:
                return cached['is_open']
        
        try:
            async with asyncio.timeout(5):
                pos = self.storage.get_position(symbol)
                if not pos:
                    result = True
                else:
                    result = pos.status != "OPEN"
                
                self._position_status_cache[symbol] = {
                    'is_open': result,
                    'timestamp': current_time
                }
                return result
                
        except asyncio.TimeoutError:
            logger.error(f"❌ [FAST_STATUS_CHECK] Timeout for {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ [FAST_STATUS_CHECK] Error for {symbol}: {e}")
            return True

    async def start(self):
        if self._task:
            return
        self._running = True
        self._task = asyncio.create_task(self._optimized_loop())
        logger.info("✅ [ORCH] Trading orchestrator started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("✅ [ORCH] Trading orchestrator cancelled")
            except Exception as e:
                logger.error(f"❌ [ORCH] error during stop: {e}")
            self._task = None
        logger.info("✅ [ORCH] Trading orchestrator stopped")

    async def _optimized_loop(self):
        """Оптимізований головний цикл"""
        interval = settings.trading.decision_interval_sec
        symbol_batches = self._create_symbol_batches(settings.pairs.trade_pairs, batch_size=3)
        batch_index = 0
        
        while self._running:
            start_iter = time.time()
            try:
                current_batch = symbol_batches[batch_index]
                await self._process_symbol_batch(current_batch)
                batch_index = (batch_index + 1) % len(symbol_batches)
            except Exception as e:
                logger.error(f"❌ [ORCH] iteration error: {e}", exc_info=True)
            
            elapsed = time.time() - start_iter
            await asyncio.sleep(max(0.0, interval - elapsed))

    def _create_symbol_batches(self, symbols: list, batch_size: int = 3) -> list:
        """Розділення символів на батчі"""
        return [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

    async def _process_symbol_batch(self, symbols: list):
        """Обробка батчу символів"""
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._process_single_symbol(symbol))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_single_symbol(self, symbol: str):
        """Обробка одного символу з O'Hara методами"""
        try:
            can_process = await self._fast_check_exchange_position_status(symbol)
            if not can_process:
                return

            ob = self.storage.get_order_book(symbol)
            if not ob:
                return

            # Обчислення сигналів
            vol_data = self.vol.compute(symbol)
            imb_data = self.imb.compute(symbol)
            
            # Оновлюємо кеш волатильності
            self.imb.update_volatility_cache(symbol, vol_data)
            
            # 🆕 O'HARA METHOD 7: Spread calculation
            spread_bps = self.storage.get_current_spread_bps(symbol)
            if spread_bps is None and ob and ob.best_bid and ob.best_ask and ob.best_bid > 0 and ob.best_ask > 0:
                spread_bps = (ob.best_ask - ob.best_bid) / ob.best_bid * 10000
                if spread_bps < 0 or spread_bps > 1000:
                    spread_bps = None
            
            # Update spread monitor
            if spread_bps is not None and ob and ob.best_bid and ob.best_ask:
                self.sig_gen.spread_monitor.update(symbol, ob.best_bid, ob.best_ask)

            sig = self.sig_gen.generate(symbol, imb_data, vol_data, spread_bps)
            self._last_signal[symbol] = sig
            
            # Паралельна обробка
            await asyncio.gather(
                self._optimized_maybe_close(symbol, sig, ob, vol_data),
                self._optimized_maybe_open(symbol, sig, ob, vol_data),
                return_exceptions=True
            )
                
        except Exception as e:
            logger.error(f"❌ [ORCH] Error processing {symbol}: {e}")

    async def _optimized_maybe_open(self, symbol: str, sig: Dict, ob, vol_data: Dict):
        """Оновлена логіка відкриття з O'Hara фільтрами"""
        can_open = await self._fast_check_exchange_position_status(symbol)
        if not can_open:
            return

        if not self._quick_open_checks(symbol, sig):
            return

        action = sig.get("action", "HOLD")
        strength = sig.get("strength", 0)
        
        if action == "HOLD" or strength < self.executor.tcfg.entry_signal_min_strength:
            return
        
        # 🆕 O'HARA FILTER: Check spread risk
        factors = sig.get("factors", {})
        if factors:
            spread_factor = factors.get("spread", 0)
            if spread_factor < -0.4:  # Дуже негативний spread factor
                logger.warning(f"[OHARA_FILTER] {symbol}: Spread too wide, avoiding trade")
                return

        is_reverse, double_size = await self._fast_determine_reverse(symbol, action)
        
        if symbol in self.executor.active_orders and not is_reverse:
            return

        best_bid = getattr(ob, "best_bid", None)
        best_ask = getattr(ob, "best_ask", None)
        if best_bid and best_ask:
            mid = (best_bid + best_ask) / 2
        else:
            mid = best_bid or best_ask
        
        if mid is None:
            return

        effective_action = action
        if self.executor.tcfg.reverse_signals:
            effective_action = "SELL" if action == "BUY" else ("BUY" if action == "SELL" else action)

        signal_info = await self._fast_create_signal_info(symbol, action, strength, sig, is_reverse)

        logger.info(f"[ORCH] 🎯 Opening {symbol}: {effective_action} with signal {signal_info}")
        
        await self.executor.open_position_limit(
            symbol=symbol,
            direction=effective_action,
            ref_price=mid,
            best_bid=best_bid,
            best_ask=best_ask,
            is_reversed=is_reverse,
            double_size=double_size,
            signal_info=signal_info,
            volatility_data=vol_data
        )
        
        self._last_open_ts[symbol] = time.time()
        if action != "HOLD":
            self._last_trade_time[symbol] = time.time()

    def _quick_open_checks(self, symbol: str, sig: Dict) -> bool:
        """ШВИДКІ перевірки з O'Hara фільтрами"""
        current_time = time.time()
        
        last_trade_time = self._last_trade_time.get(symbol, 0)
        if current_time - last_trade_time < self.executor.tcfg.min_time_between_trades_sec:
            return False

        last_close = self._last_close_ts.get(symbol, 0)
        if current_time - last_close < self.executor.tcfg.reopen_cooldown_sec:
            return False

        # 🆕 O'HARA FILTER: Check O'Hara score
        ohara_score = sig.get("ohara_score", 0)
        if settings.ohara.enable_combined_ohara_score and ohara_score < settings.ohara.min_ohara_score_for_trade:
            logger.debug(f"[OHARA_FILTER] {symbol}: O'Hara score too low ({ohara_score}/{settings.ohara.min_ohara_score_for_trade})")
            return False

        if self.executor.tcfg.enable_aggressive_filtering:
            raw_values = sig.get('factors', {}).get('raw_values', {})
            momentum_score = raw_values.get('momentum_score', 0)
            if abs(momentum_score) > 90 and sig.get('strength', 0) >= 4:
                return False

        return True

    async def _fast_determine_reverse(self, symbol: str, action: str) -> tuple:
        """Швидке визначення реверсу"""
        current_pos = self.storage.get_position(symbol)
        is_reverse = False
        double_size = False
        
        if current_pos and current_pos.status == "OPEN":
            if (current_pos.side == "LONG" and action == "SELL") or \
               (current_pos.side == "SHORT" and action == "BUY"):
                is_reverse = True
                double_size = self.executor.tcfg.reverse_double_size
                logger.info(f"[REVERSE] 🔄 {symbol}: closing {current_pos.side} and opening {action}")
                self._reverse_pending[symbol] = True

        return is_reverse, double_size

    async def _fast_create_signal_info(self, symbol: str, action: str, 
                                      strength: int, sig: Dict, is_reverse: bool) -> str:
        """Швидке створення інформації про сигнал з O'Hara даними"""
        try:
            signal_parts = []
            if is_reverse:
                signal_parts.append("REVERSE")
            display_action = "SELL" if action == "BUY" else "BUY" if action == "SELL" else action
            if self.executor.tcfg. reverse_signals:
                signal_parts.append(f"{display_action.upper()}{strength}")
            else:
                signal_parts.append(f"{action.upper()}{strength}")

            factors = sig.get('factors', {})
            raw_values = factors.get('raw_values', {})
            if raw_values:
                imb_score = raw_values.get('imbalance_score', 0)
                mom_score = raw_values.get('momentum_score', 0)
                ohara_score = sig.get('ohara_score', 0)
                signal_parts.append(f"(imb:{imb_score:.0f},mom:{mom_score:.0f},oh:{ohara_score})")

            return " ".join(signal_parts)

        except Exception as e:
            logger.error(f"❌ [FAST_SIGNAL] {symbol}: {e}")
            return f"{action.upper()}{strength}" + (" (reverse)" if is_reverse else "")

    async def _optimized_maybe_close(self, symbol: str, sig: Dict, ob, vol_data: Dict):
        """Оновлена логіка закриття з адаптивним lifetime"""
        can_process = await self._fast_check_exchange_position_status(symbol)
        if not can_process:
            return
            
        pos = self.storage.get_position(symbol)
        if not pos or pos.status != "OPEN":
            return

        if not self._quick_close_checks(symbol, pos):
            return

        current_time = time.time()
        
        # Адаптивний lifetime
        if hasattr(pos, 'max_lifetime_sec') and pos.max_lifetime_sec > 0:
            max_life = pos.max_lifetime_sec
        else:
            current_volatility = vol_data.get('recent_volatility', 0.1)
            max_life = self.executor.risk.get_adaptive_lifetime_seconds(symbol, current_volatility)
        
        # Пріоритетна перевірка TIME_EXIT
        if current_time - pos.timestamp > max_life:
            lifetime_min = (current_time - pos.timestamp) / 60.0
            logger.info(f"[ORCH] ⏰ Closing {symbol} {pos.side} due to TIME_EXIT "
                       f"({lifetime_min:.1f}min > {max_life/60:.1f}min)")
            await self.executor.close_position(symbol, reason="TIME_EXIT")
            self._last_close_ts[symbol] = current_time
            return

        # Перевірка реверсу
        if symbol in self._reverse_pending and self._reverse_pending[symbol]:
            logger.info(f"[ORCH] 🔄 Closing {symbol} {pos.side} for REVERSE")
            await self.executor.close_position(symbol, reason="REVERSE")
            self._reverse_pending.pop(symbol, None)
            self._last_close_ts[symbol] = current_time
            return

        # Перевірка протилежного сигналу
        action = sig.get("action", "HOLD")
        strength = sig.get("strength", 0)
        opposite_strength_req = self.executor.tcfg.close_on_opposite_strength

        if (pos.side == "LONG" and action == "SELL" and strength >= opposite_strength_req) or \
           (pos.side == "SHORT" and action == "BUY" and strength >= opposite_strength_req):
            logger.info(f"[ORCH] 🔒 Closing {symbol} {pos.side} due to opposite signal")
            await self.executor.close_position(symbol, reason="opp_signal")
            self._last_close_ts[symbol] = current_time
            return

    def _quick_close_checks(self, symbol: str, pos: Position) -> bool:
        """ШВИДКІ перевірки для закриття"""
        current_time = time.time()
        
        if current_time - pos.timestamp < self.executor.tcfg.min_position_hold_time_sec:
            return False
            
        return True

    def get_last_signal(self, symbol: str) -> Optional[Dict]:
        return self._last_signal.get(symbol)

    async def close_all(self, reason: str = "force_close"):
        """Закрити всі позиції"""
        symbols = list(self.storage.positions.keys())
        close_tasks = []
        for sym in symbols:
            pos = self.storage.positions[sym]
            if pos.status == "OPEN":
                task = asyncio.create_task(self.executor.close_position(sym, reason=reason))
                close_tasks.append(task)
        
        await asyncio.gather(*close_tasks, return_exceptions=True)
        logger.info(f"[ORCH] 🔒 Force closed {len(close_tasks)} positions")