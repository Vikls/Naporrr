# trading/risk_manager.py
import time
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from collections import deque
from config.settings import settings
from utils.logger import logger

def _safe_float(x, default=0.0):
    """Безпечне перетворення в float"""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

class PositionHistory:
    """Історія позицій для адаптації параметрів"""
    def __init__(self, max_size: int = 100):
        self.history: deque = deque(maxlen=max_size)
    
    def add(self, symbol: str, side: str, pnl: float, close_reason: str, lifetime_sec: float):
        """Додати закриту позицію в історію"""
        self.history.append({
            "symbol": symbol,
            "side": side,
            "pnl": pnl,
            "close_reason": close_reason,
            "lifetime_sec": lifetime_sec,
            "timestamp": time.time(),
            "win": pnl > 0
        })
    
    def get_win_rate(self, symbol: Optional[str] = None, min_trades: int = 10) -> float:
        """Розрахувати win_rate"""
        if len(self.history) < min_trades:
            return 0.5  # Нейтральний fallback
        
        if symbol:
            trades = [t for t in self.history if t["symbol"] == symbol]
        else:
            trades = list(self.history)
        
        if len(trades) < min_trades:
            return 0.5
        
        wins = sum(1 for t in trades if t["win"])
        return wins / len(trades)
    
    def get_avg_lifetime(self, symbol: Optional[str] = None) -> float:
        """Середній час життя позицій"""
        if not self.history:
            return 0.0
        
        if symbol:
            trades = [t for t in self.history if t["symbol"] == symbol]
        else:
            trades = list(self.history)
        
        if not trades:
            return 0.0
        
        return sum(t["lifetime_sec"] for t in trades) / len(trades)
    
    def get_close_reason_stats(self) -> Dict[str, int]:
        """Статистика причин закриття"""
        stats = {}
        for trade in self.history:
            reason = trade["close_reason"]
            stats[reason] = stats.get(reason, 0) + 1
        return stats

class RiskManager:
    """🆕 ВИПРАВЛЕНИЙ Risk Manager з правильним розрахунком розміру позиції"""
    
    def __init__(self, api_manager=None):
        self.cfg = settings.risk
        self.tcfg = settings.trading
        self.history = PositionHistory(max_size=self.cfg.position_history_size)
        self.api_manager = api_manager
        
        logger.info(f"[RISK] Initialized with base_order_pct={self.tcfg.base_order_pct*100}%, leverage={self.tcfg.leverage}x, max_notional_pct={self.cfg.max_position_notional_pct*100}%")
    
    async def calc_base_qty(self, symbol: str, price: float, balance: float, api_manager=None) -> float:
        """
        🆕 ВИПРАВЛЕНИЙ розрахунок розміру позиції
        
        Формула: (balance × base_order_pct × leverage) / price
        """
        if price <= 0:
            logger.error(f"[RISK] Invalid price for {symbol}: {price}")
            return 0.0
        
        if balance <= 0:
            logger.error(f"[RISK] Invalid balance for {symbol}: {balance}")
            return 0.0
        
        # Використовуємо переданий api_manager або збережений
        api = api_manager or self.api_manager
        if not api:
            logger.error(f"[RISK] No API manager available for {symbol}")
            return 0.0
        
        # Отримуємо інформацію про інструмент
        inst_info = await api.get_instrument_info(symbol)
        if not inst_info:
            logger.error(f"[RISK] No instrument info for {symbol}")
            return 0.0
        
        # 🆕 ВИПРАВЛЕНА ФОРМУЛА: баланс × відсоток × плече
        if self.tcfg.base_order_pct > 0:
            # Розрахунок номіналу позиції
            position_notional = balance * self.tcfg.base_order_pct * self.tcfg.leverage
            logger.info(f"[RISK_CALC] {symbol}: Balance=${balance:.2f} × {self.tcfg.base_order_pct*100:.1f}% × {self.tcfg.leverage}x = ${position_notional:.2f} notional")
        else:
            logger.error("[RISK] base_order_pct should be > 0")
            return 0.0
        
        # Базова кількість
        base_qty = position_notional / price
        logger.info(f"[RISK_CALC] {symbol}: ${position_notional:.2f} / ${price:.2f} = {base_qty:.6f} base qty")
        
        # Нормалізація до кроку ціни та кількості
        normalized_qty, normalized_price, meta = api.normalize_qty_price(
            symbol, inst_info, base_qty, price
        )
        
        logger.info(f"[RISK_CALC] {symbol}: Normalized {base_qty:.6f} -> {normalized_qty:.6f}")
        
        # Перевірка мінімального номіналу
        lot_filter = inst_info.get("lotSizeFilter", {})
        min_notional = _safe_float(lot_filter.get("minOrderAmt", 0))
        calculated_notional = normalized_qty * normalized_price
        
        if min_notional > 0 and calculated_notional < min_notional:
            logger.warning(f"[RISK] {symbol}: Calculated notional ${calculated_notional:.2f} < min ${min_notional:.2f}")
            # Збільшуємо до мінімального номіналу
            required_qty = min_notional / normalized_price
            normalized_qty, _, _ = api.normalize_qty_price(
                symbol, inst_info, required_qty, normalized_price
            )
            logger.info(f"[RISK] {symbol}: Adjusted to min notional: {normalized_qty:.6f}")
        
        # 🆕 ВИПРАВЛЕНА перевірка максимального розміру позиції
        # Тепер використовуємо розрахований номінал, а не обмежуємо його
        max_allowed_notional = balance * self.cfg.max_position_notional_pct
        final_notional = normalized_qty * normalized_price
        
        # Логуємо для відладки
        logger.info(f"[RISK_DEBUG] {symbol}: Calculated notional=${final_notional:.2f}, Max allowed=${max_allowed_notional:.2f}")
        
        # Якщо розрахований номінал перевищує максимально дозволений, використовуємо максимальний
        if final_notional > max_allowed_notional:
            logger.warning(f"[RISK] {symbol}: Calculated notional ${final_notional:.2f} > max ${max_allowed_notional:.2f}")
            # Розраховуємо кількість на основі максимального номіналу
            adjusted_qty = max_allowed_notional / normalized_price
            normalized_qty, _, _ = api.normalize_qty_price(
                symbol, inst_info, adjusted_qty, normalized_price
            )
            logger.info(f"[RISK] {symbol}: Adjusted to max position: {normalized_qty:.6f}")
        
        final_notional = normalized_qty * normalized_price
        
        logger.info(f"[RISK_FINAL] {symbol}: Final Qty={normalized_qty:.6f}, Price=${normalized_price:.6f}, Notional=${final_notional:.2f}")
        
        return normalized_qty

    # ==================== АДАПТИВНИЙ LIFETIME ====================
    
    def get_adaptive_lifetime_seconds(self, symbol: str, volatility: float) -> int:
        """
        🆕 Розрахунок адаптивного lifetime на основі волатильності
        
        Args:
            symbol: Торговий символ
            volatility: Поточна волатільність (у %)
        
        Returns:
            Lifetime у секундах
        """
        base_lifetime_sec = self.cfg.base_position_lifetime_minutes * 60
        
        if not self.cfg.enable_adaptive_lifetime:
            return base_lifetime_sec
        
        # Визначаємо множник на основі волатільності
        if volatility < self.cfg.volatility_threshold_low:
            # Низька волатільність - збільшуємо час
            multiplier = self.cfg.low_volatility_lifetime_multiplier
            logger.debug(f"[ADAPTIVE_LIFETIME] {symbol}: LOW volatility {volatility:.3f}% -> x{multiplier}")
        elif volatility > self.cfg.volatility_threshold_high:
            # Висока волатільність - зменшуємо час
            multiplier = self.cfg.high_volatility_lifetime_multiplier
            logger.debug(f"[ADAPTIVE_LIFETIME] {symbol}: HIGH volatility {volatility:.3f}% -> x{multiplier}")
        else:
            # Нормальна волатільність
            multiplier = 1.0
        
        adaptive_lifetime = int(base_lifetime_sec * multiplier)
        
        # Обмеження (мінімум 5 хвилин, максимум 2 години)
        adaptive_lifetime = max(300, min(adaptive_lifetime, 7200))
        
        if multiplier != 1.0:
            logger.info(f"[ADAPTIVE_LIFETIME] {symbol}: {base_lifetime_sec}s -> {adaptive_lifetime}s "
                       f"(vol={volatility:.3f}%, x{multiplier:.2f})")
        
        return adaptive_lifetime
    
    # ==================== ДИНАМІЧНЕ TP/SL ====================
    
    def calc_sl_tp(self, side: str, entry_price: float, 
                   volatility_data: Dict[str, Any] = None, 
                   symbol: str = "") -> Tuple[float, float]:
        """
        🆕 Розрахунок SL/TP з динамічною адаптацією
        
        Args:
            side: "LONG" або "SHORT"
            entry_price: Ціна входу
            volatility_data: Дані про волатільність
            symbol: Торговий символ (для історичної статистики)
        
        Returns:
            (stop_loss, take_profit)
        """
        if entry_price <= 0:
            logger.error(f"[RISK] Invalid entry_price: {entry_price}")
            return entry_price, entry_price
        
        # Отримуємо волатільність
        range_position = 0.0
        atr_position = 0.0
        if volatility_data:
            range_position = volatility_data.get('range_position_lifetime', 0)
            atr_position = volatility_data.get('atr_position_lifetime', 0)
        
        logger.debug(f"[RISK] {symbol} {side}: entry={entry_price:.6f}, "
                    f"range_vol={range_position:.3f}%, atr_vol={atr_position:.3f}%")
        
        # Розраховуємо базові відсотки
        if self.cfg.enable_dynamic_tpsl and (range_position > 0 or atr_position > 0):
            sl_pct, tp_pct = self._calculate_dynamic_sltp(
                symbol, range_position, atr_position
            )
        else:
            # Fallback до мінімальних значень
            sl_pct = self.cfg.min_sl_pct
            tp_pct = self.cfg.min_tp_pct
            logger.debug(f"[RISK] Using fallback SL/TP: {sl_pct*100:.2f}% / {tp_pct*100:.2f}%")
        
        # Застосовуємо до ціни
        if side == "LONG":
            sl = entry_price * (1 - sl_pct)
            tp = entry_price * (1 + tp_pct)
        else:  # SHORT
            sl = entry_price * (1 + sl_pct)
            tp = entry_price * (1 - tp_pct)
        
        # Валідація
        sl, tp = self._validate_sltp_prices(side, entry_price, sl, tp)
        
        logger.info(f"[RISK] {symbol} {side}: Entry={entry_price:.6f} -> "
                   f"SL={sl:.6f} ({sl_pct*100:.2f}%), TP={tp:.6f} ({tp_pct*100:.2f}%)")
        
        return sl, tp
    
    def _calculate_dynamic_sltp(self, symbol: str, range_position: float, 
                               atr_position: float) -> Tuple[float, float]:
        """
        🆕 Динамічний розрахунок SL/TP з урахуванням волатільності та win_rate
        
        Returns:
            (sl_pct, tp_pct)
        """
        # 1. Базовий розрахунок на основі волатільності
        range_position = min(range_position, self.cfg.max_vol_used_pct)
        atr_position = min(atr_position, self.cfg.max_vol_used_pct)
        
        if range_position > 0:
            sl_from_range = (range_position * self.cfg.sl_vol_multiplier) / 100
            tp_from_range = (range_position * self.cfg.tp_vol_multiplier) / 100
        else:
            sl_from_range = self.cfg.min_sl_pct
            tp_from_range = self.cfg.min_tp_pct
        
        if atr_position > 0:
            sl_from_atr = (atr_position * 1.5) / 100
            tp_from_atr = (atr_position * 3.0) / 100
        else:
            sl_from_atr = self.cfg.min_sl_pct
            tp_from_atr = self.cfg.min_tp_pct
        
        # Беремо максимум
        sl_pct = max(sl_from_range, sl_from_atr, self.cfg.min_sl_pct)
        tp_pct = max(tp_from_range, tp_from_atr, self.cfg.min_tp_pct)
        
        # 2. Адаптація на основі win_rate (якщо є історія)
        if self.cfg.enable_dynamic_tpsl_ratio:
            win_rate = self.history.get_win_rate(
                symbol=symbol,
                min_trades=self.cfg.min_history_for_adaptation
            )
            
            if win_rate != 0.5:  # Є достатньо історії
                # Визначаємо цільове співвідношення TP/SL
                if win_rate > 0.6:
                    target_ratio = self.cfg.tpsl_ratio_high_winrate
                    logger.debug(f"[DYNAMIC_TPSL] {symbol}: High win_rate {win_rate:.2%} -> ratio {target_ratio}")
                elif win_rate < 0.4:
                    target_ratio = self.cfg.tpsl_ratio_low_winrate
                    logger.debug(f"[DYNAMIC_TPSL] {symbol}: Low win_rate {win_rate:.2%} -> ratio {target_ratio}")
                else:
                    target_ratio = self.cfg.tpsl_ratio_medium_winrate
                
                # Коригуємо TP для досягнення потрібного співвідношення
                tp_pct = sl_pct * target_ratio
                
                logger.info(f"[DYNAMIC_TPSL] {symbol}: win_rate={win_rate:.2%} -> "
                           f"SL={sl_pct*100:.2f}%, TP={tp_pct*100:.2f}% (ratio={target_ratio})")
        
        # 3. Обмеження
        sl_pct = min(sl_pct, self.cfg.max_sl_pct)
        tp_pct = min(tp_pct, self.cfg.max_tp_pct)
        
        return round(sl_pct, 4), round(tp_pct, 4)
    
    def _validate_sltp_prices(self, side: str, entry_price: float, 
                             sl: float, tp: float) -> Tuple[float, float]:
        """Валідація SL/TP цін"""
        if side == "LONG":
            if sl >= entry_price:
                sl = entry_price * 0.995
                logger.warning(f"[RISK] Corrected invalid SL for LONG: {sl:.6f}")
            if tp <= entry_price:
                tp = entry_price * 1.01
                logger.warning(f"[RISK] Corrected invalid TP for LONG: {tp:.6f}")
        else:  # SHORT
            if sl <= entry_price:
                sl = entry_price * 1.005
                logger.warning(f"[RISK] Corrected invalid SL for SHORT: {sl:.6f}")
            if tp >= entry_price:
                tp = entry_price * 0.99
                logger.warning(f"[RISK] Corrected invalid TP for SHORT: {tp:.6f}")
        
        # Мінімальна відстань між TP та SL
        min_distance = entry_price * 0.005
        if abs(tp - sl) < min_distance:
            if side == "LONG":
                tp = sl + min_distance
            else:
                tp = sl - min_distance
            logger.warning(f"[RISK] Adjusted TP to maintain minimum distance: {tp:.6f}")
        
        return sl, tp
    
    # ==================== TRAILING STOP ====================
    
    def update_trailing_stop(self, side: str, entry_price: float, current_sl: float,
                            current_price: float) -> Optional[float]:
        """
        🆕 Оновлення trailing stop
        
        Args:
            side: "LONG" або "SHORT"
            entry_price: Ціна входу
            current_sl: Поточний SL
            current_price: Поточна ціна
        
        Returns:
            Новий SL або None якщо не потрібно оновлювати
        """
        if not self.cfg.enable_trailing_stop:
            return None
        
        # Перевіряємо чи досягнуто активації
        if side == "LONG":
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct < self.cfg.trailing_stop_activation_pct:
                return None  # Недостатньо профіту для активації
            
            # Розраховуємо новий trailing SL
            new_sl = current_price * (1 - self.cfg.trailing_stop_distance_pct)
            
            # Оновлюємо тільки якщо новий SL вище поточного
            if new_sl > current_sl:
                logger.info(f"[TRAILING_STOP] LONG: {current_sl:.6f} -> {new_sl:.6f} "
                           f"(profit={profit_pct*100:.2f}%)")
                return new_sl
        
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct < self.cfg.trailing_stop_activation_pct:
                return None
            
            new_sl = current_price * (1 + self.cfg.trailing_stop_distance_pct)
            
            # Оновлюємо тільки якщо новий SL нижче поточного
            if new_sl < current_sl:
                logger.info(f"[TRAILING_STOP] SHORT: {current_sl:.6f} -> {new_sl:.6f} "
                           f"(profit={profit_pct*100:.2f}%)")
                return new_sl
        
        return None
    
    # ==================== ІНШІ МЕТОДИ ====================
    
    def can_open_new(self, open_positions_count: int) -> bool:
        """Перевірка можливості відкриття нової позиції"""
        return open_positions_count < self.cfg.max_open_positions
    
    def should_close_by_reverse(self, reverse_strength: int) -> bool:
        """Перевірка чи потрібно закрити позицію через реверс"""
        return reverse_strength >= self.tcfg.close_on_opposite_strength
    
    def add_to_history(self, symbol: str, side: str, pnl: float, 
                      close_reason: str, lifetime_sec: float):
        """Додати позицію в історію"""
        self.history.add(symbol, side, pnl, close_reason, lifetime_sec)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Отримати статистику ризик-менеджменту"""
        close_reasons = self.history.get_close_reason_stats()
        
        return {
            "total_trades": len(self.history.history),
            "overall_win_rate": self.history.get_win_rate(),
            "avg_lifetime_sec": self.history.get_avg_lifetime(),
            "close_reasons": close_reasons,
            "settings": {
                "adaptive_lifetime": self.cfg.enable_adaptive_lifetime,
                "dynamic_tpsl": self.cfg.enable_dynamic_tpsl,
                "trailing_stop": self.cfg.enable_trailing_stop,
                "base_lifetime_min": self.cfg.base_position_lifetime_minutes
            }
        }