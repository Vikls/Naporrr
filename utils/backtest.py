#!/usr/bin/env python3
"""
🔬 BACKTEST ENGINE - Аналіз сигналів на історичних даних
=========================================================

Завантажує 1-хв свічки з Bybit та накладає сигнали з signals.csv
для аналізу ефективності та оптимізації параметрів.

Використання:
    python -m utils.backtest
    python -m utils.backtest --hours 24 --optimize
    python -m utils.backtest --symbol BTCUSDT --verbose
"""

import os
import sys
import csv
import json
import argparse
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

# Додаємо кореневу директорію до path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


# =============================================================================
# 📊 DATA CLASSES
# =============================================================================

@dataclass
class Candle:
    """1-хвилинна свічка"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    
    @property
    def timestamp_ms(self) -> int:
        return int(self.timestamp.timestamp() * 1000)


@dataclass
class Signal:
    """Сигнал з signals.csv"""
    timestamp: datetime
    symbol: str
    action: str
    strength: int
    composite: float
    ema: float
    imbalance: float
    momentum: float
    bayesian: str
    large_orders: str
    frequency: str
    vol_confirm: str
    ohara_score: int
    reason: str
    accepted: bool


@dataclass
class SimulatedTrade:
    """Симульований трейд"""
    signal: Signal
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    max_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    duration_sec: float = 0.0
    
    # TP/SL що використовувались
    tp_pct: float = 0.0
    sl_pct: float = 0.0
    
    @property
    def is_winner(self) -> bool:
        return self.pnl_pct > 0


@dataclass
class BacktestConfig:
    """Конфігурація бектесту"""
    # Часові параметри
    hours_back: int = 12
    
    # TP/SL для симуляції (можна тестувати різні)
    tp_pct_options: List[float] = field(default_factory=lambda: [0.003, 0.005, 0.008, 0.01, 0.015])
    sl_pct_options: List[float] = field(default_factory=lambda: [0.002, 0.003, 0.005, 0.008, 0.01])
    
    # Максимальний час утримання (хвилини)
    max_hold_minutes_options: List[int] = field(default_factory=lambda: [30, 60, 120, 180])
    
    # Мінімальна сила сигналу для входу
    min_strength: int = 3
    
    # Фільтри сигналів для тестування
    test_filters: Dict[str, Any] = field(default_factory=dict)
    
    # Розмір позиції для розрахунку PnL
    position_size_usd: float = 100.0


# =============================================================================
# 📡 DATA LOADER - Завантаження свічок з Bybit
# =============================================================================

class BybitDataLoader:
    """Завантажувач історичних даних з Bybit"""
    
    BASE_URL = "https://api.bybit.com"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BacktestEngine/1.0'
        })
    
    def get_klines(
        self, 
        symbol: str, 
        interval: str = "1",  # 1 хвилина
        hours_back: int = 12,
        limit: int = 1000
    ) -> List[Candle]:
        """
        Завантажує свічки з Bybit API
        
        Args:
            symbol: Торгова пара (BTCUSDT)
            interval: Інтервал свічки ("1" = 1 хв)
            hours_back: Скільки годин назад
            limit: Максимум свічок за запит (макс 1000)
        """
        candles = []
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(hours=hours_back)).timestamp() * 1000)
        
        print(f"  📥 Завантаження {symbol} ({hours_back}h)...", end=" ")
        
        current_end = end_time
        total_fetched = 0
        
        while current_end > start_time:
            try:
                response = self.session.get(
                    f"{self.BASE_URL}/v5/market/kline",
                    params={
                        "category": "linear",
                        "symbol": symbol,
                        "interval": interval,
                        "end": current_end,
                        "limit": limit
                    },
                    timeout=10
                )
                
                data = response.json()
                
                if data.get("retCode") != 0:
                    print(f"❌ API Error: {data.get('retMsg')}")
                    break
                
                klines = data.get("result", {}).get("list", [])
                
                if not klines:
                    break
                
                for k in klines:
                    ts = datetime.fromtimestamp(int(k[0]) / 1000)
                    
                    if ts.timestamp() * 1000 < start_time:
                        continue
                    
                    candle = Candle(
                        timestamp=ts,
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                        turnover=float(k[6])
                    )
                    candles.append(candle)
                
                total_fetched += len(klines)
                
                # Наступний запит - раніше найстарішої свічки
                oldest_ts = min(int(k[0]) for k in klines)
                current_end = oldest_ts - 1
                
                if len(klines) < limit:
                    break
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        # Сортуємо по часу (від старих до нових)
        candles.sort(key=lambda c: c.timestamp)
        
        print(f"✅ {len(candles)} свічок")
        return candles
    
    def get_all_pairs_data(
        self, 
        symbols: List[str], 
        hours_back: int = 12
    ) -> Dict[str, List[Candle]]:
        """Завантажує дані для всіх пар"""
        all_data = {}
        
        print(f"\n📊 Завантаження даних за {hours_back} годин:")
        print("=" * 50)
        
        for symbol in symbols:
            candles = self.get_klines(symbol, hours_back=hours_back)
            if candles:
                all_data[symbol] = candles
        
        print("=" * 50)
        print(f"✅ Завантажено {len(all_data)} пар\n")
        
        return all_data


# =============================================================================
# 📜 SIGNAL PARSER - Парсинг signals.csv
# =============================================================================

class SignalParser:
    """Парсер сигналів з CSV файлу"""
    
    def __init__(self, signals_path: str = "logs/signals.csv"):
        self.signals_path = signals_path
    
    def parse(self, hours_back: int = 12) -> List[Signal]:
        """Парсить сигнали за останні N годин"""
        signals = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        if not os.path.exists(self.signals_path):
            print(f"❌ Файл {self.signals_path} не знайдено!")
            return signals
        
        with open(self.signals_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Пропускаємо заголовок
            
            for row in reader:
                try:
                    if len(row) < 15:
                        continue
                    
                    # Парсимо timestamp
                    ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    
                    if ts < cutoff_time:
                        continue
                    
                    signal = Signal(
                        timestamp=ts,
                        symbol=row[1],
                        action=row[2],
                        strength=int(row[3]),
                        composite=float(row[4]),
                        ema=float(row[5]),
                        imbalance=float(row[6]),
                        momentum=float(row[7]),
                        bayesian=row[8],
                        large_orders=row[9],
                        frequency=row[10],
                        vol_confirm=row[11],
                        ohara_score=int(row[12]),
                        reason=row[13],
                        accepted=row[14].upper() == "YES"
                    )
                    signals.append(signal)
                    
                except Exception as e:
                    continue  # Пропускаємо некоректні рядки
        
        print(f"📜 Завантажено {len(signals)} сигналів за {hours_back}h")
        return signals
    
    def get_actionable_signals(
        self, 
        signals: List[Signal], 
        min_strength: int = 3,
        actions: List[str] = ["BUY", "SELL"]
    ) -> List[Signal]:
        """Фільтрує сигнали до тих, що підходять для торгівлі"""
        filtered = [
            s for s in signals 
            if s.action in actions and s.strength >= min_strength
        ]
        print(f"🎯 Відфільтровано {len(filtered)} активних сигналів (strength >= {min_strength})")
        return filtered


# =============================================================================
# 🔬 TRADE SIMULATOR - Симуляція трейдів
# =============================================================================

class TradeSimulator:
    """Симулятор трейдів на історичних даних"""
    
    def __init__(self, candles_data: Dict[str, List[Candle]]):
        self.candles = candles_data
        self._candle_index = {}  # Індекс для швидкого пошуку
        self._build_index()
    
    def _build_index(self):
        """Будує індекс свічок по timestamp для швидкого пошуку"""
        for symbol, candles in self.candles.items():
            self._candle_index[symbol] = {
                c.timestamp_ms: i for i, c in enumerate(candles)
            }
    
    def _find_candle_at(self, symbol: str, timestamp: datetime) -> Tuple[Optional[Candle], int]:
        """Знаходить свічку для заданого часу"""
        if symbol not in self.candles:
            return None, -1
        
        target_ms = int(timestamp.timestamp() * 1000)
        candles = self.candles[symbol]
        
        # Бінарний пошук
        left, right = 0, len(candles) - 1
        
        while left <= right:
            mid = (left + right) // 2
            candle_ms = candles[mid].timestamp_ms
            
            if candle_ms <= target_ms < candle_ms + 60000:  # В межах хвилини
                return candles[mid], mid
            elif candle_ms < target_ms:
                left = mid + 1
            else:
                right = mid - 1
        
        # Повертаємо найближчу
        if left < len(candles):
            return candles[left], left
        
        return None, -1
    
    def simulate_trade(
        self,
        signal: Signal,
        tp_pct: float = 0.005,
        sl_pct: float = 0.003,
        max_hold_minutes: int = 120,
        position_size_usd: float = 100.0
    ) -> Optional[SimulatedTrade]:
        """
        Симулює один трейд на основі сигналу
        
        Args:
            signal: Сигнал для симуляції
            tp_pct: Take Profit у відсотках (0.01 = 1%)
            sl_pct: Stop Loss у відсотках
            max_hold_minutes: Максимальний час утримання
            position_size_usd: Розмір позиції в USD
        """
        symbol = signal.symbol
        
        if symbol not in self.candles:
            return None
        
        # Знаходимо свічку входу
        entry_candle, entry_idx = self._find_candle_at(symbol, signal.timestamp)
        
        if entry_candle is None:
            return None
        
        # Ціна входу - close свічки сигналу (або open наступної)
        entry_price = entry_candle.close
        entry_time = entry_candle.timestamp
        
        # Напрямок
        is_long = signal.action == "BUY"
        
        # Розраховуємо TP/SL рівні
        if is_long:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
        
        # Ініціалізуємо трейд
        trade = SimulatedTrade(
            signal=signal,
            entry_price=entry_price,
            entry_time=entry_time,
            tp_pct=tp_pct,
            sl_pct=sl_pct
        )
        
        max_profit = 0.0
        max_drawdown = 0.0
        
        # Проходимо по свічках після входу
        candles = self.candles[symbol]
        max_candles = max_hold_minutes  # 1 свічка = 1 хвилина
        
        for i in range(entry_idx + 1, min(entry_idx + max_candles + 1, len(candles))):
            candle = candles[i]
            
            # Перевіряємо High/Low для TP/SL
            if is_long:
                # Для LONG: high може бити TP, low може бити SL
                current_profit = (candle.high - entry_price) / entry_price
                current_drawdown = (entry_price - candle.low) / entry_price
                
                max_profit = max(max_profit, current_profit)
                max_drawdown = max(max_drawdown, current_drawdown)
                
                # SL Hit (перевіряємо першим - песимістичний сценарій)
                if candle.low <= sl_price:
                    trade.exit_price = sl_price
                    trade.exit_time = candle.timestamp
                    trade.exit_reason = "SL_HIT"
                    trade.pnl_pct = -sl_pct
                    break
                
                # TP Hit
                if candle.high >= tp_price:
                    trade.exit_price = tp_price
                    trade.exit_time = candle.timestamp
                    trade.exit_reason = "TP_HIT"
                    trade.pnl_pct = tp_pct
                    break
                    
            else:
                # Для SHORT: low може бити TP, high може бити SL
                current_profit = (entry_price - candle.low) / entry_price
                current_drawdown = (candle.high - entry_price) / entry_price
                
                max_profit = max(max_profit, current_profit)
                max_drawdown = max(max_drawdown, current_drawdown)
                
                # SL Hit
                if candle.high >= sl_price:
                    trade.exit_price = sl_price
                    trade.exit_time = candle.timestamp
                    trade.exit_reason = "SL_HIT"
                    trade.pnl_pct = -sl_pct
                    break
                
                # TP Hit
                if candle.low <= tp_price:
                    trade.exit_price = tp_price
                    trade.exit_time = candle.timestamp
                    trade.exit_reason = "TP_HIT"
                    trade.pnl_pct = tp_pct
                    break
        
        # Якщо не закрились по TP/SL - TIME_EXIT
        if trade.exit_price is None:
            last_candle = candles[min(entry_idx + max_candles, len(candles) - 1)]
            trade.exit_price = last_candle.close
            trade.exit_time = last_candle.timestamp
            trade.exit_reason = "TIME_EXIT"
            
            if is_long:
                trade.pnl_pct = (trade.exit_price - entry_price) / entry_price
            else:
                trade.pnl_pct = (entry_price - trade.exit_price) / entry_price
        
        # Розраховуємо фінальні метрики
        trade.max_profit_pct = max_profit
        trade.max_drawdown_pct = max_drawdown
        trade.pnl_usd = trade.pnl_pct * position_size_usd
        trade.duration_sec = (trade.exit_time - trade.entry_time).total_seconds()
        
        return trade


# =============================================================================
# 📈 BACKTEST ANALYZER - Аналіз результатів
# =============================================================================

class BacktestAnalyzer:
    """Аналізатор результатів бектесту"""
    
    def __init__(self, trades: List[SimulatedTrade]):
        self.trades = trades
    
    def get_summary(self) -> Dict[str, Any]:
        """Загальна статистика"""
        if not self.trades:
            return {"error": "No trades"}
        
        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]
        
        total_pnl = sum(t.pnl_usd for t in self.trades)
        total_pnl_pct = sum(t.pnl_pct for t in self.trades) * 100
        
        avg_winner = sum(t.pnl_usd for t in winners) / len(winners) if winners else 0
        avg_loser = sum(t.pnl_usd for t in losers) / len(losers) if losers else 0
        
        # По причинам виходу
        by_exit = defaultdict(list)
        for t in self.trades:
            by_exit[t.exit_reason].append(t)
        
        return {
            "total_trades": len(self.trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(self.trades) * 100,
            "total_pnl_usd": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "avg_pnl_usd": round(total_pnl / len(self.trades), 2),
            "avg_winner_usd": round(avg_winner, 2),
            "avg_loser_usd": round(avg_loser, 2),
            "profit_factor": abs(sum(t.pnl_usd for t in winners) / sum(t.pnl_usd for t in losers)) if losers and sum(t.pnl_usd for t in losers) != 0 else 0,
            "avg_duration_min": round(sum(t.duration_sec for t in self.trades) / len(self.trades) / 60, 1),
            "max_profit_seen": round(max(t.max_profit_pct for t in self.trades) * 100, 2),
            "max_drawdown_seen": round(max(t.max_drawdown_pct for t in self.trades) * 100, 2),
            "by_exit_reason": {
                reason: {
                    "count": len(trades),
                    "win_rate": len([t for t in trades if t.is_winner]) / len(trades) * 100 if trades else 0,
                    "avg_pnl": round(sum(t.pnl_usd for t in trades) / len(trades), 2) if trades else 0
                }
                for reason, trades in by_exit.items()
            }
        }
    
    def analyze_by_symbol(self) -> Dict[str, Dict]:
        """Аналіз по символах"""
        by_symbol = defaultdict(list)
        for t in self.trades:
            by_symbol[t.signal.symbol].append(t)
        
        results = {}
        for symbol, trades in by_symbol.items():
            winners = [t for t in trades if t.is_winner]
            results[symbol] = {
                "trades": len(trades),
                "win_rate": round(len(winners) / len(trades) * 100, 1) if trades else 0,
                "total_pnl": round(sum(t.pnl_usd for t in trades), 2),
                "avg_pnl": round(sum(t.pnl_usd for t in trades) / len(trades), 2) if trades else 0
            }
        
        return dict(sorted(results.items(), key=lambda x: x[1]["total_pnl"], reverse=True))
    
    def analyze_by_signal_params(self) -> Dict[str, Any]:
        """Аналіз по параметрах сигналу - що працює краще"""
        
        # По силі сигналу
        by_strength = defaultdict(list)
        for t in self.trades:
            by_strength[t.signal.strength].append(t)
        
        strength_analysis = {}
        for strength, trades in sorted(by_strength.items()):
            winners = [t for t in trades if t.is_winner]
            strength_analysis[f"strength_{strength}"] = {
                "count": len(trades),
                "win_rate": round(len(winners) / len(trades) * 100, 1) if trades else 0,
                "avg_pnl": round(sum(t.pnl_usd for t in trades) / len(trades), 2) if trades else 0
            }
        
        # По O'Hara score
        by_ohara = defaultdict(list)
        for t in self.trades:
            score_bucket = t.signal.ohara_score
            by_ohara[score_bucket].append(t)
        
        ohara_analysis = {}
        for score, trades in sorted(by_ohara.items()):
            winners = [t for t in trades if t.is_winner]
            ohara_analysis[f"ohara_{score}"] = {
                "count": len(trades),
                "win_rate": round(len(winners) / len(trades) * 100, 1) if trades else 0,
                "avg_pnl": round(sum(t.pnl_usd for t in trades) / len(trades), 2) if trades else 0
            }
        
        # По імбалансу (buckets)
        imb_buckets = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
        imbalance_analysis = {}
        
        for low, high in imb_buckets:
            trades = [t for t in self.trades if low <= abs(t.signal.imbalance) < high]
            if trades:
                winners = [t for t in trades if t.is_winner]
                imbalance_analysis[f"imb_{low}-{high}"] = {
                    "count": len(trades),
                    "win_rate": round(len(winners) / len(trades) * 100, 1),
                    "avg_pnl": round(sum(t.pnl_usd for t in trades) / len(trades), 2)
                }
        
        # По моментуму (buckets)
        mom_buckets = [(0, 30), (30, 50), (50, 70), (70, 85), (85, 100)]
        momentum_analysis = {}
        
        for low, high in mom_buckets:
            trades = [t for t in self.trades if low <= abs(t.signal.momentum) < high]
            if trades:
                winners = [t for t in trades if t.is_winner]
                momentum_analysis[f"mom_{low}-{high}"] = {
                    "count": len(trades),
                    "win_rate": round(len(winners) / len(trades) * 100, 1),
                    "avg_pnl": round(sum(t.pnl_usd for t in trades) / len(trades), 2)
                }
        
        return {
            "by_strength": strength_analysis,
            "by_ohara_score": ohara_analysis,
            "by_imbalance": imbalance_analysis,
            "by_momentum": momentum_analysis
        }
    
    def find_optimal_filters(self) -> Dict[str, Any]:
        """Знаходить оптимальні фільтри для входу"""
        
        best_filters = {
            "min_imbalance": 0,
            "min_momentum": 0,
            "max_momentum": 100,
            "min_ohara": 0,
            "min_strength": 3,
            "best_win_rate": 0,
            "best_pnl": -float('inf')
        }
        
        # Тестуємо різні комбінації
        for min_imb in [5, 8, 10, 12, 15, 18, 20]:
            for min_mom in [30, 40, 50, 60]:
                for max_mom in [80, 85, 90, 95]:
                    for min_ohara in [3, 4, 5, 6]:
                        
                        filtered = [
                            t for t in self.trades
                            if abs(t.signal.imbalance) >= min_imb
                            and abs(t.signal.momentum) >= min_mom
                            and abs(t.signal.momentum) <= max_mom
                            and t.signal.ohara_score >= min_ohara
                        ]
                        
                        if len(filtered) < 5:  # Мінімум 5 трейдів
                            continue
                        
                        winners = [t for t in filtered if t.is_winner]
                        win_rate = len(winners) / len(filtered) * 100
                        total_pnl = sum(t.pnl_usd for t in filtered)
                        
                        # Оптимізуємо по PnL (або можна по win_rate)
                        if total_pnl > best_filters["best_pnl"]:
                            best_filters = {
                                "min_imbalance": min_imb,
                                "min_momentum": min_mom,
                                "max_momentum": max_mom,
                                "min_ohara": min_ohara,
                                "min_strength": 3,
                                "trades_count": len(filtered),
                                "best_win_rate": round(win_rate, 1),
                                "best_pnl": round(total_pnl, 2)
                            }
        
        return best_filters


# =============================================================================
# 🔧 PARAMETER OPTIMIZER - Оптимізація TP/SL
# =============================================================================

class ParameterOptimizer:
    """Оптимізатор параметрів TP/SL"""
    
    def __init__(
        self, 
        simulator: TradeSimulator, 
        signals: List[Signal],
        config: BacktestConfig
    ):
        self.simulator = simulator
        self.signals = signals
        self.config = config
    
    def optimize_tpsl(self) -> Dict[str, Any]:
        """Знаходить оптимальні TP/SL"""
        
        results = []
        
        print("\n🔧 Оптимізація TP/SL параметрів...")
        print("=" * 60)
        
        total_combinations = (
            len(self.config.tp_pct_options) * 
            len(self.config.sl_pct_options) *
            len(self.config.max_hold_minutes_options)
        )
        current = 0
        
        for tp in self.config.tp_pct_options:
            for sl in self.config.sl_pct_options:
                for hold_min in self.config.max_hold_minutes_options:
                    current += 1
                    
                    # Симулюємо всі трейди з цими параметрами
                    trades = []
                    for signal in self.signals:
                        trade = self.simulator.simulate_trade(
                            signal, 
                            tp_pct=tp, 
                            sl_pct=sl,
                            max_hold_minutes=hold_min,
                            position_size_usd=self.config.position_size_usd
                        )
                        if trade:
                            trades.append(trade)
                    
                    if not trades:
                        continue
                    
                    winners = [t for t in trades if t.is_winner]
                    total_pnl = sum(t.pnl_usd for t in trades)
                    
                    results.append({
                        "tp_pct": tp,
                        "sl_pct": sl,
                        "max_hold_min": hold_min,
                        "rr_ratio": round(tp / sl, 2),
                        "trades": len(trades),
                        "win_rate": round(len(winners) / len(trades) * 100, 1),
                        "total_pnl": round(total_pnl, 2),
                        "avg_pnl": round(total_pnl / len(trades), 2),
                        "tp_hits": len([t for t in trades if t.exit_reason == "TP_HIT"]),
                        "sl_hits": len([t for t in trades if t.exit_reason == "SL_HIT"]),
                        "time_exits": len([t for t in trades if t.exit_reason == "TIME_EXIT"])
                    })
                    
                    # Прогрес
                    if current % 10 == 0:
                        print(f"  Progress: {current}/{total_combinations}", end="\r")
        
        print(f"  ✅ Протестовано {len(results)} комбінацій")
        
        # Сортуємо по PnL
        results.sort(key=lambda x: x["total_pnl"], reverse=True)
        
        return {
            "best_params": results[0] if results else None,
            "top_5": results[:5],
            "worst_5": results[-5:] if len(results) >= 5 else results
        }


# =============================================================================
# 📋 REPORT GENERATOR - Генерація звітів
# =============================================================================

class ReportGenerator:
    """Генератор звітів"""
    
    @staticmethod
    def print_summary(summary: Dict, title: str = "BACKTEST SUMMARY"):
        """Друкує саммарі"""
        print(f"\n{'=' * 60}")
        print(f"📊 {title}")
        print(f"{'=' * 60}")
        
        print(f"\n📈 Загальна статистика:")
        print(f"  • Всього трейдів: {summary['total_trades']}")
        print(f"  • Виграшних: {summary['winners']} ({summary['win_rate']:.1f}%)")
        print(f"  • Програшних: {summary['losers']}")
        print(f"  • Total PnL: ${summary['total_pnl_usd']:.2f} ({summary['total_pnl_pct']:.2f}%)")
        print(f"  • Avg PnL/trade: ${summary['avg_pnl_usd']:.2f}")
        print(f"  • Avg Winner: ${summary['avg_winner_usd']:.2f}")
        print(f"  • Avg Loser: ${summary['avg_loser_usd']:.2f}")
        print(f"  • Profit Factor: {summary['profit_factor']:.2f}")
        print(f"  • Avg Duration: {summary['avg_duration_min']:.1f} min")
        
        print(f"\n📉 По причинах виходу:")
        for reason, stats in summary.get("by_exit_reason", {}).items():
            print(f"  • {reason}: {stats['count']} trades, WR: {stats['win_rate']:.1f}%, Avg: ${stats['avg_pnl']:.2f}")
    
    @staticmethod
    def print_by_symbol(by_symbol: Dict):
        """Друкує статистику по символах"""
        print(f"\n📊 По символах:")
        print("-" * 50)
        print(f"{'Symbol':<12} {'Trades':>8} {'Win Rate':>10} {'Total PnL':>12} {'Avg PnL':>10}")
        print("-" * 50)
        
        for symbol, stats in by_symbol.items():
            wr = stats['win_rate']
            pnl = stats['total_pnl']
            pnl_color = "+" if pnl >= 0 else ""
            print(f"{symbol:<12} {stats['trades']:>8} {wr:>9.1f}% {pnl_color}${pnl:>10.2f} ${stats['avg_pnl']:>9.2f}")
    
    @staticmethod
    def print_signal_analysis(analysis: Dict):
        """Друкує аналіз параметрів сигналу"""
        print(f"\n🔍 Аналіз параметрів сигналу:")
        
        for category, data in analysis.items():
            print(f"\n  {category.replace('_', ' ').title()}:")
            for key, stats in data.items():
                wr = stats.get('win_rate', 0)
                marker = "✅" if wr >= 55 else "⚠️" if wr >= 45 else "❌"
                print(f"    {marker} {key}: {stats['count']} trades, WR: {wr:.1f}%, Avg: ${stats['avg_pnl']:.2f}")
    
    @staticmethod
    def print_optimal_filters(filters: Dict):
        """Друкує оптимальні фільтри"""
        print(f"\n🎯 ОПТИМАЛЬНІ ФІЛЬТРИ ДЛЯ settings.py:")
        print("=" * 50)
        print(f"  min_imbalance_for_entry: {filters['min_imbalance']}")
        print(f"  min_momentum_for_entry: {filters['min_momentum']}")
        print(f"  max_momentum_for_entry: {filters['max_momentum']}")
        print(f"  min_ohara_for_entry: {filters['min_ohara']}")
        print(f"  min_strength_for_action: {filters['min_strength']}")
        print("-" * 50)
        print(f"  📊 Очікуваний результат:")
        print(f"     Трейдів: {filters.get('trades_count', 'N/A')}")
        print(f"     Win Rate: {filters['best_win_rate']}%")
        print(f"     Total PnL: ${filters['best_pnl']:.2f}")
    
    @staticmethod
    def print_tpsl_optimization(results: Dict):
        """Друкує результати оптимізації TP/SL"""
        print(f"\n⚙️ ОПТИМАЛЬНІ TP/SL ПАРАМЕТРИ:")
        print("=" * 60)
        
        best = results.get("best_params")
        if best:
            print(f"\n  🥇 НАЙКРАЩІ ПАРАМЕТРИ:")
            print(f"     TP: {best['tp_pct']*100:.2f}%")
            print(f"     SL: {best['sl_pct']*100:.2f}%")
            print(f"     Max Hold: {best['max_hold_min']} min")
            print(f"     R:R Ratio: {best['rr_ratio']}")
            print(f"     ---")
            print(f"     Trades: {best['trades']}")
            print(f"     Win Rate: {best['win_rate']}%")
            print(f"     Total PnL: ${best['total_pnl']:.2f}")
            print(f"     TP Hits: {best['tp_hits']}, SL Hits: {best['sl_hits']}, Time: {best['time_exits']}")
        
        print(f"\n  📊 TOP 5 КОМБІНАЦІЙ:")
        print("-" * 60)
        print(f"{'TP%':>6} {'SL%':>6} {'Hold':>6} {'RR':>5} {'WR%':>6} {'PnL':>10}")
        print("-" * 60)
        
        for r in results.get("top_5", []):
            print(f"{r['tp_pct']*100:>5.2f}% {r['sl_pct']*100:>5.2f}% {r['max_hold_min']:>5}m {r['rr_ratio']:>5.1f} {r['win_rate']:>5.1f}% ${r['total_pnl']:>9.2f}")
    
    @staticmethod
    def save_report(data: Dict, filepath: str = "logs/backtest_report.json"):
        """Зберігає звіт у JSON"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n💾 Звіт збережено: {filepath}")


# =============================================================================
# 🚀 MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="🔬 Backtest Engine")
    parser.add_argument("--hours", type=int, default=12, help="Години назад для аналізу")
    parser.add_argument("--symbol", type=str, help="Конкретний символ (або всі)")
    parser.add_argument("--optimize", action="store_true", help="Оптимізувати TP/SL")
    parser.add_argument("--min-strength", type=int, default=3, help="Мінімальна сила сигналу")
    parser.add_argument("--verbose", action="store_true", help="Детальний вивід")
    parser.add_argument("--save", action="store_true", help="Зберегти звіт")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🔬 BACKTEST ENGINE v1.0")
    print("=" * 60)
    
    # Конфігурація
    config = BacktestConfig(
        hours_back=args.hours,
        min_strength=args.min_strength
    )
    
    # Символи
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = settings.pairs.trade_pairs
    
    print(f"\n⚙️ Конфігурація:")
    print(f"  • Період: {args.hours} годин")
    print(f"  • Символи: {len(symbols)}")
    print(f"  • Мін.сила: {args.min_strength}")
    
    # 1.Завантажуємо свічки
    loader = BybitDataLoader()
    candles_data = loader.get_all_pairs_data(symbols, hours_back=args.hours)
    
    if not candles_data:
        print("❌ Не вдалося завантажити дані!")
        return
    
    # 2.Парсимо сигнали
    signal_parser = SignalParser()
    all_signals = signal_parser.parse(hours_back=args.hours)
    
    # Фільтруємо до активних
    active_signals = signal_parser.get_actionable_signals(
        all_signals, 
        min_strength=args.min_strength
    )
    
    if not active_signals:
        print("❌ Немає активних сигналів для аналізу!")
        return
    
    # 3.Симулюємо трейди
    print(f"\n🔄 Симуляція трейдів...")
    simulator = TradeSimulator(candles_data)
    
    trades = []
    for signal in active_signals:
        trade = simulator.simulate_trade(
            signal,
            tp_pct=settings.risk.min_tp_pct,
            sl_pct=settings.risk.min_sl_pct,
            max_hold_minutes=settings.risk.base_position_lifetime_minutes,
            position_size_usd=config.position_size_usd
        )
        if trade:
            trades.append(trade)
    
    print(f"✅ Симульовано {len(trades)} трейдів")
    
    if not trades:
        print("❌ Жоден трейд не симульовано!")
        return
    
    # 4.Аналізуємо результати
    analyzer = BacktestAnalyzer(trades)
    
    # Основна статистика
    summary = analyzer.get_summary()
    ReportGenerator.print_summary(summary)
    
    # По символах
    by_symbol = analyzer.analyze_by_symbol()
    ReportGenerator.print_by_symbol(by_symbol)
    
    # По параметрах сигналу
    signal_analysis = analyzer.analyze_by_signal_params()
    ReportGenerator.print_signal_analysis(signal_analysis)
    
    # Оптимальні фільтри
    optimal_filters = analyzer.find_optimal_filters()
    ReportGenerator.print_optimal_filters(optimal_filters)
    
    # 5.Оптимізація TP/SL (якщо потрібно)
    if args.optimize:
        optimizer = ParameterOptimizer(simulator, active_signals, config)
        tpsl_results = optimizer.optimize_tpsl()
        ReportGenerator.print_tpsl_optimization(tpsl_results)
    
    # 6.Зберігаємо звіт
    if args.save:
        report_data = {
            "config": {
                "hours": args.hours,
                "symbols": symbols,
                "min_strength": args.min_strength
            },
            "summary": summary,
            "by_symbol": by_symbol,
            "signal_analysis": signal_analysis,
            "optimal_filters": optimal_filters
        }
        
        if args.optimize:
            report_data["tpsl_optimization"] = tpsl_results
        
        ReportGenerator.save_report(report_data)
    
    print("\n" + "=" * 60)
    print("✅ Бектест завершено!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()