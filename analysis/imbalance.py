# analysis/imbalance.py
import time
import statistics
from collections import deque
from typing import Dict, Any, List
from config.settings import settings
from data.storage import DataStorage
from utils.logger import logger

class ImbalanceAnalyzer:
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.cfg = settings.imbalance
        self.adaptive_cfg = settings.adaptive
        self.pairs_cfg = settings.pairs
        self.ohara_cfg = settings.ohara
        self.symbol_stats = {}
        self.last_imbalance = {}
        self.historical_imbalance = {}
        self.imbalance_history = {}
        self._last_volatility_cache = {}
        
        # 🆕 O'HARA METHOD 1: Bayesian Price Updating
        self._bayesian_priors = {}  # {symbol: prior_prob_high}
        
        # 🆕 O'HARA METHOD 4: Buy/Sell Imbalance (Enhanced)
        self._trade_imbalance_history = {}  # {symbol: deque of buy/sell counts}
        
        self._init_historical_settings()

    def _init_historical_settings(self):
        """Ініціалізація налаштувань для історичного аналізу"""
        self.historical_enabled = getattr(self.cfg, 'enable_historical_imbalance', True)
        self.historical_window_minutes = getattr(self.cfg, 'historical_window_minutes', 15)
        self.historical_samples = getattr(self.cfg, 'historical_samples', 10)
        self.long_term_smoothing = getattr(self.cfg, 'long_term_smoothing', 0.1)

    def compute(self, symbol: str) -> Dict[str, Any]:
        ob = self.storage.get_order_book(symbol)
        if not ob:
            return self._empty_result(symbol)
        
        # Основний розрахунок імбалансу
        imb_result = self.compute_imbalance(symbol, ob)
        
        # Історичний аналіз імбалансу
        if self.historical_enabled:
            historical_analysis = self.compute_historical_imbalance(symbol, imb_result['imbalance_score'])
            imb_result.update({
                "historical_imbalance": historical_analysis['historical_imb'],
                "imbalance_trend": historical_analysis['trend'],
                "imbalance_volatility": historical_analysis['volatility'],
                "combined_imbalance": historical_analysis['combined_imb']
            })
        else:
            imb_result.update({
                "historical_imbalance": imb_result['imbalance_score'],
                "imbalance_trend": 0,
                "imbalance_volatility": 0,
                "combined_imbalance": imb_result['imbalance_score']
            })
        
        # 🆕 O'HARA METHOD 1: Bayesian updating на основі трейдів
        if self.ohara_cfg.enable_bayesian_updating:
            bayesian_data = self._compute_bayesian_update(symbol)
            imb_result['bayesian_data'] = bayesian_data
        else:
            imb_result['bayesian_data'] = self._empty_bayesian_data()
        
        # 🆕 O'HARA METHOD 4: Enhanced Buy/Sell imbalance
        trade_imbalance = self._compute_trade_imbalance(symbol)
        imb_result['trade_imbalance'] = trade_imbalance
        
        # Додаємо аналіз глибини
        depth_analysis = self.analyze_orderbook_depth(ob)
        
        # Додаємо аналіз кластерів обсягів
        cluster_analysis = self.volume_cluster_analysis(symbol, ob)
        
        # Додаємо ваги для різних рівнів стакану
        weighted_imbalance = self.calculate_weighted_imbalance(ob)
        
        # Ефективний імбалланс
        effective_imbalance = self.calculate_effective_imbalance_enhanced(
            imb_result, depth_analysis, cluster_analysis
        )
        
        imb_result.update({
            "depth_analysis": depth_analysis,
            "cluster_analysis": cluster_analysis,
            "weighted_imbalance": weighted_imbalance,
            "effective_imbalance": effective_imbalance
        })
        
        logger.debug(f"[IMBALANCE] {symbol}: imb={imb_result['imbalance_score']:.1f}, "
                    f"hist={imb_result.get('historical_imbalance', 0):.1f}, "
                    f"weighted={weighted_imbalance:.1f}, effective={effective_imbalance:.1f}, "
                    f"bayesian={imb_result['bayesian_data']['signal']}, "
                    f"trade_imb={trade_imbalance['imbalance_pct']:.1f}%")
        
        return imb_result

    def _compute_bayesian_update(self, symbol: str) -> Dict[str, Any]:
        """
        🆕 O'HARA METHOD 1: Bayesian Price Updating
        Оновлюємо ймовірність "висока ціна" на основі напрямку трейдів
        """
        # Ініціалізація prior для символу
        if symbol not in self._bayesian_priors:
            self._bayesian_priors[symbol] = 0.5  # Нейтральна початкова ймовірність
        
        # Отримуємо останні трейди
        trades = self.storage.get_trades(symbol)
        if not trades or len(trades) < 3:
            return self._empty_bayesian_data()
        
        # Беремо останні N трейдів (останні 30 секунд)
        now = time.time()
        recent_trades = [t for t in trades if t.ts >= now - 30]
        
        if len(recent_trades) < 3:
            return self._empty_bayesian_data()
        
        # Рахуємо покупки vs продажі
        buy_count = sum(1 for t in recent_trades if t.side.lower() == 'buy')
        sell_count = len(recent_trades) - buy_count
        
        # Оновлюємо prior на основі кожного трейду
        prior = self._bayesian_priors[symbol]
        update_step = self.ohara_cfg.bayesian_update_step
        
        # Простий байєсівський апдейт
        for trade in recent_trades[-10:]:  # Останні 10 трейдів
            if trade.side.lower() == 'buy':
                prior += update_step
            else:
                prior -= update_step
        
        # Обмеження 0-1
        prior = max(0.0, min(1.0, prior))
        
        # Природне згасання до 0.5 (нейтральності)
        decay = self.ohara_cfg.bayesian_decay_factor
        prior = prior * decay + 0.5 * (1 - decay)
        
        # Зберігаємо оновлений prior
        self._bayesian_priors[symbol] = prior
        
        # Визначаємо сигнал
        if prior > self.ohara_cfg.bayesian_bullish_threshold:
            signal = "BULLISH"
            confidence = (prior - 0.5) * 2  # 0-1 scale
        elif prior < self.ohara_cfg.bayesian_bearish_threshold:
            signal = "BEARISH"
            confidence = (0.5 - prior) * 2  # 0-1 scale
        else:
            signal = "NEUTRAL"
            confidence = 0.0
        
        return {
            'prior_prob_high': round(prior, 3),
            'signal': signal,
            'confidence': round(confidence, 3),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'recent_trades': len(recent_trades)
        }

    def _compute_trade_imbalance(self, symbol: str) -> Dict[str, Any]:
        """
        🆕 O'HARA METHOD 4: Buy/Sell Trade Imbalance (Enhanced)
        Рахуємо дисбаланс не тільки по orderbook, а й по фактичних трейдах
        """
        trades = self.storage.get_trades(symbol)
        if not trades or len(trades) < 10:
            return {'imbalance_pct': 0.0, 'signal': 'NEUTRAL', 'buy_count': 0, 'sell_count': 0}
        
        # Беремо останні 50 трейдів
        recent_trades = trades[-50:]
        
        buy_count = sum(1 for t in recent_trades if t.side.lower() == 'buy')
        sell_count = len(recent_trades) - buy_count
        
        # Дисбаланс у відсотках
        total = buy_count + sell_count
        if total == 0:
            return {'imbalance_pct': 0.0, 'signal': 'NEUTRAL', 'buy_count': 0, 'sell_count': 0}
        
        imbalance_pct = (buy_count - sell_count) / total * 100.0
        
        # Визначаємо сигнал згідно таблиці O'Hara
        if imbalance_pct > 40:
            signal = "STRONG_BUY"
        elif imbalance_pct > 20:
            signal = "MEDIUM_BUY"
        elif imbalance_pct < -40:
            signal = "STRONG_SELL"
        elif imbalance_pct < -20:
            signal = "MEDIUM_SELL"
        else:
            signal = "NEUTRAL"
        
        return {
            'imbalance_pct': round(imbalance_pct, 1),
            'signal': signal,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'total_trades': total
        }

    def _empty_bayesian_data(self) -> Dict[str, Any]:
        """Порожні дані для байєсівського аналізу"""
        return {
            'prior_prob_high': 0.5,
            'signal': 'NEUTRAL',
            'confidence': 0.0,
            'buy_count': 0,
            'sell_count': 0,
            'recent_trades': 0
        }

    def compute_historical_imbalance(self, symbol: str, current_imb: float) -> Dict[str, Any]:
        """Адаптивний історичний аналіз імбалансу"""
        now = time.time()
        
        # Отримуємо поточну волатильність з кешу
        current_volatility = self._last_volatility_cache.get(symbol, 0.1)
        
        # Адаптивна кількість семплів на основі волатильності
        base_samples = self.cfg.historical_samples
        if current_volatility <= self.adaptive_cfg.base_volatility_threshold * 0.7:
            adaptive_samples = int(base_samples * 1.5)
        elif current_volatility >= self.adaptive_cfg.base_volatility_threshold * 2.0:
            adaptive_samples = int(base_samples * 0.7)
        else:
            adaptive_samples = base_samples
            
        adaptive_samples = max(5, min(adaptive_samples, 50))
        
        if symbol not in self.imbalance_history:
            self.imbalance_history[symbol] = deque(maxlen=adaptive_samples)
        else:
            if self.imbalance_history[symbol].maxlen != adaptive_samples:
                old_data = list(self.imbalance_history[symbol])
                self.imbalance_history[symbol] = deque(old_data, maxlen=adaptive_samples)
        
        self.imbalance_history[symbol].append({
            'timestamp': now,
            'imbalance': current_imb
        })
        
        history = list(self.imbalance_history[symbol])
        
        if len(history) < 3:
            return {
                'historical_imb': current_imb,
                'trend': 0,
                'volatility': 0,
                'combined_imb': current_imb
            }
        
        historical_values = [item['imbalance'] for item in history]
        historical_avg = statistics.mean(historical_values)
        
        if len(history) >= 2:
            oldest = history[0]['imbalance']
            newest = history[-1]['imbalance']
            trend = newest - oldest
        else:
            trend = 0
        
        volatility = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
        
        alpha_current = 0.7
        alpha_historical = 0.3
        
        combined_imb = (current_imb * alpha_current + 
                       historical_avg * alpha_historical)
        
        if abs(trend) > 10:
            trend_factor = min(abs(trend) / 50.0, 0.2)
            combined_imb += trend * trend_factor
        
        combined_imb = max(-self.cfg.universal_imbalance_cap, 
                          min(self.cfg.universal_imbalance_cap, combined_imb))
        
        return {
            'historical_imb': round(historical_avg, 2),
            'trend': round(trend, 2),
            'volatility': round(volatility, 2),
            'combined_imb': round(combined_imb, 2)
        }

    def compute_imbalance(self, symbol: str, ob) -> Dict[str, Any]:
        """Основний розрахунок імбалансу"""
        if not ob or not ob.bids or not ob.asks:
            return self._empty_imbalance_data(symbol, ob.ts if ob else time.time())

        depth = self.cfg.depth_limit_for_calc
        bids = ob.bids[:depth] if ob.bids else []
        asks = ob.asks[:depth] if ob.asks else []

        if not bids or not asks:
            logger.warning(f"[IMBALANCE] {symbol}: Empty bids or asks after slicing")
            return self._empty_imbalance_data(symbol, ob.ts)

        try:
            raw_bid_volume = sum(l.size * l.price for l in bids)
            raw_ask_volume = sum(l.size * l.price for l in asks)
        except (TypeError, AttributeError) as e:
            logger.error(f"[IMBALANCE] {symbol}: Error calculating volumes: {e}")
            return self._empty_imbalance_data(symbol, ob.ts)

        logger.info(f"[IMBALANCE_CALC] {symbol}: {len(bids)} bids, {len(asks)} asks, "
                f"bid_volume=${raw_bid_volume:.2f}, ask_volume=${raw_ask_volume:.2f}")

        eff_bid = raw_bid_volume
        eff_ask = raw_ask_volume
        spoof_filtered = 0.0

        if self.cfg.enable_spoof_filter:
            eff_bid, eff_ask, spoof_filtered = self.apply_spoof_filtering(symbol, bids, asks, eff_bid, eff_ask)

        denom = eff_bid + eff_ask
        if denom < self.cfg.min_volume_epsilon:
            imbalance_score = 0.0
        else:
            imbalance_score = (eff_bid - eff_ask) / denom * 100.0

        imbalance_score = self.apply_imbalance_constraints(symbol, imbalance_score)

        return {
            "symbol": symbol,
            "bid_volume": raw_bid_volume,
            "ask_volume": raw_ask_volume,
            "effective_bid_volume": eff_bid,
            "effective_ask_volume": eff_ask,
            "imbalance_score": round(imbalance_score, 2),
            "spoof_filtered_volume": round(spoof_filtered, 6),
            "timestamp": ob.ts
        }

    def calculate_weighted_imbalance(self, ob) -> float:
        """Розрахунок імбалансу з вагами для різних рівнів"""
        bids = ob.bids[:5] if ob.bids else []
        asks = ob.asks[:5] if ob.asks else []
        
        if not bids or not asks:
            return 0.0
        
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        weighted_bid = sum(b.size * weights[i] for i, b in enumerate(bids[:5]))
        weighted_ask = sum(a.size * weights[i] for i, a in enumerate(asks[:5]))
        
        total_weighted = weighted_bid + weighted_ask
        if total_weighted < self.cfg.min_volume_epsilon:
            return 0.0
        
        return (weighted_bid - weighted_ask) / total_weighted * 100.0

    def analyze_orderbook_depth(self, ob) -> Dict[str, Any]:
        """Аналіз глибини стакану"""
        bids = ob.bids[:10] if ob.bids else []
        asks = ob.asks[:10] if ob.asks else []
        
        if not bids or not asks:
            return {}
        
        bid_liquidity = sum(b.size * b.price for b in bids)
        ask_liquidity = sum(a.size * a.price for a in asks)
        
        total_liquidity = bid_liquidity + ask_liquidity
        if total_liquidity == 0:
            return {
                "bid_liquidity": 0,
                "ask_liquidity": 0,
                "support_resistance_ratio": 0.5,
                "liquidity_imbalance": 0.0
            }
        
        support_resistance_ratio = bid_liquidity / total_liquidity
        liquidity_imbalance = (bid_liquidity - ask_liquidity) / total_liquidity * 100.0
        
        return {
            "bid_liquidity": bid_liquidity,
            "ask_liquidity": ask_liquidity,
            "support_resistance_ratio": support_resistance_ratio,
            "liquidity_imbalance": liquidity_imbalance,
            "best_bid_size": bids[0].size if bids else 0,
            "best_ask_size": asks[0].size if asks else 0
        }

    def volume_cluster_analysis(self, symbol: str, ob, lookback_minutes: int = 15) -> Dict[str, Any]:
        """Кластерний аналіз з адаптивними вікнами"""
        current_volatility = self._last_volatility_cache.get(symbol, 0.1)
        
        base_lookback = lookback_minutes
        if current_volatility <= self.adaptive_cfg.base_volatility_threshold * 0.7:
            adaptive_lookback = int(base_lookback * 1.5)
        elif current_volatility >= self.adaptive_cfg.base_volatility_threshold * 2.0:
            adaptive_lookback = int(base_lookback * 0.7)
        else:
            adaptive_lookback = base_lookback
            
        adaptive_lookback = max(5, min(adaptive_lookback, 60))

        trades = self.storage.get_trades(symbol)
        if not trades or len(trades) < 20:
            return {}
        
        now = time.time()
        window_start = now - (adaptive_lookback * 60)
        recent_trades = [t for t in trades if t.ts >= window_start]
        
        if len(recent_trades) < 10:
            return {}
        
        price_levels = {}
        current_price = (ob.best_bid + ob.best_ask) / 2
        
        for trade in recent_trades:
            if current_price > 100:
                bin_size = 0.01
            elif current_price > 10:
                bin_size = 0.001
            else:
                bin_size = 0.0001
                
            level = round(trade.price / bin_size) * bin_size
            price_levels[level] = price_levels.get(level, 0) + trade.size * trade.price
        
        if not price_levels:
            return {}
        
        poc_level = max(price_levels, key=price_levels.get)
        poc_volume = price_levels[poc_level]
        
        poc_distance_pct = (poc_level - current_price) / current_price * 100 if current_price > 0 else 0
        
        avg_volume = sum(price_levels.values()) / len(price_levels)
        hvn_levels = {level: vol for level, vol in price_levels.items() if vol > avg_volume * 2}
        
        cluster_strength = self.analyze_cluster_strength(price_levels, current_price)
        support_resistance = self.identify_support_resistance(price_levels, current_price)
        
        result = {
            "poc_level": poc_level,
            "poc_volume": poc_volume,
            "poc_distance_pct": poc_distance_pct,
            "current_price": current_price,
            "hvn_count": len(hvn_levels),
            "total_volume_clusters": len(price_levels),
            "cluster_strength": cluster_strength,
            "support_levels": support_resistance['support'],
            "resistance_levels": support_resistance['resistance'],
            "time_window_minutes": adaptive_lookback,
            "trades_analyzed": len(recent_trades),
            "volume_clusters": dict(sorted(price_levels.items(), key=lambda x: x[1], reverse=True)[:8])
        }

        if cluster_strength > 0.3 or abs(poc_distance_pct) > 1.0:
            logger.info(f"[ADAPTIVE_CLUSTER] {symbol}: {adaptive_lookback}min - "
                       f"POC={poc_level:4f} (dist={poc_distance_pct:.2f}%), "
                       f"strength={cluster_strength:.2f}, "
                       f"supports={len(support_resistance['support'])}, "
                       f"resistances={len(support_resistance['resistance'])}")

        return result

    def analyze_cluster_strength(self, price_levels: Dict[float, float], current_price: float) -> float:
        """Аналіз сили кластерів"""
        if not price_levels or current_price <= 0:
            return 0.0
        
        nearby_volume = 0
        total_volume = sum(price_levels.values())
        
        for level, volume in price_levels.items():
            distance_pct = abs(level - current_price) / current_price * 100
            if distance_pct <= 2.0:
                nearby_volume += volume
        
        if total_volume == 0:
            return 0.0
        
        return nearby_volume / total_volume

    def identify_support_resistance(self, price_levels: Dict[float, float], current_price: float) -> Dict[str, List]:
        """Ідентифікація рівнів підтримки та опору"""
        if not price_levels:
            return {'support': [], 'resistance': []}
        
        sorted_levels = sorted(price_levels.items(), key=lambda x: x[1], reverse=True)
        
        support = []
        resistance = []
        
        for level, volume in sorted_levels[:10]:
            if level < current_price:
                support.append({'level': level, 'volume': volume})
            else:
                resistance.append({'level': level, 'volume': volume})
        
        support.sort(key=lambda x: x['level'])
        resistance.sort(key=lambda x: x['level'], reverse=True)
        
        return {
            'support': support[:3],
            'resistance': resistance[:3]
        }

    def calculate_effective_imbalance_enhanced(self, imb_data: Dict, depth_data: Dict, 
                                             cluster_data: Dict) -> float:
        """Покращена формула ефективного імбалансу"""
        base_imb = imb_data.get('imbalance_score', 0)
        historical_imb = imb_data.get('historical_imbalance', base_imb)
        combined_imb = imb_data.get('combined_imbalance', base_imb)
        weighted_imb = imb_data.get('weighted_imbalance', 0)
        depth_ratio = depth_data.get('support_resistance_ratio', 0.5)
        
        depth_imb = (depth_ratio - 0.5) * 200
        
        effective_imb = (
            combined_imb * 0.4 +
            historical_imb * 0.3 +
            weighted_imb * 0.2 +
            depth_imb * 0.1
        )
        
        return round(effective_imb, 2)

    def apply_spoof_filtering(self, symbol: str, bids, asks, eff_bid, eff_ask):
        """Фільтрація спуфінгу"""
        spoof_filtered = 0.0
        suspicious = self.storage.get_suspicious_orders(symbol, last_seconds=settings.websocket.data_retention_seconds)
        
        bid_price_map = {l.price: l.size for l in bids}
        ask_price_map = {l.price: l.size for l in asks}
        
        for s in suspicious:
            if s.side == "bid" and s.price in bid_price_map:
                to_sub = min(bid_price_map[s.price], s.size)
                eff_bid -= to_sub
                spoof_filtered += to_sub
            elif s.side == "ask" and s.price in ask_price_map:
                to_sub = min(ask_price_map[s.price], s.size)
                eff_ask -= to_sub
                spoof_filtered += to_sub
        
        return max(eff_bid, 0.0), max(eff_ask, 0.0), spoof_filtered

    def apply_imbalance_constraints(self, symbol: str, imbalance_score: float) -> float:
        """Обмеження та згладжування імбалансу"""
        if imbalance_score > self.cfg.universal_imbalance_cap:
            imbalance_score = self.cfg.universal_imbalance_cap
        elif imbalance_score < -self.cfg.universal_imbalance_cap:
            imbalance_score = -self.cfg.universal_imbalance_cap
        
        if symbol in self.last_imbalance:
            last_imb = self.last_imbalance[symbol]
            imbalance_score = (last_imb * (1 - self.cfg.smoothing_factor) + 
                             imbalance_score * self.cfg.smoothing_factor)
        
        self.last_imbalance[symbol] = imbalance_score
        return imbalance_score

    def update_volatility_cache(self, symbol: str, volatility_data: Dict[str, Any]):
        """Оновлення кешу волатильності для адаптивних вікон"""
        if volatility_data:
            current_volatility = volatility_data.get("recent_volatility", 0.1)
            self._last_volatility_cache[symbol] = current_volatility

    def _empty_result(self, symbol: str) -> Dict[str, Any]:
        now = time.time()
        empty_data = self._empty_imbalance_data(symbol, now)
        empty_data.update({
            "depth_analysis": {},
            "cluster_analysis": {},
            "weighted_imbalance": 0.0,
            "effective_imbalance": 0.0,
            "historical_imbalance": 0.0,
            "imbalance_trend": 0.0,
            "imbalance_volatility": 0.0,
            "combined_imbalance": 0.0,
            "bayesian_data": self._empty_bayesian_data(),
            "trade_imbalance": {'imbalance_pct': 0.0, 'signal': 'NEUTRAL', 'buy_count': 0, 'sell_count': 0}
        })
        return empty_data

    def _empty_imbalance_data(self, symbol: str, timestamp: float) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "imbalance_score": 0.0,
            "bid_volume": 0.0,
            "ask_volume": 0.0,
            "effective_bid_volume": 0.0,
            "effective_ask_volume": 0.0,
            "spoof_filtered_volume": 0.0,
            "timestamp": timestamp
        }