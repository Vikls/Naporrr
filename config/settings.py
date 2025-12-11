# config/settings.py
import os
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

class SystemSettings(BaseSettings):
    """Системні налаштування для різних режимів"""
    rest_market_base: str = "https://api.bybit.com"
    rest_market_base_demo: str = "https://api-demo.bybit.com"
    ws_public_linear: str = "wss://stream.bybit.com/v5/public/linear"
    ws_public_linear_demo: str = "wss://stream-demo.bybit.com/v5/public/linear"
    ws_private: str = "wss://stream.bybit.com/v5/private"
    ws_private_demo: str = "wss://stream-demo.bybit.com/v5/private"

    def get_mode_info(self) -> Dict[str, Any]:
        from config.settings import settings
        mode = settings.trading.mode.upper()
        
        if mode == "DEMO":
            return {
                "mode": "DEMO (Paper Trading)",
                "ws_public": self.ws_public_linear_demo,
                "ws_private": self.ws_private_demo,
                "rest_api": self.rest_market_base_demo,
                "note": "Using demo environment with virtual funds"
            }
        else:
            return {
                "mode": "LIVE (Real Trading)",
                "ws_public": self.ws_public_linear,
                "ws_private": self.ws_private,
                "rest_api": self.rest_market_base,
                "note": "⚠️ REAL MONEY - Trading with actual funds"
            }

class SecretsSettings(BaseSettings):
    """API ключі та секрети"""
    bybit_api_key: str = Field(default="", alias="BYBIT_API_KEY")
    bybit_api_secret: str = Field(default="", alias="BYBIT_API_SECRET")
    
    demo_bybit_api_key: str = Field(default="", alias="BYBIT_API_KEY_DEMO")
    demo_bybit_api_secret: str = Field(default="", alias="BYBIT_API_SECRET_DEMO")
    
    live_bybit_api_key: str = Field(default="", alias="BYBIT_API_KEY_LIVE")
    live_bybit_api_secret: str = Field(default="", alias="BYBIT_API_SECRET_LIVE")
    
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    class Config:
        env_file = "config/.env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        populate_by_name = True

class PairsSettings(BaseSettings):
    """Налаштування торгових пар"""
    trade_pairs: list = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", 
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT",
        "AAVEUSDT", "STRKUSDT"
    ]
    
    low_liquidity_pairs: list = ["HFTUSDT", "TRXUSDT"]
    excluded_pairs: list = ["HFTUSDT"]

class TradingSettings(BaseSettings):
    """Основні налаштування торгівлі"""
    mode: str = "DEMO"
    leverage: int = 10
    base_order_usdt: float = 0.0
    base_order_pct: float = 0.1
    start_balance_usdt: float = 0.0
    
    max_orders_per_second: int = 5
    max_orders_per_minute: int = 100
    max_reprice_attempts: int = 8
    
    entry_signal_min_strength: int = 3
    close_on_opposite_strength: int = 5
    
    decision_interval_sec: float = 2.0
    min_time_between_trades_sec: float = 15.0
    reopen_cooldown_sec: float = 10.0
    min_position_hold_time_sec: float = 30.0
    
    monitor_positions_interval_sec: float = 5.0
    enable_parallel_monitoring: bool = True
    monitoring_batch_size: int = 5
    
    reverse_signals: bool = False
    reverse_double_size: bool = False
    
    enable_aggressive_filtering: bool = True

class RiskSettings(BaseSettings):
    """Налаштування ризик-менеджменту - ОПТИМІЗОВАНО З БЕКТЕСТУ"""
    
    max_open_positions: int = 5
    max_position_notional_pct: float = 1.0
    
    # =====================================================
    # ⏱️ LIFETIME - 60 хв оптимум з бектесту
    # =====================================================
    base_position_lifetime_minutes: int = 60   # було 120 → 60
    enable_adaptive_lifetime: bool = True
    
    low_volatility_lifetime_multiplier: float = 1.5
    high_volatility_lifetime_multiplier: float = 0.7
    volatility_threshold_low: float = 0.5
    volatility_threshold_high: float = 2.0
    
    # =====================================================
    # 🎯 TP/SL - ОПТИМІЗОВАНО З БЕКТЕСТУ
    # Стратегія: тайтовий SL + далекий TP = високий RR
    # =====================================================
    enable_dynamic_tpsl: bool = True
    
    # SL - тайтовий, швидко ріжемо збитки
    min_sl_pct: float = 0.002       # 0.2% - оптимум! 
    max_sl_pct: float = 0.004       # 0.4% макс
    
    # TP - далекий, даємо прибутку рости
    min_tp_pct: float = 0.010       # 1.0% мін
    max_tp_pct: float = 0.015       # 1.5% - оптимум!
    
    # Множники волатильності
    sl_vol_multiplier: float = 1.0
    tp_vol_multiplier: float = 2.5
    max_vol_used_pct: float = 5.0
    
    # =====================================================
    # 📊 RR RATIO - ВИСОКИЙ (7.5:1 з бектесту)
    # =====================================================
    enable_dynamic_tpsl_ratio: bool = True
    tpsl_ratio_high_winrate: float = 5.0    # було 2.0
    tpsl_ratio_medium_winrate: float = 6.0  # було 2.5
    tpsl_ratio_low_winrate: float = 7.5     # було 3.0
    
    # =====================================================
    # 🔄 TRAILING STOP - вимкнено при високому RR
    # =====================================================
    enable_trailing_stop: bool = False      # вимкнено
    trailing_stop_activation_pct: float = 0.008
    trailing_stop_distance_pct: float = 0.003
    
    position_history_size: int = 100
    min_history_for_adaptation: int = 20
    
    @property
    def position_lifetime_minutes(self) -> int:
        return self.base_position_lifetime_minutes
    
    @property
    def max_position_lifetime_sec(self) -> int:
        return self.base_position_lifetime_minutes * 60

class ExecutionSettings(BaseSettings):
    """Налаштування виконання ордерів"""
    poll_interval_sec: float = 0.5
    max_wait_sec: float = 60.0
    reprice_every_sec: float = 3.0
    reprice_step_bps: float = 5.0
    passive_improve_bps: float = 2.0
    
    require_full_fill: bool = False
    min_partial_pct: float = 0.8
    
    fallback_mode: str = "market"
    fallback_after_sec: float = 30.0
    cancel_before_fallback: bool = True

class WebSocketSettings(BaseSettings):
    """Налаштування WebSocket"""
    subscription_depth: int = 50
    ping_interval: float = 20.0
    reconnect_delay_seconds: float = 5.0
    data_retention_seconds: int = 300
    
    enable_private_ws: bool = True
    private_ws_heartbeat_interval: float = 20.0
    private_ws_reconnect_attempts: int = 5

class APISettings(BaseSettings):
    """Налаштування API"""
    retry_attempts: int = 3
    retry_delay: float = 1.0
    validate_time_diff_sec: int = 5
    instrument_cache_ttl: int = 3600
    ticker_cache_ttl: int = 5

class LoggingSettings(BaseSettings):
    """Налаштування логування"""
    mode: str = "work"
    
    console_level_debug: str = "DEBUG"
    file_level_debug: str = "DEBUG"
    console_level_work: str = "INFO"
    file_level_work: str = "DEBUG"
    
    log_dir: Path = Path("logs")
    common_log: Path = Path("logs/bot.log")
    errors_log: Path = Path("logs/errors.log")
    trades_log: Path = Path("logs/trades.csv")

class ImbalanceSettings(BaseSettings):
    """Налаштування аналізу дисбалансу"""
    depth_limit_for_calc: int = 50
    min_volume_epsilon: float = 1e-9
    large_order_side_percent: float = 0.05
    
    enable_adaptive_large_orders: bool = True
    large_order_zscore_threshold: float = 2.0
    large_order_lookback_periods: int = 100
    large_order_min_samples: int = 20
    
    large_order_min_notional_abs: float = 500.0
    
    spoof_lifetime_ms: int = 3000
    
    enable_spoof_filter: bool = True
    smoothing_factor: float = 0.3
    universal_imbalance_cap: float = 100.0
    
    enable_historical_imbalance: bool = True
    historical_window_minutes: int = 15
    historical_samples: int = 10
    long_term_smoothing: float = 0.1

class VolumeSettings(BaseSettings):
    """Налаштування аналізу обсягів"""
    short_window_sec: int = 30
    long_window_sec: int = 300
    default_min_trades: int = 5
    vwap_min_volume: float = 100.0
    
    enable_multi_timeframe_momentum: bool = True
    momentum_windows: list = [15, 30, 60, 120]
    momentum_weights: list = [0.4, 0.3, 0.2, 0.1]
    
    enable_adaptive_volume_analysis: bool = True
    volume_zscore_threshold_high: float = 2.0
    volume_zscore_threshold_very_high: float = 3.0
    volume_zscore_threshold_low: float = -1.0
    volume_lookback_periods: int = 96
    volume_min_samples: int = 20
    
    enable_percentile_method: bool = True
    volume_percentile_very_high: float = 95.0
    volume_percentile_high: float = 75.0
    volume_percentile_low: float = 25.0
    
    enable_ema_volume_analysis: bool = True
    ema_fast_period: int = 20
    ema_slow_period: int = 100
    ema_ratio_high: float = 2.0
    ema_ratio_very_high: float = 3.0
    
    enable_trade_frequency_analysis: bool = True
    frequency_baseline_window_sec: int = 300
    frequency_very_high_multiplier: float = 5.0
    frequency_high_multiplier: float = 2.5
    frequency_very_low_multiplier: float = 0.3
    
    enable_volume_confirmation: bool = True
    volume_baseline_window_sec: int = 86400
    volume_confirmation_zscore: float = 1.2
    volume_weak_zscore: float = -0.5
    
    enable_large_order_tracker: bool = True
    large_order_lookback_sec: int = 600
    large_order_strong_threshold: int = 3

class AdaptiveSettings(BaseSettings):
    """Налаштування адаптивних механізмів"""
    enable_adaptive_windows: bool = True
    base_volatility_threshold: float = 1.0
    
    low_volatility_multiplier: float = 1.5
    high_volatility_multiplier: float = 0.7
    
    max_window_expansion: float = 2.0
    min_window_reduction: float = 0.5

class SignalSettings(BaseSettings):
    """Налаштування генерації сигналів - ОПТИМІЗОВАНО З БЕКТЕСТУ"""
    
    # =====================================================
    # ⚖️ ВАГИ КОМПОНЕНТІВ
    # =====================================================
    weight_momentum: float = 0.20
    weight_ohara_bayesian: float = 0.12
    weight_ohara_large_orders: float = 0.08
    weight_imbalance: float = 0.45          # imbalance найважливіший! 
    weight_ohara_frequency: float = 0.075
    weight_ohara_volume_confirm: float = 0.075
    spike_bonus: float = 0.1
    
    smoothing_alpha: float = 0.75
    hold_threshold: float = 0.12
    
    # =====================================================
    # 📊 ПОРОГИ СИЛИ СИГНАЛУ
    # =====================================================
    composite_thresholds: dict = {
        "strength_1": 0.15,
        "strength_2": 0.30,
        "strength_3": 0.40,
        "strength_4": 0.65,
        "strength_5": 0.80
    }
    
    min_strength_for_action: int = 3
    
    # =====================================================
    # 🎚️ АДАПТИВНІ ПОРОГИ
    # =====================================================
    enable_adaptive_threshold: bool = True
    base_threshold: float = 0.40
    min_threshold: float = 0.32
    max_threshold: float = 0.50
    
    high_volatility_threshold_reduction: float = 0.05
    low_volatility_threshold_increase: float = 0.03
    volatility_high_level: float = 2.0
    volatility_low_level: float = 0.5
    
    high_liquidity_threshold_reduction: float = 0.03
    low_liquidity_threshold_increase: float = 0.05
    
    # =====================================================
    # 🚫 EARLY ENTRY - ВИМКНЕНО (причина збитків)
    # =====================================================
    early_entry_enabled: bool = False
    early_entry_momentum_threshold: float = 40.0
    early_entry_volatility_threshold: float = 0.3
    early_entry_ohara_threshold: int = 6
    early_entry_imbalance_threshold: float = 35.0
    early_entry_threshold_multiplier: float = 0.72
    
    # =====================================================
    # 🎯 ФІЛЬТРИ ВХОДУ - ОПТИМІЗОВАНО З БЕКТЕСТУ
    # =====================================================
    
    # Imbalance - КЛЮЧОВИЙ!  WR 62.5% при imb 30-50%
    min_imbalance_for_entry: float = 20.0   # було 8 → 20 (бектест)
    
    # Momentum - діапазон 50-85%
    min_momentum_for_entry: float = 50.0    # було 45 → 50
    max_momentum_for_entry: float = 85.0    # було 92 → 85
    
    # O'Hara score
    min_ohara_for_entry: int = 5            # було 4 → 5
    
    # Large orders - не блокуємо, тільки перевіряємо напрямок
    min_large_orders_for_entry: int = 0     # було 1 → 0
    
    # =====================================================
    # ⏰ LATE ENTRY
    # =====================================================
    late_entry_momentum_threshold: float = 85.0
    late_entry_allow_strong_trend: bool = True
    late_entry_min_ohara_for_override: int = 7
    late_entry_position_size_reduction: float = 0.5
    late_entry_high_momentum_threshold: float = 80.0
    
    # =====================================================
    # 📈 LARGE ORDER BONUS
    # =====================================================
    large_order_count_bonus_threshold: int = 3
    large_order_count_bonus_per_order: float = 0.03
    large_order_count_bonus_max: float = 0.15
    
    # =====================================================
    # 🔍 O'HARA SCORE
    # =====================================================
    ohara_strong_score_threshold: int = 8
    ohara_threshold_reduction: float = 0.03
    
    # =====================================================
    # ⚔️ CONTRADICTORY SIGNALS
    # =====================================================
    allow_override_contradictory_orders: bool = False  # не йдемо проти китів
    override_imbalance_threshold: float = 40.0
    override_momentum_threshold: float = 50.0
    strong_cooldown_level: int = 3
    cooldown_seconds: float = 180.0
    
    allow_reversal_during_cooldown: bool = True
    require_signal_consistency: bool = True
    max_imbalance_contradiction: float = 20.0
    
    # =====================================================
    # ✅ ВАЛІДАЦІЯ
    # =====================================================
    enable_volume_validation: bool = True
    min_short_volume_for_signal: float = 1000.0
    min_trades_for_signal: int = 10
    
    volatility_filter_threshold: float = 0.25
    
    enable_exhaustion_filter: bool = True
    max_momentum_for_entry: float = 85.0    # sync з фільтрами
    min_imbalance_for_high_momentum: float = 15.0

class SpreadSettings(BaseSettings):
    """O'HARA METHOD 7: Spread as Risk Measure"""
    enable_spread_monitor: bool = True
    
    max_spread_threshold_bps: float = 20.0
    high_risk_spread_multiplier: float = 3.0
    very_high_risk_spread_multiplier: float = 5.0
    
    spread_history_size: int = 100
    spread_baseline_window_sec: int = 3600
    
    avoid_trading_on_very_high_spread: bool = True
    reduce_size_on_high_spread: bool = True
    high_spread_size_reduction_pct: float = 0.5

class OHaraSettings(BaseSettings):
    """O'HARA METHODS: Comprehensive Settings"""
    
    enable_bayesian_updating: bool = True
    bayesian_update_step: float = 0.05
    bayesian_bullish_threshold: float = 0.65
    bayesian_bearish_threshold: float = 0.35
    bayesian_decay_factor: float = 0.98
    
    large_order_min_count_strong: int = 3
    large_order_min_count_medium: int = 2
    large_order_net_threshold: int = 2
    
    enable_combined_ohara_score: bool = True
    min_ohara_score_for_trade: int = 4
    strong_ohara_score_threshold: int = 8

class Settings(BaseSettings):
    """Головний клас налаштувань"""
    system: SystemSettings = SystemSettings()
    secrets: SecretsSettings = SecretsSettings()
    pairs: PairsSettings = PairsSettings()
    trading: TradingSettings = TradingSettings()
    risk: RiskSettings = RiskSettings()
    execution: ExecutionSettings = ExecutionSettings()
    websocket: WebSocketSettings = WebSocketSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    imbalance: ImbalanceSettings = ImbalanceSettings()
    volume: VolumeSettings = VolumeSettings()
    adaptive: AdaptiveSettings = AdaptiveSettings()
    signals: SignalSettings = SignalSettings()
    spread: SpreadSettings = SpreadSettings()
    ohara: OHaraSettings = OHaraSettings()

settings = Settings()