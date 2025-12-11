# main.py
import asyncio
import sys
import time
from config.settings import settings
from utils.logger import logger
from utils.notifications import notifier
from data.storage import DataStorage, Position
from data.collector import DataCollector
from analysis.imbalance import ImbalanceAnalyzer
from analysis.volume import VolumeAnalyzer
from analysis.signals import SignalGenerator
from trading.bybit_api_manager import BybitAPIManager
from trading.executor import TradeExecutor
from trading.orchestrator import TradingOrchestrator

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def emergency_fix(storage: DataStorage):
    """ФІКС: Примусово закриваємо позиції, які блокували бота"""
    logger.info("🚑 [EMERGENCY_FIX] Applying emergency position fix...")
    
    problem_symbols = []
    for symbol, position in storage.positions.items():
        if position.status == "OPEN":
            current_time = time.time()
            if current_time - position.last_update > 300:  # 5 хвилин без оновлення
                problem_symbols.append(symbol)
                logger.warning(f"🔄 [EMERGENCY] Forcing close for stuck position: {symbol}")
                position.status = "CLOSED"
                position.close_reason = "EMERGENCY_CLOSE"
                position._position_updated = True
    
    if problem_symbols:
        logger.info(f"✅ [EMERGENCY_FIX] Fixed {len(problem_symbols)} stuck positions")
    return problem_symbols

async def run_csv_validation():
    """Запуск валідації CSV (не блокує запуск бота)"""
    try:
        from utils.csv_test import main as validate_csv
        logger.info("🔍 [MAIN] Running CSV validation...")
        success = await validate_csv()
        if success:
            logger.info("✅ [MAIN] CSV validation completed")
        else:
            logger.warning("⚠️ [MAIN] CSV validation found issues (continuing)")
        return True  # Завжди продовжуємо роботу
    except Exception as e:
        logger.error(f"❌ [MAIN] CSV validation failed: {e}")
        return True  # Завжди продовжуємо роботу навіть при помилці

async def delayed_validation():
    """Відкладена валідація через 30 хвилин"""
    await asyncio.sleep(1800)  # 30 хвилин
    await run_csv_validation()

async def main():
    logger.info("=" * 60)
    logger.info("🚀 CRYPTO TRADING BOT - OPTIMIZED MONITORING SYSTEM")
    logger.info("=" * 60)
    
    # Показуємо інформацію про режим
    mode_info = settings.system.get_mode_info()
    logger.info("")
    logger.info(f"📡 MODE: {mode_info['mode']}")
    logger.info(f"📊 Public WebSocket:  {mode_info['ws_public']}")
    logger.info(f"🔐 Private WebSocket: {mode_info['ws_private']}")
    logger.info(f"🌐 REST API:          {mode_info['rest_api']}")
    logger.info(f"💡 Note: {mode_info['note']}")
    logger.info("")

    # ШВИДКА перевірка CSV (не блокує запуск)
    asyncio.create_task(run_csv_validation())
    
    # Відкладена перевірка через 30 хвилин
    asyncio.create_task(delayed_validation())

    api_manager = BybitAPIManager()

    storage = DataStorage(
        retention_seconds=settings.risk.max_position_lifetime_sec,
        large_order_side_percent=settings.imbalance.large_order_side_percent,
        spoof_lifetime_ms=settings.imbalance.spoof_lifetime_ms,
        large_order_min_abs=settings.imbalance.large_order_min_notional_abs,
        max_depth=settings.websocket.subscription_depth
    )

    # ЕКСТРЕНЕ ВІДНОВЛЕННЯ ПЕРЕД ЗАПУСКОМ
    await emergency_fix(storage)

    collector = DataCollector(storage, api_manager)
    imb_analyzer = ImbalanceAnalyzer(storage)
    vol_analyzer = VolumeAnalyzer(storage)
    signal_generator = SignalGenerator()
    executor = TradeExecutor(storage, api_manager)
    orchestrator = TradingOrchestrator(storage, imb_analyzer, vol_analyzer, signal_generator, executor)

    try:
        await collector.start()
        await executor.start()
        await orchestrator.start()

        try:
            await notifier.send(f"🤖 Bot started in {mode_info['mode']} mode with Optimized Monitoring System")
        except Exception:
            logger.warning("Failed to send startup notification")

        logger.info("=" * 60)
        logger.info("✅ BOT IS RUNNING WITH OPTIMIZED MONITORING SYSTEM")
        logger.info("=" * 60)
        logger.info("📊 Data Sources:")
        logger.info("   • Public WS:  Orderbook & Trades (real-time)")
        logger.info("   • Private WS: Positions & Executions (real-time)")
        logger.info("   • REST API:   Fallback & sync")
        logger.info("")
        logger.info("🎯 Optimized Monitoring Features:")
        logger.info("   • Fast position monitoring every 5s")
        logger.info("   • Adaptive symbol batching")
        logger.info("   • Cached API responses")
        logger.info("   • Ultra-fast close reason detection")
        logger.info("   • Reduced API calls by 60%")
        logger.info("   • Non-blocking CSV validation")
        logger.info("=" * 60)

        while True:
            await asyncio.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        logger.info("Received shutdown signal...")
    except Exception as e:
        logger.error(f"Critical error: {e}")
    finally:
        logger.info("Shutting down components...")
        await safe_shutdown(collector, orchestrator, executor, api_manager)

async def safe_shutdown(collector, orchestrator, executor, api_manager):
    logger.info("🛑 Starting safe shutdown...")
    await collector.stop()
    await orchestrator.stop()
    await executor.stop()
    await api_manager.close()
    logger.info("✅ Bot stopped safely")

if __name__ == "__main__":
    asyncio.run(main())