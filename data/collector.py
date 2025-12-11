# data/collector.py
import asyncio
import json
import websockets
import time
from typing import Dict, Optional
from utils.logger import logger
from config.settings import settings
from data.storage import DataStorage
from trading.bybit_api_manager import BybitAPIManager
from data.private_ws_collector import PrivateWSCollector


class DataCollector:
    """
    Hybrid data collector:
    - Primary: Private WebSocket (позиції, executions)
    - Primary: Public WebSocket (orderbook, trades)
    - Fallback: REST API
    """

    def __init__(self, storage: DataStorage, api_manager: BybitAPIManager):
        self.storage = storage
        self.api = api_manager
        self.pairs = settings.pairs.trade_pairs
        self.cfg = settings.websocket
        
        # Public WebSocket
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Private WebSocket
        self.private_ws = PrivateWSCollector(storage, api_manager)
        
        # REST fallback
        self._rest_fallback_task: Optional[asyncio.Task] = None

    async def start(self):
        """Запуск збирача даних"""
        if self._ws_task:
            logger.warning("[COLLECTOR] Already running")
            return
        
        logger.info("🚀 [COLLECTOR] Starting data collection...")
        
        # Показуємо інформацію про режим
        mode_info = settings.system.get_mode_info()
        logger.info(f"📡 [COLLECTOR] Mode: {mode_info['mode']}")
        logger.info(f"📡 [COLLECTOR] Public WS: {mode_info['ws_public']}")
        logger.info(f"📡 [COLLECTOR] Private WS: {mode_info['ws_private']}")
        logger.info(f"📡 [COLLECTOR] REST API: {mode_info['rest_api']}")
        
        # Ініціалізуємо orderbook через REST (fallback)
        for symbol in self.pairs:
            await self.storage.init_orderbook_rest(symbol)
            await asyncio.sleep(0.1)
        
        # Запускаємо Private WebSocket (primary для позицій)
        await self.private_ws.start()
        
        # Запускаємо Public WebSocket (для orderbook/trades)
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        
        # Запускаємо REST fallback
        self._rest_fallback_task = asyncio.create_task(self._rest_fallback_loop())
        
        logger.info("✅ [COLLECTOR] Data collection started")

    async def stop(self):
        """Зупинка збирача даних"""
        logger.info("🛑 [COLLECTOR] Stopping data collection...")
        self._running = False
        
        # Зупиняємо Private WebSocket
        await self.private_ws.stop()
        
        # Зупиняємо Public WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        
        # Зупиняємо REST fallback
        if self._rest_fallback_task:
            self._rest_fallback_task.cancel()
            try:
                await self._rest_fallback_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ [COLLECTOR] Data collection stopped")

    async def _ws_loop(self):
        """Public WebSocket loop для orderbook/trades"""
        attempt = 0
        while self._running:
            try:
                ws_url = settings.system.ws_public_linear
                logger.info(f"🔗 [PUBLIC_WS] Connecting to {ws_url}")
                
                async with websockets.connect(
                    ws_url,
                    ping_interval=settings.websocket.ping_interval,
                    ping_timeout=30
                ) as ws:
                    self._ws = ws
                    attempt = 0
                    
                    # Підписка на orderbook і trades
                    await self._subscribe_public(ws)
                    
                    # Обробка повідомлень
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            await self._handle_public_message(data)
                        except Exception as e:
                            logger.error(f"[PUBLIC_WS] Message error: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                attempt += 1
                logger.error(f"[PUBLIC_WS] Error (attempt {attempt}): {e}")
                
                if self._running:
                    delay = min(self.cfg.reconnect_delay_seconds * attempt, 30)
                    await asyncio.sleep(delay)

    async def _subscribe_public(self, ws):
        """Підписка на Public топіки"""
        try:
            # Orderbook
            orderbook_topics = [
                f"orderbook.{self.cfg.subscription_depth}.{symbol}"
                for symbol in self.pairs
            ]
            
            # Public trades
            trade_topics = [f"publicTrade.{symbol}" for symbol in self.pairs]
            
            all_topics = orderbook_topics + trade_topics
            
            sub_msg = {
                "op": "subscribe",
                "args": all_topics
            }
            
            await ws.send(json.dumps(sub_msg))
            logger.info(f"✅ [PUBLIC_WS] Subscribed to {len(all_topics)} topics")
            
        except Exception as e:
            logger.error(f"[PUBLIC_WS] Subscribe error: {e}")
            raise

    async def _handle_public_message(self, data: dict):
        """Обробка Public повідомлень"""
        try:
            topic = data.get("topic", "")
            
            if topic.startswith("orderbook"):
                await self._handle_orderbook(data)
            elif topic.startswith("publicTrade"):
                await self._handle_trades(data)
                
        except Exception as e:
            logger.error(f"[PUBLIC_WS] Handler error: {e}")

    async def _handle_orderbook(self, data: dict):
        """Обробка orderbook оновлень"""
        try:
            topic = data["topic"]
            symbol = topic.split(".")[-1]
            
            msg_type = data.get("type")
            ob_data = data.get("data", {})
            
            if msg_type == "snapshot":
                bids = ob_data.get("b", [])
                asks = ob_data.get("a", [])
                self.storage.update_order_book(symbol, bids, asks)
            elif msg_type == "delta":
                bids_delta = ob_data.get("b", [])
                asks_delta = ob_data.get("a", [])
                self.storage.apply_order_book_delta(symbol, bids_delta, asks_delta)
                
        except Exception as e:
            logger.error(f"[PUBLIC_WS] Orderbook error: {e}")

    async def _handle_trades(self, data: dict):
        """Обробка публічних трейдів"""
        try:
            topic = data["topic"]
            symbol = topic.split(".")[-1]
            trades = data.get("data", [])
            
            for trade in trades:
                price = float(trade["p"])
                size = float(trade["v"])
                side = trade["S"].lower()
                
                self.storage.add_trade(
                    symbol=symbol,
                    price=price,
                    size=size,
                    side=side,
                    is_aggressive=True
                )
                
        except Exception as e:
            logger.error(f"[PUBLIC_WS] Trades error: {e}")

    async def _rest_fallback_loop(self):
        """REST API fallback для позицій (якщо WS відключився)"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Перевіряємо кожні 5с
                
                # Якщо Private WS не працює, синхронізуємо через REST
                if not self.private_ws._running or not self.private_ws.ws:
                    logger.warning("[REST_FALLBACK] Private WS down, using REST...")
                    await self.storage.force_sync_positions(self.api)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[REST_FALLBACK] Error: {e}")
                await asyncio.sleep(10)