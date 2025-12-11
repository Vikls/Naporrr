# data/private_ws_collector.py
import asyncio
import json
import websockets
import time
from typing import Optional
from utils.logger import logger
from config.settings import settings
from data.storage import DataStorage
from trading.bybit_api_manager import BybitAPIManager


class PrivateWSCollector:
    """
    Private WebSocket collector для отримання:
    - Position updates (real-time позиції)
    - Execution events (TP/SL спрацювання)
    - Order updates (статус ордерів)
    - Wallet updates (баланс)
    """

    def __init__(self, storage: DataStorage, api_manager: BybitAPIManager):
        self.storage = storage
        self.api = api_manager
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_pong = time.time()
        self._reconnect_delay = settings.websocket.reconnect_delay_seconds
        self._heartbeat_interval = settings.websocket.private_ws_heartbeat_interval

    async def start(self):
        """Запуск Private WebSocket"""
        if not settings.websocket.enable_private_ws:
            logger.info("[PRIVATE_WS] Disabled in settings")
            return
        
        if self._task:
            logger.warning("[PRIVATE_WS] Already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("✅ [PRIVATE_WS] Started")

    async def stop(self):
        """Зупинка Private WebSocket"""
        logger.info("🛑 [PRIVATE_WS] Stopping...")
        self._running = False
        
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.error(f"[PRIVATE_WS] Close error: {e}")
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info("✅ [PRIVATE_WS] Stopped")

    async def _run(self):
        """Основний цикл з reconnect логікою"""
        attempt = 0
        max_attempts = settings.websocket.private_ws_reconnect_attempts
        
        while self._running:
            try:
                ws_url = settings.system.ws_private
                logger.info(f"🔗 [PRIVATE_WS] Connecting to {ws_url} (attempt {attempt + 1}/{max_attempts})")
                
                async with websockets.connect(
                    ws_url,
                    ping_interval=None,  # Ми самі керуємо ping
                    ping_timeout=30,
                    close_timeout=10
                ) as ws:
                    self.ws = ws
                    attempt = 0  # Скидаємо лічильник після успішного підключення
                    
                    # Аутентифікація
                    await self._authenticate(ws)
                    
                    # Підписка на топіки
                    await self._subscribe(ws)
                    
                    # Запускаємо heartbeat
                    heartbeat_task = asyncio.create_task(self._heartbeat(ws))
                    
                    # Обробка повідомлень
                    try:
                        await self._handle_messages(ws)
                    finally:
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass
                
            except asyncio.CancelledError:
                logger.info("[PRIVATE_WS] Cancelled")
                break
            except Exception as e:
                attempt += 1
                logger.error(f"❌ [PRIVATE_WS] Error (attempt {attempt}): {e}")
                
                if attempt >= max_attempts:
                    logger.error(f"[PRIVATE_WS] Max reconnect attempts reached, stopping")
                    break
                
                if self._running:
                    delay = min(self._reconnect_delay * (2 ** (attempt - 1)), 60)
                    logger.info(f"[PRIVATE_WS] Reconnecting in {delay}s...")
                    await asyncio.sleep(delay)
            
            finally:
                self.ws = None

    async def _authenticate(self, ws):
        """Аутентифікація через WebSocket"""
        try:
            auth_args = self.api.ws_auth_args()
            auth_msg = {
                "op": "auth",
                "args": auth_args
            }
            
            await ws.send(json.dumps(auth_msg))
            logger.debug(f"[PRIVATE_WS] Auth sent: {auth_args[0][:10]}...")
            
            # Чекаємо відповідь на auth
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(response)
            
            if data.get("success") or data.get("op") == "auth":
                logger.info("✅ [PRIVATE_WS] Authenticated successfully")
            else:
                raise Exception(f"Auth failed: {data}")
                
        except Exception as e:
            logger.error(f"❌ [PRIVATE_WS] Authentication error: {e}")
            raise

    async def _subscribe(self, ws):
        """Підписка на топіки"""
        try:
            topics = [
                "position",      # Оновлення позицій
                "execution",     # Execution events (TP/SL)
                "order",         # Оновлення ордерів
                "wallet"         # Баланс
            ]
            
            sub_msg = {
                "op": "subscribe",
                "args": topics
            }
            
            await ws.send(json.dumps(sub_msg))
            logger.info(f"✅ [PRIVATE_WS] Subscribed to: {', '.join(topics)}")
            
        except Exception as e:
            logger.error(f"❌ [PRIVATE_WS] Subscribe error: {e}")
            raise

    async def _heartbeat(self, ws):
        """Heartbeat (ping-pong)"""
        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                
                # Перевіряємо чи отримували pong
                if time.time() - self._last_pong > 60:
                    logger.warning("[PRIVATE_WS] No pong received for 60s, reconnecting...")
                    await ws.close()
                    break
                
                # Відправляємо ping
                ping_msg = {"op": "ping"}
                await ws.send(json.dumps(ping_msg))
                logger.debug("[PRIVATE_WS] Ping sent")
                
            except Exception as e:
                logger.error(f"[PRIVATE_WS] Heartbeat error: {e}")
                break

    async def _handle_messages(self, ws):
        """Обробка вхідних повідомлень"""
        async for message in ws:
            try:
                data = json.loads(message)
                
                # Pong
                if data.get("op") == "pong":
                    self._last_pong = time.time()
                    logger.debug("[PRIVATE_WS] Pong received")
                    continue
                
                # Subscription confirmation
                if data.get("op") == "subscribe":
                    logger.debug(f"[PRIVATE_WS] Subscription confirmed: {data.get('success')}")
                    continue
                
                # Data updates
                topic = data.get("topic", "")
                
                if topic.startswith("position"):
                    await self._handle_position(data)
                elif topic.startswith("execution"):
                    await self._handle_execution(data)
                elif topic.startswith("order"):
                    await self._handle_order(data)
                elif topic.startswith("wallet"):
                    await self._handle_wallet(data)
                else:
                    logger.debug(f"[PRIVATE_WS] Unknown topic: {topic}")
                
            except json.JSONDecodeError as e:
                logger.error(f"[PRIVATE_WS] JSON decode error: {e}")
            except Exception as e:
                logger.error(f"[PRIVATE_WS] Message handling error: {e}")

    async def _handle_position(self, data: dict):
        """Обробка position updates"""
        try:
            positions = data.get("data", [])
            for pos_data in positions:
                symbol = pos_data.get("symbol")
                if symbol in settings.pairs.trade_pairs:
                    await self.storage.update_position_from_exchange(pos_data)
                    logger.info(f"📊 [PRIVATE_WS] Position update: {symbol}")
        except Exception as e:
            logger.error(f"[PRIVATE_WS] Position handling error: {e}")

    async def _handle_execution(self, data: dict):
        """Покращена обробка execution events"""
        try:
            executions = data.get("data", [])
            for exec_data in executions:
                symbol = exec_data.get("symbol")
                
                # Негайна синхронізація з біржевими даними
                await self.storage.force_sync_positions(self.api)
                
                # Точне визначення причини через order matching
                await self.storage.apply_execution_event(exec_data)
                
        except Exception as e:
            logger.error(f"[PRIVATE_WS] Execution handling error: {e}")

    async def _handle_order(self, data: dict):
        """Обробка order updates"""
        try:
            orders = data.get("data", [])
            for order_data in orders:
                symbol = order_data.get("symbol")
                order_status = order_data.get("orderStatus")
                
                logger.debug(
                    f"📋 [PRIVATE_WS] Order update: {symbol} "
                    f"status={order_status} "
                    f"orderId={order_data.get('orderId')}"
                )
        except Exception as e:
            logger.error(f"[PRIVATE_WS] Order handling error: {e}")

    async def _handle_wallet(self, data: dict):
        """Обробка wallet updates"""
        try:
            wallets = data.get("data", [])
            for wallet_data in wallets:
                account_type = wallet_data.get("accountType")
                if account_type == "UNIFIED":
                    total_equity = float(wallet_data.get("totalEquity", 0))
                    logger.info(f"💰 [PRIVATE_WS] Wallet update: equity={total_equity:.2f} USDT")
        except Exception as e:
            logger.error(f"[PRIVATE_WS] Wallet handling error: {e}")