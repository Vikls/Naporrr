import asyncio
import aiohttp
import time
from typing import Dict
from config.settings import settings
from utils.logger import logger

class Notifier:
    def __init__(self):
        self.token = settings.secrets.telegram_bot_token
        self.chat_id = settings.secrets.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)
        self._report_task = None
        self._last_report_trades = 0
        self._last_report_time = 0.0

    async def send(self, text: str):
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status != 200:
                        logger.warning(f"[TG] status={resp.status}")
        except Exception as e:
            logger.error(f"[TG] send error: {e}")

    def start_periodic_reports(self, executor_ref):
        if self._report_task:
            return
        self._report_task = asyncio.create_task(self._report_loop(executor_ref))

    async def _report_loop(self, executor_ref):
        interval = settings.logging.periodic_report_interval_sec
        self._last_report_time = time.time()
        while True:
            await asyncio.sleep(interval)
            try:
                stats = executor_ref.get_stats()
                # Умова: якщо 0 активності (жодної угоди і немає позицій) — пропускаємо
                if stats["total_trades"] == self._last_report_trades and stats["open_positions"] == 0:
                    continue
                self._last_report_trades = stats["total_trades"]
                await self.send(self._format_report(stats))
            except Exception as e:
                logger.error(f"[TG] periodic report error: {e}")

    def _format_report(self, stats: Dict) -> str:
        lines = [
            "<b>30m Report</b>",
            f"Trades total: {stats['total_trades']} (opens={stats['opens']} closes={stats['closes']})",
            f"Realized PnL: {stats['realized_pnl']:.2f}",
            f"Unrealized PnL: {stats['unrealized_pnl']:.2f}",
            f"Equity Δ vs start: {stats['equity_diff_vs_start']:.2f}",
        ]
        if stats["open_positions"] > 0:
            for p in stats["positions_details"]:
                lines.append(f"{p['symbol']} {p['side']} qty={p['qty']:.4f} ep={p['entry_price']:.2f} UPNL={p['upnl']:+.2f}")
        return "\n".join(lines)

notifier = Notifier()