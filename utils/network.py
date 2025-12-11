import time
import aiohttp
from utils.logger import logger

async def check_network_quality():
    """Перевірка якості мережевого з'єднання"""
    sites = [
        "https://api.bybit.com",
        "https://www.google.com",
        "https://www.cloudflare.com"
    ]
    
    for site in sites:
        try:
            async with aiohttp.ClientSession() as session:
                start = time.time()
                async with session.get(site, timeout=5) as response:
                    latency = (time.time() - start) * 1000
                    if latency > 1000:  # 1 second
                        logger.warning(f"High latency to {site}: {latency:.0f}ms")
        except Exception as e:
            logger.error(f"Network check failed for {site}: {e}")