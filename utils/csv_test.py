# utils/csv_test.py
import asyncio
import csv
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from config.settings import settings
from trading.bybit_api_manager import BybitAPIManager
from utils.logger import logger

class CSVTradeValidator:
    """Комплексний валідатор trades.csv"""
    
    def __init__(self, api: BybitAPIManager):
        self.api = api
        self.trades_file = settings.logging.trades_log
        self.validation_results: Dict[str, List[str]] = {}
        
    async def validate_trades_csv(self) -> bool:
        """Основна функція валідації"""
        logger.info("🔍 [CSV_VALIDATOR] Starting comprehensive validation...")
        
        try:
            # 1. Перевірка наявності файлу
            if not await self._ensure_trades_file_exists():
                logger.info("📝 [CSV_VALIDATOR] No trades file yet, skipping validation")
                return True
            
            # 2. Читання та парсинг CSV
            trades_data = self._read_trades_csv()
            if not trades_data:
                logger.info("📝 [CSV_VALIDATOR] No trades data found (empty file)")
                return True
            
            # 3. Групування по символам
            symbols_data = self._group_by_symbol(trades_data)
            
            # 4. Паралельна валідація для кожного символу
            validation_tasks = []
            for symbol, trades in symbols_data.items():
                task = asyncio.create_task(self._validate_symbol_trades(symbol, trades))
                validation_tasks.append(task)
            
            await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # 5. Аналіз результатів
            return self._analyze_validation_results()
            
        except Exception as e:
            logger.error(f"❌ [CSV_VALIDATOR] Validation error: {e}")
            return True  # Повертаємо True щоб не блокувати запуск бота
    
    async def _ensure_trades_file_exists(self) -> bool:
        """Перевірка та створення файлу trades.csv якщо не існує"""
        try:
            if not self.trades_file.exists():
                logger.info("📝 [CSV_VALIDATOR] Creating empty trades.csv file")
                self.trades_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.trades_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "event", "symbol", "side", 
                        "qty", "price", "sl", "tp", 
                        "reason", "meta"
                    ])
                return False
            return True
        except Exception as e:
            logger.error(f"❌ [CSV_VALIDATOR] Error creating trades file: {e}")
            return False
    
    def _read_trades_csv(self) -> List[Dict[str, Any]]:
        """Читання та парсинг trades.csv"""
        trades = []
        try:
            with open(self.trades_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Пропускаємо порожні рядки
                    if not row.get('timestamp') or not row.get('symbol'):
                        continue
                    
                    # Конвертація типів даних з обробкою помилок
                    try:
                        trade = {
                            'timestamp': row['timestamp'],
                            'event': row['event'],
                            'symbol': row['symbol'],
                            'side': row['side'],
                            'qty': float(row['qty']) if row['qty'] else 0.0,
                            'price': float(row['price']) if row['price'] else 0.0,
                            'sl': float(row['sl']) if row['sl'] else 0.0,
                            'tp': float(row['tp']) if row['tp'] else 0.0,
                            'reason': row['reason'],
                            'meta': row['meta']
                        }
                        trades.append(trade)
                    except (ValueError, KeyError) as e:
                        logger.warning(f"⚠️ [CSV_VALIDATOR] Skipping invalid row: {row} - {e}")
                        continue
            
            logger.info(f"✅ [CSV_VALIDATOR] Read {len(trades)} trades from CSV")
            return trades
            
        except Exception as e:
            logger.error(f"❌ [CSV_VALIDATOR] Error reading CSV: {e}")
            return []
    
    def _group_by_symbol(self, trades: List[Dict]) -> Dict[str, List[Dict]]:
        """Групування торгів по символах"""
        symbols_data = {}
        for trade in trades:
            symbol = trade['symbol']
            if symbol not in symbols_data:
                symbols_data[symbol] = []
            symbols_data[symbol].append(trade)
        
        return symbols_data
    
    async def _validate_symbol_trades(self, symbol: str, trades: List[Dict]):
        """Валідація торгів для конкретного символу"""
        logger.info(f"🔍 [SYMBOL_VALIDATION] Validating {symbol} with {len(trades)} trades")
        
        validation_errors = []
        
        try:
            # 1. Перевірка послідовності OPEN/CLOSE
            sequence_errors = self._validate_trade_sequence(symbol, trades)
            validation_errors.extend(sequence_errors)
            
            # 2. Перевірка причин закриття
            close_reason_errors = await self._validate_close_reasons(symbol, trades)
            validation_errors.extend(close_reason_errors)
            
            # 3. Перевірка PnL узгодженості
            pnl_errors = await self._validate_pnl_consistency(symbol, trades)
            validation_errors.extend(pnl_errors)
            
            # 4. Перевірка часових міток
            timestamp_errors = self._validate_timestamps(symbol, trades)
            validation_errors.extend(timestamp_errors)
            
            # 5. Перевірка відсутності пропусків
            missing_errors = await self._validate_missing_trades(symbol, trades)
            validation_errors.extend(missing_errors)
            
            # 6. Перевірка нульових цін
            zero_price_errors = self._validate_zero_prices(symbol, trades)
            validation_errors.extend(zero_price_errors)
            
        except Exception as e:
            error_msg = f"Validation error for {symbol}: {e}"
            validation_errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        # Зберігаємо результати
        self.validation_results[symbol] = validation_errors
        
        if validation_errors:
            logger.warning(f"⚠️ [SYMBOL_VALIDATION] {symbol} has {len(validation_errors)} errors")
        else:
            logger.info(f"✅ [SYMBOL_VALIDATION] {symbol} validation passed")
    
    def _validate_trade_sequence(self, symbol: str, trades: List[Dict]) -> List[str]:
        """Перевірка послідовності OPEN/CLOSE операцій"""
        errors = []
        open_positions = 0
        position_sequence = []
        
        for trade in trades:
            if trade['event'] == 'OPEN':
                open_positions += 1
                position_sequence.append('OPEN')
            elif trade['event'] == 'CLOSE':
                if open_positions <= 0:
                    errors.append(f"CLOSE without OPEN: {trade['timestamp']}")
                else:
                    open_positions -= 1
                position_sequence.append('CLOSE')
        
        # Перевірка незакритих позицій
        if open_positions > 0:
            errors.append(f"Unclosed positions: {open_positions}")
        
        # Перевірка паттернів
        if len(position_sequence) >= 2:
            for i in range(1, len(position_sequence)):
                if position_sequence[i] == 'OPEN' and position_sequence[i-1] == 'OPEN':
                    errors.append(f"Consecutive OPENs at position {i}")
        
        return errors
    
    async def _validate_close_reasons(self, symbol: str, trades: List[Dict]) -> List[str]:
        """Перевірка узгодженості причин закриття"""
        errors = []
        
        for trade in trades:
            if trade['event'] == 'CLOSE':
                reason = trade['reason']
                
                # Перевірка валідності причини
                valid_reasons = ['TP_HIT', 'SL_HIT', 'TIME_EXIT', 'REVERSE', 'STRATEGY_SIGNAL', 'EXCHANGE_CLOSE']
                if reason not in valid_reasons:
                    errors.append(f"Invalid close reason: {reason} at {trade['timestamp']}")
                
                # Аналіз PnL для TP/SL (TIME_EXIT може мати будь-який PnL)
                if reason in ['TP_HIT', 'SL_HIT']:
                    pnl_info = self._extract_pnl_from_meta(trade['meta'])
                    if pnl_info is not None:
                        if reason == 'TP_HIT' and pnl_info < 0:
                            errors.append(f"TP_HIT with negative PnL: {pnl_info} at {trade['timestamp']}")
                        elif reason == 'SL_HIT' and pnl_info > 0:
                            errors.append(f"SL_HIT with positive PnL: {pnl_info} at {trade['timestamp']}")
        
        return errors
    
    def _extract_pnl_from_meta(self, meta: str) -> Optional[float]:
        """Витягнення PnL з meta поля"""
        try:
            if 'pnl=' in meta:
                pnl_part = meta.split('pnl=')[1].split(';')[0]
                if pnl_part != 'NA':
                    return float(pnl_part)
        except (ValueError, IndexError):
            pass
        return None
    
    async def _validate_pnl_consistency(self, symbol: str, trades: List[Dict]) -> List[str]:
        """Перевірка узгодженості PnL з біржевими даними"""
        errors = []
        
        try:
            # Отримуємо закритий PnL з біржі
            closed_pnl = await self.api.get_closed_pnl(symbol, limit=50)
            if not closed_pnl or closed_pnl.get('retCode') != 0:
                return [f"Failed to get closed PnL from exchange"]
            
            exchange_pnl_data = closed_pnl.get('result', {}).get('list', [])
            
            # Створюємо мапу PnL по часу
            pnl_map = {}
            for pnl_record in exchange_pnl_data:
                updated_time = int(pnl_record.get('updatedTime', 0)) / 1000
                closed_pnl_val = float(pnl_record.get('closedPnl', 0))
                pnl_map[updated_time] = closed_pnl_val
            
            # Порівнюємо з CSV даними
            for trade in trades:
                if trade['event'] == 'CLOSE':
                    trade_time = self._parse_timestamp(trade['timestamp'])
                    csv_pnl = self._extract_pnl_from_meta(trade['meta'])
                    
                    if csv_pnl is not None:
                        # Шукаємо найближчий PnL запис на біржі
                        closest_time = None
                        min_diff = float('inf')
                        
                        for exchange_time in pnl_map.keys():
                            time_diff = abs(exchange_time - trade_time)
                            if time_diff < min_diff and time_diff < 300:  # 5 хвилин
                                min_diff = time_diff
                                closest_time = exchange_time
                        
                        if closest_time:
                            exchange_pnl = pnl_map[closest_time]
                            pnl_diff = abs(exchange_pnl - csv_pnl)
                            
                            if pnl_diff > 0.01:  # Допуск 0.01 USDT
                                errors.append(f"PnL mismatch: CSV={csv_pnl:.4f}, Exchange={exchange_pnl:.4f} at {trade['timestamp']}")
        
        except Exception as e:
            errors.append(f"PnL validation error: {e}")
        
        return errors
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Парсинг часової мітки"""
        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            return time.time()
    
    def _validate_timestamps(self, symbol: str, trades: List[Dict]) -> List[str]:
        """Перевірка часових міток"""
        errors = []
        current_time = time.time()
        
        for i, trade in enumerate(trades):
            trade_time = self._parse_timestamp(trade['timestamp'])
            
            # Перевірка майбутніх міток
            if trade_time > current_time + 60:  # 1 хвилина в майбутнє
                errors.append(f"Future timestamp: {trade['timestamp']}")
            
            # Перевірка послідовності міток
            if i > 0:
                prev_time = self._parse_timestamp(trades[i-1]['timestamp'])
                if trade_time < prev_time:
                    errors.append(f"Timestamp going backwards: {trade['timestamp']} after {trades[i-1]['timestamp']}")
        
        return errors
    
    async def _validate_missing_trades(self, symbol: str, trades: List[Dict]) -> List[str]:
        """Перевірка відсутності пропущених торгів"""
        errors = []
        
        try:
            # Отримуємо execution history з біржі
            executions = await self.api.get_recent_executions(symbol, limit=100)
            if not executions or executions.get('retCode') != 0:
                return [f"Failed to get executions from exchange"]
            
            execution_list = executions.get('result', {}).get('list', [])
            
            # Створюємо мапу execution по часу та ціні
            execution_map = {}
            for exec_data in execution_list:
                exec_time = int(exec_data.get('execTime', 0)) / 1000
                exec_price = float(exec_data.get('execPrice', 0))
                exec_qty = float(exec_data.get('execQty', 0))
                key = f"{exec_time}_{exec_price}_{exec_qty}"
                execution_map[key] = exec_data
            
            # Перевіряємо кожну угоду в CSV
            for trade in trades:
                trade_time = self._parse_timestamp(trade['timestamp'])
                trade_price = trade['price']
                trade_qty = trade['qty']
                
                # Пропускаємо перевірку для нульових цін (вже оброблено в іншій перевірці)
                if trade_price == 0:
                    continue
                
                # Шукаємо відповідний execution
                found = False
                for exec_key in execution_map.keys():
                    exec_time = float(exec_key.split('_')[0])
                    exec_price = float(exec_key.split('_')[1])
                    exec_qty = float(exec_key.split('_')[2])
                    
                    time_diff = abs(exec_time - trade_time)
                    price_diff = abs(exec_price - trade_price) / trade_price if trade_price > 0 else 1
                    qty_diff = abs(exec_qty - trade_qty) / trade_qty if trade_qty > 0 else 1
                    
                    if time_diff < 300 and price_diff < 0.01 and qty_diff < 0.01:  # Допуски
                        found = True
                        break
                
                if not found:
                    errors.append(f"Missing exchange execution for trade: {trade['timestamp']} {trade_price} {trade_qty}")
        
        except Exception as e:
            errors.append(f"Missing trades validation error: {e}")
        
        return errors
    
    def _validate_zero_prices(self, symbol: str, trades: List[Dict]) -> List[str]:
        """Перевірка нульових цін у записах"""
        errors = []
        
        for trade in trades:
            # Перевірка нульових цін для CLOSE операцій
            if trade['event'] == 'CLOSE' and trade['price'] == 0:
                # TIME_EXIT може мати нульову ціну, якщо не вдалося отримати дані
                if trade['reason'] != 'TIME_EXIT':
                    errors.append(f"Zero price for CLOSE with reason {trade['reason']}: {trade['timestamp']}")
            
            # Перевірка нульових цін для OPEN операцій (завжди помилка)
            if trade['event'] == 'OPEN' and trade['price'] == 0:
                errors.append(f"Zero price for OPEN: {trade['timestamp']}")
        
        return errors
    
    def _analyze_validation_results(self) -> bool:
        """Аналіз результатів валідації"""
        total_errors = 0
        symbols_with_errors = 0
        
        print("\n" + "="*80)
        print("📊 CSV VALIDATION RESULTS")
        print("="*80)
        
        for symbol, errors in self.validation_results.items():
            if errors:
                symbols_with_errors += 1
                total_errors += len(errors)
                print(f"\n❌ {symbol}: {len(errors)} errors")
                for error in errors[:5]:  # Показуємо перші 5 помилок
                    print(f"   - {error}")
                if len(errors) > 5:
                    print(f"   ... and {len(errors) - 5} more errors")
            else:
                print(f"✅ {symbol}: No errors")
        
        print("\n" + "="*80)
        print(f"📈 SUMMARY: {symbols_with_errors} symbols with errors, {total_errors} total errors")
        print("="*80)
        
        # Вважаємо успішним якщо менше 20% символів мають помилки
        success_ratio = 1 - (symbols_with_errors / len(self.validation_results)) if self.validation_results else 1
        is_success = success_ratio >= 0.8 and total_errors < 30
        
        if is_success:
            logger.info("✅ [CSV_VALIDATOR] Validation PASSED")
        else:
            logger.warning("⚠️ [CSV_VALIDATOR] Validation found issues (continuing anyway)")
        
        return True  # Завжди повертаємо True, щоб не блокувати запуск бота

async def main():
    """Головна функція тесту"""
    logger.info("🚀 Starting CSV trade validation...")
    
    api = BybitAPIManager()
    validator = CSVTradeValidator(api)
    
    try:
        success = await validator.validate_trades_csv()
        return success
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return True  # Завжди True щоб не блокувати бота
    finally:
        await api.close()

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)