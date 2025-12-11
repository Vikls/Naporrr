"""
Backtest utility module for strategy testing with proper UTC timezone handling.
Ensures all timestamps are converted to UTC for accurate backtesting.
"""

import pytz
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import pandas as pd


class BacktestTimezoneHandler:
    """Handles timezone conversion for backtest data."""
    
    UTC = pytz.UTC
    
    @staticmethod
    def to_utc(dt: datetime, from_tz: Optional[str] = None) -> datetime:
        """
        Convert a datetime object to UTC.
        
        Args:
            dt: Datetime object to convert
            from_tz: Timezone string (e.g., 'US/Eastern'). If None, assumes naive datetime is UTC.
        
        Returns:
            Datetime object in UTC timezone
        """
        if dt is None:
            return None
        
        # If datetime is naive, assume it's UTC
        if dt.tzinfo is None:
            if from_tz:
                tz = pytz.timezone(from_tz)
                dt = tz.localize(dt)
                return dt.astimezone(pytz.UTC)
            else:
                return dt.replace(tzinfo=pytz.UTC)
        
        # If datetime is already aware, convert to UTC
        return dt.astimezone(pytz.UTC)
    
    @staticmethod
    def ensure_utc_series(series: pd.Series) -> pd.Series:
        """
        Convert all timestamps in a pandas Series to UTC.
        
        Args:
            series: Pandas Series with datetime objects
        
        Returns:
            Series with all datetimes converted to UTC
        """
        if not isinstance(series, pd.Series):
            raise ValueError("Input must be a pandas Series")
        
        return series.apply(lambda dt: BacktestTimezoneHandler.to_utc(dt) if pd.notna(dt) else dt)
    
    @staticmethod
    def ensure_utc_dataframe(df: pd.DataFrame, datetime_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert specified datetime columns in a DataFrame to UTC.
        
        Args:
            df: DataFrame to process
            datetime_columns: List of column names containing datetimes. If None, auto-detect.
        
        Returns:
            DataFrame with specified datetime columns converted to UTC
        """
        df_copy = df.copy()
        
        if datetime_columns is None:
            # Auto-detect datetime columns
            datetime_columns = df_copy.select_dtypes(include=['datetime64']).columns.tolist()
        
        for col in datetime_columns:
            if col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].dt.tz_convert(pytz.UTC)
                elif df_copy[col].dtype == 'object':
                    df_copy[col] = BacktestTimezoneHandler.ensure_utc_series(df_copy[col])
        
        return df_copy


class BacktestDataProcessor:
    """Processes backtest data with UTC timezone normalization."""
    
    def __init__(self):
        self.tz_handler = BacktestTimezoneHandler()
    
    def prepare_ohlcv_data(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        source_tz: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare OHLCV data for backtesting with UTC timestamps.
        
        Args:
            data: Raw OHLCV DataFrame
            timestamp_col: Name of the timestamp column
            source_tz: Source timezone of the data
        
        Returns:
            Processed DataFrame with UTC timestamps
        """
        df = data.copy()
        
        # Convert timestamp column to UTC
        if timestamp_col in df.columns:
            if source_tz:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col]).dt.tz_localize(source_tz).dt.tz_convert(pytz.UTC)
            else:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col]).dt.tz_convert(pytz.UTC) if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]) else pd.to_datetime(df[timestamp_col]).dt.tz_localize(pytz.UTC)
            
            # Set as index if needed
            df = df.sort_values(by=timestamp_col)
        
        return df
    
    def validate_utc_timestamps(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> bool:
        """
        Validate that all timestamps in the DataFrame are in UTC.
        
        Args:
            df: DataFrame to validate
            timestamp_col: Name of the timestamp column
        
        Returns:
            True if all timestamps are UTC, False otherwise
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")
        
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            return False
        
        # Check if timezone-aware
        if df[timestamp_col].dt.tz is None:
            return False
        
        # Check if timezone is UTC
        return df[timestamp_col].dt.tz == pytz.UTC


class BacktestSession:
    """Manages a backtest session with proper timezone handling."""
    
    def __init__(self, start_time: datetime, end_time: datetime, timezone: Optional[str] = None):
        """
        Initialize a backtest session.
        
        Args:
            start_time: Session start time
            end_time: Session end time
            timezone: Source timezone (if times are naive)
        """
        self.tz_handler = BacktestTimezoneHandler()
        self.processor = BacktestDataProcessor()
        
        # Convert times to UTC
        self.start_time = self.tz_handler.to_utc(start_time, timezone)
        self.end_time = self.tz_handler.to_utc(end_time, timezone)
        
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
    
    def filter_data_by_session(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Filter data to only include timestamps within the session range (in UTC).
        
        Args:
            data: DataFrame to filter
            timestamp_col: Name of the timestamp column
        
        Returns:
            Filtered DataFrame
        """
        df = data.copy()
        
        # Ensure timestamps are UTC
        if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            if df[timestamp_col].dt.tz is None:
                df[timestamp_col] = df[timestamp_col].dt.tz_localize(pytz.UTC)
            else:
                df[timestamp_col] = df[timestamp_col].dt.tz_convert(pytz.UTC)
        
        # Filter by session times
        mask = (df[timestamp_col] >= self.start_time) & (df[timestamp_col] <= self.end_time)
        return df[mask].reset_index(drop=True)
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get session information with UTC times.
        
        Returns:
            Dictionary containing session metadata
        """
        return {
            'start_time_utc': self.start_time.isoformat(),
            'end_time_utc': self.end_time.isoformat(),
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'timezone': 'UTC'
        }


def validate_backtest_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    timestamp_col: str = 'timestamp'
) -> Dict[str, Any]:
    """
    Validate backtest data for proper UTC timezone handling.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required columns
        timestamp_col: Name of the timestamp column
    
    Returns:
        Validation result dictionary
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(data.columns)
        if missing:
            results['valid'] = False
            results['errors'].append(f"Missing columns: {missing}")
    
    # Check timestamp column
    if timestamp_col not in data.columns:
        results['valid'] = False
        results['errors'].append(f"Timestamp column '{timestamp_col}' not found")
        return results
    
    # Check timezone
    if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
        results['warnings'].append(f"Timestamp column '{timestamp_col}' is not datetime type")
    else:
        if data[timestamp_col].dt.tz is None:
            results['warnings'].append("Timestamps are not timezone-aware. Assuming UTC.")
        elif data[timestamp_col].dt.tz != pytz.UTC:
            results['warnings'].append(f"Timestamps are in {data[timestamp_col].dt.tz}, not UTC")
    
    # Check for NaN timestamps
    nan_count = data[timestamp_col].isna().sum()
    if nan_count > 0:
        results['warnings'].append(f"Found {nan_count} NaN timestamps")
    
    return results
