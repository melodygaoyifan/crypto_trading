"""
ZeroDiskRAMManager - Zero-Disk I/O Data Persistence
====================================================
Optimized for preloading 3-year tick datasets into 128GB RAM using shared memory.

Features:
- SharedMemory allocation for cross-process access
- Memory-mapped numpy arrays for zero-copy operations
- Atomic updates with version tracking
- Efficient columnar storage for OHLCV + features

Memory Layout (per asset, 3 years of 1m data = ~1.58M rows):
- timestamps: 1.58M × 8B = 12.6MB
- OHLCV: 1.58M × 5 × 8B = 63MB
- features: 1.58M × 72 × 4B = 455MB
- Total per asset: ~530MB
- 3 assets (BTC/ETH/SOL): ~1.6GB
- With buffers and indices: ~2GB

Hardware Target: MSI Titan 18 HX (128GB DDR5 RAM)
Author: Senior Performance Engineer
Version: 3.1
"""

import os
import sys
import time
import struct
import hashlib
import logging
import threading
import mmap
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Multiprocessing shared memory
from multiprocessing import shared_memory, Lock, RLock
from multiprocessing.managers import SharedMemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('ZeroDiskRAMManager')


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Data dimensions
MINUTES_PER_YEAR = 365 * 24 * 60  # 525,600
ROWS_3_YEARS = MINUTES_PER_YEAR * 3  # 1,576,800

# Column configuration
OHLCV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
OHLCV_DTYPE = np.float64
NUM_OHLCV_COLS = 6

# Feature dimensions (matches DRL state vector)
NUM_FEATURES = 72
FEATURE_DTYPE = np.float32

# Assets
SUPPORTED_ASSETS = ['BTC', 'ETH', 'SOL']


@dataclass
class MemoryConfig:
    """Configuration for memory allocation"""
    max_rows: int = ROWS_3_YEARS
    num_ohlcv_cols: int = NUM_OHLCV_COLS
    num_features: int = NUM_FEATURES
    ohlcv_dtype: np.dtype = np.dtype(np.float64)
    feature_dtype: np.dtype = np.dtype(np.float32)
    
    @property
    def ohlcv_bytes_per_row(self) -> int:
        return self.num_ohlcv_cols * self.ohlcv_dtype.itemsize
    
    @property
    def feature_bytes_per_row(self) -> int:
        return self.num_features * self.feature_dtype.itemsize
    
    @property
    def total_bytes_per_row(self) -> int:
        return self.ohlcv_bytes_per_row + self.feature_bytes_per_row
    
    @property
    def total_bytes_per_asset(self) -> int:
        # Header (64B) + OHLCV + Features
        header_size = 64
        return header_size + (self.max_rows * self.total_bytes_per_row)
    
    @property
    def total_bytes_all_assets(self) -> int:
        return len(SUPPORTED_ASSETS) * self.total_bytes_per_asset


# =============================================================================
# MEMORY SEGMENT HEADER
# =============================================================================

@dataclass
class SegmentHeader:
    """
    Header structure for each asset's memory segment.
    
    Layout (64 bytes):
    - magic: 4 bytes (b'HMAT')
    - version: 4 bytes (uint32)
    - asset_id: 8 bytes (padded string)
    - num_rows: 8 bytes (uint64)
    - write_index: 8 bytes (uint64) - circular buffer position
    - last_timestamp: 8 bytes (float64)
    - checksum: 16 bytes (MD5 of data)
    - reserved: 8 bytes
    """
    MAGIC = b'HMAT'
    SIZE = 64
    
    FORMAT = '<4sI8sQQd16s8s'  # Little-endian
    
    @classmethod
    def pack(
        cls,
        asset: str,
        version: int,
        num_rows: int,
        write_index: int,
        last_timestamp: float,
        checksum: bytes = b'\x00' * 16
    ) -> bytes:
        """Pack header into bytes"""
        asset_bytes = asset.encode('utf-8').ljust(8, b'\x00')[:8]
        return struct.pack(
            cls.FORMAT,
            cls.MAGIC,
            version,
            asset_bytes,
            num_rows,
            write_index,
            last_timestamp,
            checksum,
            b'\x00' * 8  # reserved
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> Dict[str, Any]:
        """Unpack header from bytes"""
        magic, version, asset_bytes, num_rows, write_index, last_ts, checksum, _ = \
            struct.unpack(cls.FORMAT, data[:cls.SIZE])
        
        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")
        
        return {
            'asset': asset_bytes.rstrip(b'\x00').decode('utf-8'),
            'version': version,
            'num_rows': num_rows,
            'write_index': write_index,
            'last_timestamp': last_ts,
            'checksum': checksum
        }


# =============================================================================
# SHARED MEMORY ARRAY WRAPPER
# =============================================================================

class SharedNumpyArray:
    """
    Wrapper for numpy array backed by shared memory.
    Provides zero-copy access across processes.
    """
    
    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        create: bool = True
    ):
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(shape)) * self.dtype.itemsize
        
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._array: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        
        if create:
            self._create()
        else:
            self._attach()
    
    def _create(self):
        """Create new shared memory segment"""
        try:
            # Try to unlink existing
            existing = shared_memory.SharedMemory(name=self.name)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass
        
        self._shm = shared_memory.SharedMemory(
            name=self.name,
            create=True,
            size=self.size
        )
        self._array = np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self._shm.buf
        )
        # Initialize to zeros
        self._array.fill(0)
        
        logger.debug(f"Created shared array '{self.name}': {self.shape} {self.dtype}")
    
    def _attach(self):
        """Attach to existing shared memory segment"""
        self._shm = shared_memory.SharedMemory(name=self.name)
        self._array = np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self._shm.buf
        )
        logger.debug(f"Attached to shared array '{self.name}'")
    
    @property
    def array(self) -> np.ndarray:
        """Get the numpy array"""
        return self._array
    
    def __getitem__(self, key):
        return self._array[key]
    
    def __setitem__(self, key, value):
        self._array[key] = value
    
    def close(self):
        """Close shared memory (does not unlink)"""
        if self._shm:
            self._shm.close()
    
    def unlink(self):
        """Unlink shared memory (delete)"""
        if self._shm:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


# =============================================================================
# ASSET DATA SEGMENT
# =============================================================================

class AssetDataSegment:
    """
    Memory segment for a single asset's data.
    
    Provides circular buffer for streaming updates
    and direct numpy access for batch reads.
    """
    
    def __init__(
        self,
        asset: str,
        config: MemoryConfig,
        create: bool = True
    ):
        self.asset = asset
        self.config = config
        
        # Shared memory segments
        self._header_shm: Optional[shared_memory.SharedMemory] = None
        self._ohlcv_array: Optional[SharedNumpyArray] = None
        self._feature_array: Optional[SharedNumpyArray] = None
        
        # Thread safety
        self._lock = threading.RLock()
        self._version = 0
        
        if create:
            self._create()
        else:
            self._attach()
    
    def _create(self):
        """Create memory segments for this asset"""
        # Header segment
        header_name = f"hmats_{self.asset}_header"
        try:
            existing = shared_memory.SharedMemory(name=header_name)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass
        
        self._header_shm = shared_memory.SharedMemory(
            name=header_name,
            create=True,
            size=SegmentHeader.SIZE
        )
        
        # Initialize header
        header_data = SegmentHeader.pack(
            asset=self.asset,
            version=0,
            num_rows=0,
            write_index=0,
            last_timestamp=0.0
        )
        self._header_shm.buf[:SegmentHeader.SIZE] = header_data
        
        # OHLCV array
        self._ohlcv_array = SharedNumpyArray(
            name=f"hmats_{self.asset}_ohlcv",
            shape=(self.config.max_rows, self.config.num_ohlcv_cols),
            dtype=self.config.ohlcv_dtype,
            create=True
        )
        
        # Feature array
        self._feature_array = SharedNumpyArray(
            name=f"hmats_{self.asset}_features",
            shape=(self.config.max_rows, self.config.num_features),
            dtype=self.config.feature_dtype,
            create=True
        )
        
        logger.info(f"Created data segment for {self.asset}: "
                    f"OHLCV={self._ohlcv_array.size/1e6:.1f}MB, "
                    f"Features={self._feature_array.size/1e6:.1f}MB")
    
    def _attach(self):
        """Attach to existing memory segments"""
        header_name = f"hmats_{self.asset}_header"
        self._header_shm = shared_memory.SharedMemory(name=header_name)
        
        self._ohlcv_array = SharedNumpyArray(
            name=f"hmats_{self.asset}_ohlcv",
            shape=(self.config.max_rows, self.config.num_ohlcv_cols),
            dtype=self.config.ohlcv_dtype,
            create=False
        )
        
        self._feature_array = SharedNumpyArray(
            name=f"hmats_{self.asset}_features",
            shape=(self.config.max_rows, self.config.num_features),
            dtype=self.config.feature_dtype,
            create=False
        )
        
        logger.info(f"Attached to data segment for {self.asset}")
    
    def _read_header(self) -> Dict[str, Any]:
        """Read current header"""
        return SegmentHeader.unpack(bytes(self._header_shm.buf[:SegmentHeader.SIZE]))
    
    def _write_header(
        self,
        num_rows: int = None,
        write_index: int = None,
        last_timestamp: float = None
    ):
        """Update header atomically"""
        with self._lock:
            current = self._read_header()
            
            self._version += 1
            header_data = SegmentHeader.pack(
                asset=self.asset,
                version=self._version,
                num_rows=num_rows if num_rows is not None else current['num_rows'],
                write_index=write_index if write_index is not None else current['write_index'],
                last_timestamp=last_timestamp if last_timestamp is not None else current['last_timestamp']
            )
            self._header_shm.buf[:SegmentHeader.SIZE] = header_data
    
    @property
    def ohlcv(self) -> np.ndarray:
        """Direct access to OHLCV array"""
        return self._ohlcv_array.array
    
    @property
    def features(self) -> np.ndarray:
        """Direct access to feature array"""
        return self._feature_array.array
    
    @property
    def num_rows(self) -> int:
        """Current number of valid rows"""
        return self._read_header()['num_rows']
    
    @property
    def write_index(self) -> int:
        """Current write position (circular)"""
        return self._read_header()['write_index']
    
    def bulk_load(
        self,
        ohlcv_data: np.ndarray,
        feature_data: Optional[np.ndarray] = None,
        start_index: int = 0
    ) -> int:
        """
        Bulk load data into memory.
        
        Args:
            ohlcv_data: (N, 6) array of OHLCV data
            feature_data: (N, 72) array of features (optional)
            start_index: Starting index in buffer
            
        Returns:
            Number of rows loaded
        """
        with self._lock:
            n_rows = len(ohlcv_data)
            end_index = start_index + n_rows
            
            if end_index > self.config.max_rows:
                logger.warning(f"Data exceeds buffer, truncating: {n_rows} -> {self.config.max_rows - start_index}")
                n_rows = self.config.max_rows - start_index
                end_index = self.config.max_rows
            
            # Copy OHLCV
            self._ohlcv_array[start_index:end_index] = ohlcv_data[:n_rows].astype(self.config.ohlcv_dtype)
            
            # Copy features if provided
            if feature_data is not None:
                self._feature_array[start_index:end_index] = feature_data[:n_rows].astype(self.config.feature_dtype)
            
            # Update header
            last_ts = float(ohlcv_data[-1, 0]) if n_rows > 0 else 0.0
            self._write_header(
                num_rows=end_index,
                write_index=end_index % self.config.max_rows,
                last_timestamp=last_ts
            )
            
            logger.info(f"{self.asset}: Loaded {n_rows:,} rows (total: {end_index:,})")
            return n_rows
    
    def append_row(
        self,
        timestamp: float,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        features: Optional[np.ndarray] = None
    ):
        """
        Append a single row (streaming update).
        Uses circular buffer.
        """
        with self._lock:
            header = self._read_header()
            idx = header['write_index']
            
            # Write OHLCV
            self._ohlcv_array[idx] = [timestamp, open_, high, low, close, volume]
            
            # Write features if provided
            if features is not None:
                self._feature_array[idx] = features
            
            # Update header (circular)
            new_num_rows = min(header['num_rows'] + 1, self.config.max_rows)
            new_write_idx = (idx + 1) % self.config.max_rows
            
            self._write_header(
                num_rows=new_num_rows,
                write_index=new_write_idx,
                last_timestamp=timestamp
            )
    
    def get_latest(self, n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latest N rows of data.
        
        Returns:
            (ohlcv, features) tuple
        """
        with self._lock:
            header = self._read_header()
            num_rows = header['num_rows']
            write_idx = header['write_index']
            
            if num_rows == 0:
                return np.array([]).reshape(0, 6), np.array([]).reshape(0, 72)
            
            n = min(n, num_rows)
            
            # Handle circular buffer
            if write_idx >= n:
                start = write_idx - n
                return (
                    self._ohlcv_array[start:write_idx].copy(),
                    self._feature_array[start:write_idx].copy()
                )
            else:
                # Wrap around
                part1_start = self.config.max_rows - (n - write_idx)
                ohlcv = np.vstack([
                    self._ohlcv_array[part1_start:],
                    self._ohlcv_array[:write_idx]
                ])
                features = np.vstack([
                    self._feature_array[part1_start:],
                    self._feature_array[:write_idx]
                ])
                return ohlcv, features
    
    def get_range(
        self,
        start_ts: float,
        end_ts: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data within timestamp range.
        
        Args:
            start_ts: Start timestamp (Unix)
            end_ts: End timestamp (Unix)
            
        Returns:
            (ohlcv, features) tuple
        """
        with self._lock:
            num_rows = self._read_header()['num_rows']
            if num_rows == 0:
                return np.array([]).reshape(0, 6), np.array([]).reshape(0, 72)
            
            timestamps = self._ohlcv_array[:num_rows, 0]
            mask = (timestamps >= start_ts) & (timestamps <= end_ts)
            
            return (
                self._ohlcv_array[:num_rows][mask].copy(),
                self._feature_array[:num_rows][mask].copy()
            )
    
    def close(self):
        """Close shared memory (does not unlink)"""
        if self._header_shm:
            self._header_shm.close()
        if self._ohlcv_array:
            self._ohlcv_array.close()
        if self._feature_array:
            self._feature_array.close()
    
    def unlink(self):
        """Unlink shared memory (delete)"""
        if self._header_shm:
            self._header_shm.close()
            try:
                self._header_shm.unlink()
            except Exception:
                pass
        if self._ohlcv_array:
            self._ohlcv_array.unlink()
        if self._feature_array:
            self._feature_array.unlink()


# =============================================================================
# ZERO-DISK RAM MANAGER
# =============================================================================

class ZeroDiskRAMManager:
    """
    Main manager for zero-disk I/O data persistence.
    
    Coordinates memory allocation and data loading across all assets.
    Provides unified interface for data access.
    """
    
    def __init__(
        self,
        config: MemoryConfig = None,
        assets: List[str] = None,
        create: bool = True
    ):
        self.config = config or MemoryConfig()
        self.assets = assets or SUPPORTED_ASSETS
        
        self._segments: Dict[str, AssetDataSegment] = {}
        self._lock = threading.Lock()
        self._initialized = False
        
        if create:
            self._initialize()
    
    def _initialize(self):
        """Initialize all asset segments"""
        logger.info(f"Initializing ZeroDiskRAMManager for {self.assets}")
        logger.info(f"Memory per asset: {self.config.total_bytes_per_asset / 1e6:.1f}MB")
        logger.info(f"Total memory: {self.config.total_bytes_all_assets / 1e6:.1f}MB")
        
        for asset in self.assets:
            self._segments[asset] = AssetDataSegment(
                asset=asset,
                config=self.config,
                create=True
            )
        
        self._initialized = True
        logger.info("ZeroDiskRAMManager initialized successfully")
    
    def attach(self):
        """Attach to existing shared memory (for child processes)"""
        for asset in self.assets:
            self._segments[asset] = AssetDataSegment(
                asset=asset,
                config=self.config,
                create=False
            )
        self._initialized = True
        logger.info("ZeroDiskRAMManager attached to existing segments")
    
    def preload_from_files(
        self,
        data_dir: str,
        file_pattern: str = "{asset}_1m.parquet"
    ) -> Dict[str, int]:
        """
        Preload data from parquet files.
        
        Args:
            data_dir: Directory containing data files
            file_pattern: File naming pattern (with {asset} placeholder)
            
        Returns:
            Dict of rows loaded per asset
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas required for file loading")
            return {}
        
        results = {}
        data_path = Path(data_dir)
        
        for asset in self.assets:
            filename = file_pattern.format(asset=asset)
            filepath = data_path / filename
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                results[asset] = 0
                continue
            
            logger.info(f"Loading {filepath}...")
            start_time = time.time()
            
            try:
                df = pd.read_parquet(filepath)
                
                # Ensure proper columns
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Missing columns in {filepath}")
                    results[asset] = 0
                    continue
                
                # Convert to numpy
                ohlcv_data = df[required_cols].values.astype(np.float64)
                
                # Load features if present
                feature_cols = [col for col in df.columns if col.startswith('feature_')]
                if len(feature_cols) == self.config.num_features:
                    feature_data = df[feature_cols].values.astype(np.float32)
                else:
                    feature_data = None
                
                # Bulk load
                rows = self._segments[asset].bulk_load(ohlcv_data, feature_data)
                results[asset] = rows
                
                elapsed = time.time() - start_time
                rate = rows / elapsed / 1000
                logger.info(f"{asset}: Loaded {rows:,} rows in {elapsed:.1f}s ({rate:.1f}k rows/s)")
                
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                results[asset] = 0
        
        return results
    
    def preload_from_exchange(
        self,
        exchange_client: Any,
        start_date: datetime,
        end_date: datetime = None,
        timeframe: str = '1m'
    ) -> Dict[str, int]:
        """
        Preload data directly from exchange API.
        
        Args:
            exchange_client: CCXT exchange instance
            start_date: Start date for data
            end_date: End date (default: now)
            timeframe: Candle timeframe
            
        Returns:
            Dict of rows loaded per asset
        """
        if end_date is None:
            end_date = datetime.utcnow()
        
        results = {}
        
        for asset in self.assets:
            symbol = f"{asset}/USDT"
            logger.info(f"Fetching {symbol} from {start_date} to {end_date}...")
            
            try:
                since = int(start_date.timestamp() * 1000)
                all_ohlcv = []
                
                while since < end_date.timestamp() * 1000:
                    ohlcv = exchange_client.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        since=since,
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 1
                    
                    if len(all_ohlcv) % 10000 == 0:
                        logger.info(f"{asset}: Fetched {len(all_ohlcv):,} candles...")
                
                if all_ohlcv:
                    # Convert to numpy
                    data = np.array(all_ohlcv, dtype=np.float64)
                    data[:, 0] /= 1000  # Convert ms to seconds
                    
                    rows = self._segments[asset].bulk_load(data)
                    results[asset] = rows
                else:
                    results[asset] = 0
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                results[asset] = 0
        
        return results
    
    def preload_synthetic(self, num_rows: int = ROWS_3_YEARS) -> Dict[str, int]:
        """
        Generate synthetic data for testing.
        
        Args:
            num_rows: Number of rows to generate
            
        Returns:
            Dict of rows loaded per asset
        """
        logger.info(f"Generating {num_rows:,} synthetic rows per asset...")
        
        base_prices = {'BTC': 97000.0, 'ETH': 3300.0, 'SOL': 195.0}
        results = {}
        
        for asset in self.assets:
            start_time = time.time()
            
            # Generate timestamps (1 minute intervals)
            now = time.time()
            timestamps = np.arange(now - num_rows * 60, now, 60, dtype=np.float64)[:num_rows]
            
            # Generate prices with random walk
            base = base_prices[asset]
            returns = np.random.randn(num_rows) * 0.001  # 0.1% volatility per minute
            prices = base * np.exp(np.cumsum(returns))
            
            # Generate OHLCV
            high = prices * (1 + np.abs(np.random.randn(num_rows)) * 0.0005)
            low = prices * (1 - np.abs(np.random.randn(num_rows)) * 0.0005)
            open_ = np.roll(prices, 1)
            open_[0] = prices[0]
            close = prices
            volume = np.random.exponential(1000 if asset == 'BTC' else 100, num_rows)
            
            ohlcv_data = np.column_stack([timestamps, open_, high, low, close, volume])
            
            # Generate synthetic features
            feature_data = np.random.randn(num_rows, self.config.num_features).astype(np.float32)
            
            # Load
            rows = self._segments[asset].bulk_load(ohlcv_data, feature_data)
            results[asset] = rows
            
            elapsed = time.time() - start_time
            rate = rows / elapsed / 1000
            logger.info(f"{asset}: Generated {rows:,} rows in {elapsed:.1f}s ({rate:.1f}k rows/s)")
        
        return results
    
    def get_segment(self, asset: str) -> AssetDataSegment:
        """Get segment for an asset"""
        return self._segments.get(asset)
    
    def get_latest(self, asset: str, n: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Get latest N rows for an asset"""
        segment = self._segments.get(asset)
        if segment:
            return segment.get_latest(n)
        return np.array([]).reshape(0, 6), np.array([]).reshape(0, 72)
    
    def append(
        self,
        asset: str,
        timestamp: float,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        features: Optional[np.ndarray] = None
    ):
        """Append a row to an asset's buffer"""
        segment = self._segments.get(asset)
        if segment:
            segment.append_row(timestamp, open_, high, low, close, volume, features)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            'total_memory_mb': self.config.total_bytes_all_assets / 1e6,
            'assets': {}
        }
        
        for asset, segment in self._segments.items():
            header = segment._read_header()
            stats['assets'][asset] = {
                'num_rows': header['num_rows'],
                'write_index': header['write_index'],
                'last_timestamp': header['last_timestamp'],
                'version': header['version'],
                'ohlcv_mb': segment._ohlcv_array.size / 1e6,
                'features_mb': segment._feature_array.size / 1e6
            }
        
        return stats
    
    def close(self):
        """Close all segments"""
        for segment in self._segments.values():
            segment.close()
        self._segments.clear()
        logger.info("ZeroDiskRAMManager closed")
    
    def unlink(self):
        """Unlink all shared memory"""
        for segment in self._segments.values():
            segment.unlink()
        self._segments.clear()
        logger.info("ZeroDiskRAMManager unlinked")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def preload_to_ram(
    data_dir: str = None,
    synthetic: bool = False,
    num_rows: int = ROWS_3_YEARS
) -> ZeroDiskRAMManager:
    """
    Convenience function to preload data into RAM.
    
    Args:
        data_dir: Directory with parquet files (if not synthetic)
        synthetic: Generate synthetic data for testing
        num_rows: Number of rows for synthetic data
        
    Returns:
        Initialized ZeroDiskRAMManager
    """
    manager = ZeroDiskRAMManager()
    
    if synthetic:
        manager.preload_synthetic(num_rows)
    elif data_dir:
        manager.preload_from_files(data_dir)
    
    return manager


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstration of ZeroDiskRAMManager"""
    print("="*70)
    print("ZeroDiskRAMManager Demo")
    print("="*70)
    
    # Create manager
    manager = ZeroDiskRAMManager()
    
    # Load synthetic data (smaller for demo)
    print("\n▶ Loading synthetic data (100k rows per asset)...")
    results = manager.preload_synthetic(num_rows=100000)
    
    for asset, rows in results.items():
        print(f"  {asset}: {rows:,} rows loaded")
    
    # Get stats
    print("\n▶ Memory Statistics:")
    stats = manager.get_stats()
    print(f"  Total memory: {stats['total_memory_mb']:.1f}MB")
    for asset, info in stats['assets'].items():
        print(f"  {asset}: {info['num_rows']:,} rows, "
              f"OHLCV={info['ohlcv_mb']:.1f}MB, "
              f"Features={info['features_mb']:.1f}MB")
    
    # Test retrieval
    print("\n▶ Testing data retrieval...")
    for asset in ['BTC', 'ETH', 'SOL']:
        ohlcv, features = manager.get_latest(asset, n=5)
        print(f"  {asset} latest 5 closes: {ohlcv[:, 4].round(2)}")
    
    # Test streaming append
    print("\n▶ Testing streaming append...")
    import time
    now = time.time()
    manager.append('BTC', now, 97500, 97550, 97480, 97520, 1234.5)
    ohlcv, _ = manager.get_latest('BTC', n=1)
    print(f"  Appended and retrieved: close={ohlcv[0, 4]:.2f}")
    
    # Cleanup
    print("\n▶ Cleanup...")
    manager.unlink()
    print("  Shared memory unlinked")
    
    print("\n✓ Demo complete")


if __name__ == '__main__':
    demo()
