import json
import os
import threading
from typing import Dict, Any, Optional, Tuple
from logging_config import logger
import time


class FileCache:
    """
    Thread-safe file cache that stores parsed JSON data to eliminate redundant file reads.
    Supports cache invalidation when files are modified.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._file_locks: Dict[str, threading.RLock] = {}
        self._cache_lock = threading.RLock()
        self._lock_creation_lock = threading.Lock()
        
    def _get_file_lock(self, file_path: str) -> threading.RLock:
        """Get or create a lock for a specific file."""
        with self._lock_creation_lock:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = threading.RLock()
            return self._file_locks[file_path]
    
    def _get_file_mtime(self, file_path: str) -> float:
        """Get file modification time."""
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return 0.0
    
    def get_cached_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached file data if it exists and is still valid.
        
        :param file_path: Path to the file
        :return: Cached data or None if not cached or invalid
        """
        abs_path = os.path.abspath(file_path)
        
        with self._cache_lock:
            if abs_path not in self._cache:
                return None
            
            cache_entry = self._cache[abs_path]
            current_mtime = self._get_file_mtime(abs_path)
            
            # Check if cached data is still valid
            if cache_entry['mtime'] >= current_mtime:
                logger.debug(f"Cache hit for {file_path}")
                return cache_entry['data']
            else:
                # Remove stale cache entry
                logger.debug(f"Cache expired for {file_path}")
                del self._cache[abs_path]
                return None
    
    def cache_file_data(self, file_path: str, data: Dict[str, Any]) -> None:
        """
        Cache parsed file data.
        
        :param file_path: Path to the file
        :param data: Parsed JSON data to cache
        """
        abs_path = os.path.abspath(file_path)
        current_mtime = self._get_file_mtime(abs_path)
        
        with self._cache_lock:
            self._cache[abs_path] = {
                'data': data,
                'mtime': current_mtime,
                'cached_at': time.time()
            }
            logger.debug(f"Cached data for {file_path}")
    
    def invalidate_cache(self, file_path: str) -> None:
        """
        Invalidate cache for a specific file.
        
        :param file_path: Path to the file
        """
        abs_path = os.path.abspath(file_path)
        
        with self._cache_lock:
            if abs_path in self._cache:
                del self._cache[abs_path]
                logger.debug(f"Invalidated cache for {file_path}")
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._cache_lock:
            self._cache.clear()
            logger.debug("Cleared all cache data")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                'cached_files': len(self._cache),
                'cache_size_mb': sum(
                    len(str(entry['data'])) for entry in self._cache.values()
                ) / (1024 * 1024)
            }
    
    def read_file_with_cache(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Read file with caching. Returns parsed JSON data by line.
        
        :param file_path: Path to the file
        :return: Dictionary with line numbers as keys and parsed JSON as values
        """
        file_lock = self._get_file_lock(file_path)
        
        with file_lock:
            # Check cache first
            cached_data = self.get_cached_data(file_path)
            if cached_data is not None:
                return cached_data
            
            # File not cached or expired, read from disk
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                
                # Parse each line as JSON
                parsed_data = {}
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        try:
                            parsed_data[i] = json.loads(line)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line {i+1} in {file_path}: {e}")
                            parsed_data[i] = None
                
                # Cache the parsed data
                self.cache_file_data(file_path, parsed_data)
                logger.debug(f"Read and cached {len(parsed_data)} lines from {file_path}")
                
                return parsed_data
            
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return None
    
    def write_file_with_cache_invalidation(self, file_path: str, lines: list) -> bool:
        """
        Write file and invalidate cache.
        
        :param file_path: Path to the file
        :param lines: List of lines to write
        :return: True if successful, False otherwise
        """
        file_lock = self._get_file_lock(file_path)
        
        with file_lock:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(lines)
                
                # Invalidate cache after write
                self.invalidate_cache(file_path)
                logger.debug(f"Wrote file and invalidated cache for {file_path}")
                
                return True
            
            except Exception as e:
                logger.error(f"Error writing file {file_path}: {e}")
                return False


# Global cache instance
_global_cache = FileCache()


def get_file_cache() -> FileCache:
    """Get the global file cache instance."""
    return _global_cache


def clear_global_cache():
    """Clear the global file cache."""
    _global_cache.clear_cache()


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _global_cache.get_cache_stats()