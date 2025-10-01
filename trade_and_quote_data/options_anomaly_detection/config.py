#!/usr/bin/env python3
"""
Options Chain Downloader Configuration

Configuration management for the options chain downloader system.
Integrates with the main config module for API key management.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import get_polygon_api_key, get_polygon_headers, get_polygon_url
except ImportError:
    # Fallback if config module not available
    def get_polygon_api_key() -> str:
        return os.getenv('POLYGON_API_KEY', '')
    
    def get_polygon_headers() -> dict:
        return {'Content-Type': 'application/json'}
    
    def get_polygon_url(endpoint: str) -> str:
        return f"https://api.polygon.io/{endpoint.lstrip('/')}"

logger = logging.getLogger(__name__)

class OptionsChainConfig:
    """Configuration for options chain downloader"""
    
    def __init__(self):
        # API Configuration
        self.polygon_api_key = get_polygon_api_key()
        self.polygon_base_url = "https://api.polygon.io"
        
        # Download Configuration
        self.max_workers = 20  # Parallel download workers
        self.request_delay = 0.12  # Rate limiting delay (5000 req/min)
        self.timeout = 30  # Request timeout in seconds
        self.retry_attempts = 3  # Number of retry attempts
        
        # Storage Configuration
        self.data_dir = Path("data")  # Default data directory
        self.cache_size_limit = 100  # Number of chains to cache
        
        # Risk-free rate for options pricing
        self.risk_free_rate = 0.05  # 5% annual risk-free rate
        
        # Trading day configuration
        self.market_holidays = [
            '2024-01-01', '2024-01-15', '2024-02-19', '2024-05-27', '2024-07-04', 
            '2024-09-02', '2024-11-28', '2024-12-25',
            '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', '2025-07-04', 
            '2025-09-01', '2025-11-27', '2025-12-25',
            '2026-01-01', '2026-01-19', '2026-02-16', '2026-05-25', '2026-07-03', 
            '2026-09-07', '2026-11-26', '2026-12-25'
        ]
        
        # Validate configuration
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration settings"""
        if not self.polygon_api_key:
            logger.warning("No Polygon API key found. Set POLYGON_API_KEY environment variable.")
        
        if self.max_workers > 50:
            logger.warning("High number of workers may cause rate limiting issues")
        
        if self.request_delay < 0.012:  # Less than 100ms for 10k req/min limit
            logger.warning("Request delay may be too low for API rate limits")
    
    def get_rate_limit_delay(self, api_tier: str = "professional") -> float:
        """Get appropriate delay based on API tier"""
        delays = {
            "free": 12.0,        # 5 req/min
            "basic": 0.6,        # 100 req/min
            "professional": 0.12, # 5000 req/min
            "unlimited": 0.01    # 10000+ req/min
        }
        return delays.get(api_tier.lower(), self.request_delay)
    
    def set_data_directory(self, path: str) -> None:
        """Set custom data directory"""
        self.data_dir = Path(path)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_polygon_headers(self) -> Dict[str, str]:
        """Get headers for Polygon API requests"""
        return get_polygon_headers()
    
    def get_polygon_url(self, endpoint: str) -> str:
        """Get full Polygon API URL"""
        return get_polygon_url(endpoint)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'polygon_api_key': '***' if self.polygon_api_key else None,
            'max_workers': self.max_workers,
            'request_delay': self.request_delay,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'data_dir': str(self.data_dir),
            'cache_size_limit': self.cache_size_limit,
            'risk_free_rate': self.risk_free_rate
        }

# Global configuration instance
options_config = OptionsChainConfig()

# Convenience functions
def get_options_config() -> OptionsChainConfig:
    """Get the global options configuration instance"""
    return options_config

def update_config(**kwargs) -> None:
    """Update configuration parameters"""
    for key, value in kwargs.items():
        if hasattr(options_config, key):
            setattr(options_config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")