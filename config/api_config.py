#!/usr/bin/env python3
"""
API Configuration Module

Centralizes API configuration and key management for the trading system.
Supports environment variables and secure key storage.
"""

import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class APIConfig:
    """Centralized API configuration management"""
    
    def __init__(self):
        self.polygon_api_key = self._get_polygon_key()
        self.validate_config()
    
    def _get_polygon_key(self) -> Optional[str]:
        """Get Polygon API key from environment or configuration"""
        # Priority order: environment variable, then fallback
        key = os.getenv('POLYGON_API_KEY')
        
        if not key:
            # Fallback to the key used in the original scripts
            key = "OWgBGzgOAzjd6Ieuml6iJakY1yA9npku"
            logger.warning("Using fallback API key. Set POLYGON_API_KEY environment variable for production.")
        
        return key
    
    def validate_config(self) -> None:
        """Validate that required configuration is present"""
        if not self.polygon_api_key:
            raise ValueError("Polygon API key is required. Set POLYGON_API_KEY environment variable.")
        
        if len(self.polygon_api_key) < 20:
            logger.warning("API key appears to be invalid (too short)")
    
    @property
    def polygon_base_url(self) -> str:
        """Polygon API base URL"""
        return "https://api.polygon.io"
    
    @property
    def polygon_headers(self) -> dict:
        """Headers for Polygon API requests"""
        return {
            'Authorization': f'Bearer {self.polygon_api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_polygon_url(self, endpoint: str) -> str:
        """Construct full Polygon API URL"""
        return f"{self.polygon_base_url}/{endpoint.lstrip('/')}"

# Global configuration instance
config = APIConfig()

# Convenience functions for backward compatibility
def get_polygon_api_key() -> str:
    """Get Polygon API key"""
    return config.polygon_api_key

def get_polygon_headers() -> dict:
    """Get Polygon API headers"""
    return config.polygon_headers

def get_polygon_url(endpoint: str) -> str:
    """Get full Polygon API URL"""
    return config.get_polygon_url(endpoint)