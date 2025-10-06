#!/usr/bin/env python3
"""
SPX Early Warning System - Main Entry Point

Usage:
    python main.py train --features baseline currency volatility
    python main.py predict --date 2025-10-04
    python main.py analyze --start-date 2024-01-01 --end-date 2024-12-31
    python main.py info
"""

from cli import cli

if __name__ == "__main__":
    cli()

