#!/usr/bin/env python3
"""
Data Preprocessing Script for PhysioNet Challenge 2012

This script runs the complete preprocessing pipeline.

Author: Saahil Sanganeria
Date: October 2025

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --root-dir data/physionet
    python scripts/preprocess_data.py --no-download
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.preprocessing import main

if __name__ == '__main__':
    main()

