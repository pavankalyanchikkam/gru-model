"""
Configuration settings for PrognosAI
Centralized configuration for easy management
"""

import os
from pathlib import Path

# ============================================
# PATHS AND DIRECTORIES
# ============================================
BASE_DIR = Path(__file__).parent

# Data directories
MODELS_DIR = BASE_DIR / 'models'
TEST_DATA_DIR = BASE_DIR / 'test_data'
UPLOADS_DIR = BASE_DIR / 'uploads'
ASSETS_DIR = BASE_DIR / 'assets'

# Create directories
for directory in [MODELS_DIR, TEST_DATA_DIR, UPLOADS_DIR, ASSETS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================
# DATASET CONFIGURATIONS
# ============================================
DATASET_CONFIGS = {
    'FD001': {
        'name': 'FD001',
        'full_name': 'Single Operating Condition, Single Fault Mode',
        'seq_len': 30,
        'rul_clip': 125,
        'sensor_cols': [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
        'op_cols': [1, 2, 3],
        'training_engines': 100,
        'test_engines': 100,
        'complexity': 'Low',
        'description': 'Simplest dataset for basic prediction scenarios'
    },
    'FD002': {
        'name': 'FD002',
        'full_name': 'Multiple Operating Conditions, Single Fault Mode',
        'seq_len': 30,
        'rul_clip': 125,
        'sensor_cols': [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
        'op_cols': [1, 2, 3],
        'training_engines': 260,
        'test_engines': 259,
        'complexity': 'Medium',
        'description': 'Varying operating conditions with single fault mode'
    },
    'FD003': {
        'name': 'FD003',
        'full_name': 'Single Operating Condition, Multiple Fault Modes',
        'seq_len': 30,
        'rul_clip': 125,
        'sensor_cols': [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
        'op_cols': [1, 2, 3],
        'training_engines': 100,
        'test_engines': 100,
        'complexity': 'Medium',
        'description': 'Multiple fault types under consistent conditions'
    },
    'FD004': {
        'name': 'FD004',
        'full_name': 'Multiple Operating Conditions, Multiple Fault Modes',
        'seq_len': 30,
        'rul_clip': 125,
        'sensor_cols': [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
        'op_cols': [1, 2, 3],
        'training_engines': 249,
        'test_engines': 248,
        'complexity': 'High',
        'description': 'Most complex scenario with varying conditions and faults'
    }
}

# ============================================
# APPLICATION SETTINGS
# ============================================
# Default thresholds
DEFAULT_WARNING_THRESHOLD = 50
DEFAULT_CRITICAL_THRESHOLD = 20

# Display settings
PAGE_SIZE = 10  # Results per page in tables
CHART_HEIGHT = 400
MAX_FILE_SIZE_MB = 50

# Performance settings (optimized for i3/8GB)
MAX_MEMORY_USAGE_PERCENT = 60
CACHE_SIZE_LIMIT_MB = 500
SIMULATION_MODE = True  # Use simulation when models not available

# ============================================
# VISUALIZATION SETTINGS
# ============================================
COLOR_SCHEME = {
    'critical': '#dc2626',
    'warning': '#f59e0b',
    'normal': '#10b981',
    'info': '#3b82f6',
    'secondary': '#8b5cf6',
    'background': '#f8fafc',
    'text_primary': '#1f2937',
    'text_secondary': '#6b7280'
}

CHART_THEME = 'plotly_white'

# ============================================
# UTILITY FUNCTIONS
# ============================================
def get_feature_names():
    """Get standard CMAPSS feature names"""
    columns = ['unit', 'cycle']
    columns += [f'op_setting_{i}' for i in range(1, 4)]
    columns += [f'sensor_{i}' for i in range(1, 22)]
    return columns

def get_dataset_info(dataset_name):
    """Get information for a specific dataset"""
    return DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS['FD001'])

def get_all_dataset_names():
    """Get list of all available dataset names"""
    return list(DATASET_CONFIGS.keys())

def validate_thresholds(warning_thresh, critical_thresh):
    """Validate that thresholds are in correct order"""
    if critical_thresh >= warning_thresh:
        return False, "Critical threshold must be less than warning threshold"
    if warning_thresh <= 0 or critical_thresh <= 0:
        return False, "Thresholds must be positive values"
    return True, "Thresholds are valid"