"""
Data Processor for CMAPSS Datasets
Optimized for memory efficiency on i3/8GB systems
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Process CMAPSS datasets with memory optimization"""
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.config = self._get_dataset_config(dataset_name)
        self.feature_names = self._get_feature_names()
        self.scaler = StandardScaler()
    
    def _get_dataset_config(self, dataset_name):
        """Get configuration for specific dataset"""
        configs = {
            'FD001': {'seq_len': 30, 'rul_clip': 125, 'sensors': [2,3,4,7,8,9,11,12,13,14,15,17,20,21]},
            'FD002': {'seq_len': 30, 'rul_clip': 125, 'sensors': [2,3,4,7,8,9,11,12,13,14,15,17,20,21]},
            'FD003': {'seq_len': 30, 'rul_clip': 125, 'sensors': [2,3,4,7,8,9,11,12,13,14,15,17,20,21]},
            'FD004': {'seq_len': 30, 'rul_clip': 125, 'sensors': [2,3,4,7,8,9,11,12,13,14,15,17,20,21]}
        }
        return configs.get(dataset_name, configs['FD001'])
    
    def _get_feature_names(self):
        """Get CMAPSS column names"""
        columns = ['unit', 'cycle']
        columns += [f'op_setting_{i}' for i in range(1, 4)]
        columns += [f'sensor_{i}' for i in range(1, 22)]
        return columns
    
    def load_data(self, file_path, max_rows=None):
        """
        Load data with memory optimization
        
        Args:
            file_path: Path to data file
            max_rows: Maximum rows to load (for memory control)
        
        Returns:
            DataFrame with processed data
        """
        try:
            # Load only specified number of rows for memory control
            if max_rows:
                df = pd.read_csv(
                    file_path, 
                    sep=r'\s+', 
                    header=None,
                    names=self.feature_names,
                    nrows=max_rows,
                    engine='python'
                )
            else:
                df = pd.read_csv(
                    file_path, 
                    sep=r'\s+', 
                    header=None,
                    names=self.feature_names,
                    engine='python'
                )
            
            # Add basic features (lightweight for memory)
            df = self._add_basic_features(df)
            
            print(f"✅ Loaded {len(df)} rows from {file_path}")
            print(f"   Unique units: {df['unit'].nunique()}")
            print(f"   Max cycles: {df['cycle'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
            raise
    
    def _add_basic_features(self, df):
        """Add lightweight features to save memory"""
        # Select only important sensors for feature engineering
        important_sensors = [f'sensor_{i}' for i in self.config['sensors'][:5]]
        
        # Add simple features (avoid heavy rolling windows)
        for sensor in important_sensors:
            # Simple differences
            df[f'{sensor}_diff'] = df.groupby('unit')[sensor].diff().fillna(0)
            
            # Simple normalization within unit
            df[f'{sensor}_norm'] = df.groupby('unit')[sensor].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        
        # Add cycle-based features
        df['cycle_ratio'] = df.groupby('unit')['cycle'].transform(
            lambda x: x / x.max()
        )
        
        return df
    
    def prepare_for_prediction(self, df, scaler=None):
        """
        Prepare data for prediction (simplified for simulation)
        
        Args:
            df: Input DataFrame
            scaler: Pre-fitted scaler (optional)
        
        Returns:
            Tuple of (sequences, unit_ids, feature_count)
        """
        # Get basic info
        n_units = df['unit'].nunique()
        feature_count = len([col for col in df.columns if 'sensor' in col or 'op_setting' in col])
        
        # For simulation, return placeholder values
        return None, list(range(1, n_units + 1)), feature_count
    
    def load_true_rul(self, file_path):
        """Load true RUL values from file"""
        try:
            rul_df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['RUL'])
            rul_df['RUL'] = rul_df['RUL'].clip(upper=self.config['rul_clip'])
            return rul_df['RUL'].values
        except Exception as e:
            print(f"⚠️ Could not load true RUL: {e}")
            return None
    
    def validate_file_format(self, file_path):
        """Validate CMAPSS file format"""
        try:
            # Read first few lines to validate
            test_df = pd.read_csv(file_path, sep=r'\s+', header=None, nrows=5)
            
            if test_df.shape[1] != 26:
                return False, f"Expected 26 columns, got {test_df.shape[1]}"
            
            # Check for numeric values
            if not np.issubdtype(test_df.values.dtype, np.number):
                return False, "File contains non-numeric values"
            
            return True, "File format is valid"
            
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def get_data_summary(self, df):
        """Get summary statistics of the data"""
        if df is None or df.empty:
            return {}
        
        summary = {
            'total_rows': len(df),
            'total_units': df['unit'].nunique(),
            'max_cycles': int(df['cycle'].max()),
            'min_cycles': int(df['cycle'].min()),
            'avg_cycles_per_unit': float(df.groupby('unit')['cycle'].max().mean()),
            'sensors_count': len([col for col in df.columns if 'sensor' in col]),
            'settings_count': len([col for col in df.columns if 'op_setting' in col]),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        return summary